import os
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as L
from torch.utils.data import DataLoader, Sampler
import random

from TTSDatasetModule import (
    MyDataset,
    collate_fn,
    masked_mse_loss,
    masked_latent_loss,
    masked_smooth_l1_loss,
)
from trans_encoder import XTEncoder
from ssm_decoder import decoder
from duration_predictor import duration
from TTS_model import Model

torch.set_float32_matmul_precision("high")


class BucketBatchSampler(Sampler):
    """
    Bucket Batch Sampler: Fast bucket sorting by sequence length in memory.
    Eliminates blocking filesystem syscalls during dataloader creation.
    """
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Instant in-memory length sorting (< 2ms for 40,000 samples)
        text_col = None
        for col in ['normalized_text', 'normalized', 'text', 'transcript', 'txt']:
            if col in dataset.df.columns:
                text_col = col
                break
        if text_col is not None:
            text_lens = dataset.df[text_col].fillna('').astype(str).str.len().to_numpy()
            self.indices = np.argsort(text_lens).tolist()
        else:
            self.indices = list(range(len(dataset)))

    def __iter__(self):
        batches = [self.indices[i:i + self.batch_size] for i in range(0, len(self.indices), self.batch_size)]
        if self.drop_last and len(batches) > 0 and len(batches[-1]) < self.batch_size:
            batches.pop()
            
        if self.shuffle:
            random.shuffle(batches)
            
        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.indices) // self.batch_size
        else:
            return (len(self.indices) + self.batch_size - 1) // self.batch_size


class TTSDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_file="/home/monesh/ljspeech/LJSpeech-1.1/train.csv",
        val_file="/home/monesh/ljspeech/LJSpeech-1.1/val.csv",
        batch_size=16,
        num_workers=4,
        prefetch_data=True,
    ):
        super().__init__()
        self.train_csv = train_file
        self.val_csv = val_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_data = prefetch_data

    def setup(self, stage=None):
        self.train_dataset = MyDataset(self.train_csv, prefetch_data=self.prefetch_data)
        self.val_dataset = MyDataset(self.val_csv, prefetch_data=self.prefetch_data)

    def train_dataloader(self):
        sampler = BucketBatchSampler(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        return DataLoader(
            self.train_dataset, 
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
    
    def val_dataloader(self):
        sampler = BucketBatchSampler(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        return DataLoader(
            self.val_dataset, 
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )


class WarmupReduceLROnPlateau:
    """
    Linear Warmup followed by ReduceLROnPlateau scheduler.
    """
    def __init__(
        self,
        optimizer,
        warmup_epochs=2,
        initial_lr=1e-5,
        mode="min",
        factor=0.5,
        patience=12,
        cooldown=2,
        min_lr=1e-5,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.plateau = ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, cooldown=cooldown, min_lr=min_lr
        )
        self.current_epoch = 0
        self.base_lrs = [param_group["lr"] for param_group in self.optimizer.param_groups]
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr

    def step(self, metrics=None):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            progress = self.current_epoch / float(self.warmup_epochs)
            for i, param_group in enumerate(self.optimizer.param_groups):
                base_lr = self.base_lrs[i]
                param_group["lr"] = self.initial_lr + progress * (base_lr - self.initial_lr)
        else:
            if metrics is not None:
                self.plateau.step(metrics)

    def state_dict(self):
        return {
            "current_epoch": self.current_epoch,
            "base_lrs": self.base_lrs,
            "plateau": self.plateau.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict["current_epoch"]
        self.plateau.load_state_dict(state_dict["plateau"])
        self.base_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        self.plateau._last_lr = [pg["lr"] for pg in self.optimizer.param_groups]


class MambaFlowTTS(L.LightningModule):
    """
    Production-Grade MambaFlow-TTS PyTorch Lightning Module:
    - Balanced multi-task loss: 1.0 * Prior Latent + 1.0 * CFM + 0.1 * Duration Predictor
    - CFM-only fine-tuning mode with Cosine Annealing scheduler (1e-5 -> 1e-6 over 100 epochs)
    - Latent Loss: 1.0 * Masked MSE + 0.1 * Covariance Matching Loss
    - Duration Loss: Masked Smooth L1 Loss (beta=0.5) on log(duration + 1)
    """
    def __init__(
        self,
        lambda_latent=1.0,
        lambda_cfm=1.0,
        lambda_dur_pred=0.1,
        w_latent_mse=1.0,
        w_latent_cov=0.1,
        d_enc=256,
        d_dec=384,
        d_codec=100,
        gradient_checkpointing=False,
        train_decoder_only=False,
        freeze_latent_only=False,
        lr=1e-5,
        max_epochs=100,
        model=None,
    ):
        super().__init__()
        if model is None:
            self.model = Model(
                encoder=XTEncoder(dim=d_enc, depth=6, heads=6, gradient_checkpointing=gradient_checkpointing),
                duration_module=duration(dims=d_enc, hidden_dim=256, target_dim=d_codec),
                decoder_module=decoder(
                    d_in=d_codec,
                    d_cond=d_codec,
                    d_out=d_codec,
                    d_model=d_dec,
                    n_layers=6,
                    gradient_checkpointing=gradient_checkpointing,
                ),
                d_enc=d_enc,
                d_codec=d_codec,
                d_dec=d_dec,
                gradient_checkpointing=gradient_checkpointing,
            )
        else:
            self.model = model

        self.model.gradient_checkpointing = gradient_checkpointing
        self.train_decoder_only = train_decoder_only
        self.freeze_latent_only = freeze_latent_only
        self.lr = lr
        self.max_epochs = max_epochs

        if self.train_decoder_only:
            # Freeze non-decoder modules
            self.model.txt_emb.requires_grad_(False)
            self.model.encoder.requires_grad_(False)
            self.model.duration.requires_grad_(False)
            self.model.to_latent.requires_grad_(False)
            self.model.decoder.requires_grad_(True)
            self.lambda_latent = 0.0
            self.lambda_dur_pred = 0.0
            self.lambda_cfm = 1.0
        elif self.freeze_latent_only:
            # Freeze latent modules, allow training duration and CFM decoder
            self.model.txt_emb.requires_grad_(False)
            self.model.encoder.requires_grad_(False)
            self.model.to_latent.requires_grad_(False)
            self.model.duration.requires_grad_(True)
            self.model.decoder.requires_grad_(True)
            self.lambda_latent = 0.0  # Latent is frozen, don't update from its loss
            self.lambda_cfm = lambda_cfm
            self.lambda_dur_pred = lambda_dur_pred
        else:
            self.lambda_latent = lambda_latent
            self.lambda_cfm = lambda_cfm
            self.lambda_dur_pred = lambda_dur_pred

        self.w_latent_mse = w_latent_mse
        self.w_latent_cov = w_latent_cov
        self.save_hyperparameters(ignore=["model"])
        self.strict_loading = True

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=1e-2,
            betas=(0.9, 0.98),
            eps=1e-8,
        )

        # Cosine Annealing without warmup from initial_lr (1e-5) down to 1e-6 over 100 epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def lr_scheduler_step(self, scheduler, metric):
        if isinstance(scheduler, (ReduceLROnPlateau, WarmupReduceLROnPlateau)):
            if metric is None:
                val_loss = self.trainer.callback_metrics.get("val_loss", None)
                if val_loss is not None:
                    val_val = val_loss.item() if isinstance(val_loss, torch.Tensor) else float(val_loss)
                    scheduler.step(val_val)
                else:
                    scheduler.step()
            else:
                val_val = metric.item() if isinstance(metric, torch.Tensor) else float(metric)
                scheduler.step(val_val)
        else:
            scheduler.step()

    def _shared_step(self, batch):
        text = batch['ipa']
        
        text_lengths = batch['ipa_lengths']
        audio_lengths = batch['lengths']
        
        N_text = text.size(1)
        text_mask = torch.arange(N_text, device=self.device)[None, :] < text_lengths[:, None]
        
        target_mel = batch['mel']
        N_audio = target_mel.size(1)
        audio_mask = torch.arange(N_audio, device=self.device)[None, :] < audio_lengths[:, None]

        # Model forward
        out_tuple = self.model(
            x=text,
            target_latent=target_mel,
            mask=text_mask,
            audio_mask=audio_mask,
        )
        
        out, latent, alignment, dur_pred, mas_score = out_tuple

        # 1. Masked Latent Loss (Smooth L1 against normalized 100-dim mel target, beta=0.5)
        loss_latent = masked_smooth_l1_loss(
            latent,
            target_mel,
            audio_mask,
            beta=0.5,
        )

        # 2. CFM Velocity MSE loss on normalized 100-dim mel target
        if out is not None and isinstance(out, (tuple, list)) and len(out) == 2:
            v_pred, u_target = out
            loss_cfm = masked_mse_loss(v_pred, u_target, audio_mask)
        else:
            loss_cfm = torch.tensor(0.0, device=self.device)

        # 3. Duration Predictor Loss (Masked Smooth L1 in log-domain, beta=0.5)
        target_log_dur = torch.log(alignment.sum(dim=-1).detach().float() + 1.0)
        loss_dur_pred = masked_smooth_l1_loss(dur_pred.float(), target_log_dur, text_mask, beta=0.5)

        # Total multi-task weighted loss
        total_loss = (
            self.lambda_latent * loss_latent +
            self.lambda_cfm * loss_cfm +
            self.lambda_dur_pred * loss_dur_pred
        )

        losses = {
            "latent": loss_latent,
            "cfm": loss_cfm,
            "dur_pred": loss_dur_pred,
        }

        return total_loss, losses

    def training_step(self, batch, batch_idx):
        loss, losses = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss_latent", losses["latent"], prog_bar=False, on_step=True, on_epoch=True)
        self.log("train_loss_cfm", losses["cfm"], prog_bar=False, on_step=True, on_epoch=True)
        self.log("train_loss_dur_pred", losses["dur_pred"], prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, losses = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_loss_latent", losses["latent"], prog_bar=False, on_epoch=True)
        self.log("val_loss_cfm", losses["cfm"], prog_bar=False, on_epoch=True)
        self.log("val_loss_dur_pred", losses["dur_pred"], prog_bar=False, on_epoch=True)
        return loss


# Backward compatibility alias
TTSMODEL = MambaFlowTTS
