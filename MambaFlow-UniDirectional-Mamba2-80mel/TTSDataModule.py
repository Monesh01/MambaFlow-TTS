import os
import sys
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..")))
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
from TTS_model import MambaFlowTTSModel

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
        train_file="/home/monesh/ljspeech/LJSpeech-1.1/train_hifigan.csv",
        val_file="/home/monesh/ljspeech/LJSpeech-1.1/val_hifigan.csv",
        batch_size=16,
        num_workers=4,
        prefetch_data=True,
        stats_path=None,
    ):
        super().__init__()
        self.train_csv = train_file
        self.val_csv = val_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_data = prefetch_data
        if stats_path is None:
            if "hifigan" in str(train_file):
                self.stats_path = "/home/monesh/ljspeech/LJSpeech-1.1/stats_hifigan.py"
            elif "vocos" in str(train_file):
                self.stats_path = "/home/monesh/ljspeech/LJSpeech-1.1/stats_vocos.py"
            else:
                self.stats_path = "/home/monesh/ljspeech/LJSpeech-1.1/stats.py"
        else:
            self.stats_path = stats_path

    def setup(self, stage=None):
        self.train_dataset = MyDataset(self.train_csv, stats_path=self.stats_path, prefetch_data=self.prefetch_data)
        self.val_dataset = MyDataset(self.val_csv, stats_path=self.stats_path, prefetch_data=self.prefetch_data)

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
    - Balanced multi-task loss: 1.0 * CFM Velocity MSE + 0.5 * Prior Latent Smooth L1 + 0.5 * Duration Smooth L1
    - Full End-to-End Training with Warmup + CosineAnnealing / CosineWarmRestarts scheduler
    - Pure Audio Highway Flow Matching Decoder + 4-Layer Bidirectional Mamba-2 Text Encoder
    """
    def __init__(
        self,
        lr=1e-4,
        max_epochs=200,
        model=None,
        **kwargs,
    ):
        super().__init__()
        if model is None:
            self.model = MambaFlowTTSModel(
                n_vocab=179,
                d_enc=192,
                d_codec=80,
                d_dec=384,
                gradient_checkpointing=kwargs.get("gradient_checkpointing", False),
            )
        else:
            self.model = model

        self.train_decoder_and_duration = kwargs.get("train_decoder_and_duration", False)
        self.train_decoder_only = kwargs.get("train_decoder_only", False)
        self.train_duration_only = kwargs.get("train_duration_only", False)
        self.duration_weight = kwargs.get("duration_weight", 0.1)
        self.duration_beta = kwargs.get("duration_beta", 0.1)
        self.prior_weight = kwargs.get("prior_weight", 0.5)
        self.lr = lr
        self.max_epochs = max_epochs
        self.save_hyperparameters(ignore=["model"])
        self.strict_loading = False

        if self.train_decoder_and_duration:
            self.freeze_encoder_modules()
        elif self.train_decoder_only:
            self.freeze_non_decoder_modules()
        elif self.train_duration_only:
            self.freeze_non_duration_modules()

    def on_load_checkpoint(self, checkpoint):
        """
        Filters out keys from state_dict that have shape mismatches (e.g. when duration
        predictor receptive fields or input dimensions change).
        """
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_state = self.state_dict()
        keys_to_remove = [k for k, v in state_dict.items() if k in model_state and v.shape != model_state[k].shape]
        for k in keys_to_remove:
            del state_dict[k]

    def freeze_encoder_modules(self):
        """
        Freezes text embeddings, text encoder, and latent projection.
        Leaves the duration predictor and flow-matching decoder trainable.
        """
        frozen_modules = [
            self.model.txt_emb,
            self.model.emb_dropout,
            self.model.encoder,
            self.model.to_latent,
        ]
        for module in frozen_modules:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

        if hasattr(self.model, "duration") and self.model.duration is not None:
            for param in self.model.duration.parameters():
                param.requires_grad = True

        if hasattr(self.model, "decoder") and self.model.decoder is not None:
            for param in self.model.decoder.parameters():
                param.requires_grad = True

    def freeze_non_decoder_modules(self):
        """
        Freezes text embeddings, text encoder, latent projection, and duration predictor.
        Leaves only the flow-matching decoder trainable.
        """
        frozen_modules = [
            self.model.txt_emb,
            self.model.emb_dropout,
            self.model.encoder,
            self.model.to_latent,
            self.model.duration,
        ]
        for module in frozen_modules:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

        if hasattr(self.model, "decoder") and self.model.decoder is not None:
            for param in self.model.decoder.parameters():
                param.requires_grad = True

    def freeze_non_duration_modules(self):
        """
        Freezes text embeddings, text encoder, latent projection, and flow-matching decoder.
        Leaves only the duration predictor trainable.
        """
        frozen_modules = [
            self.model.txt_emb,
            self.model.emb_dropout,
            self.model.encoder,
            self.model.to_latent,
        ]
        if hasattr(self.model, "decoder") and self.model.decoder is not None:
            frozen_modules.append(self.model.decoder)

        for module in frozen_modules:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

        if hasattr(self.model, "duration") and self.model.duration is not None:
            for param in self.model.duration.parameters():
                param.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if self.train_decoder_and_duration:
            self.freeze_encoder_modules()
            if hasattr(self.model, "duration") and self.model.duration is not None:
                self.model.duration.train(mode)
            if hasattr(self.model, "mu_smoother") and self.model.mu_smoother is not None:
                self.model.mu_smoother.train(mode)
            if hasattr(self.model, "decoder") and self.model.decoder is not None:
                self.model.decoder.train(mode)
        elif self.train_decoder_only:
            self.freeze_non_decoder_modules()
            if hasattr(self.model, "mu_smoother") and self.model.mu_smoother is not None:
                self.model.mu_smoother.train(mode)
            if hasattr(self.model, "decoder") and self.model.decoder is not None:
                self.model.decoder.train(mode)
        elif self.train_duration_only:
            self.freeze_non_duration_modules()
            if hasattr(self.model, "duration") and self.model.duration is not None:
                self.model.duration.train(mode)
        return self

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=1e-2,
            betas=(0.9, 0.98),
            eps=1e-8,
        )

        if self.train_duration_only:
            optimizer = optim.AdamW(
                trainable_params,
                lr=self.lr,
                weight_decay=1e-2,
                betas=(0.9, 0.98),
                eps=1e-8,
            )
            warmup_epochs = 3
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.05,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            decay_epochs = max(1, self.max_epochs - warmup_epochs) if self.max_epochs > 0 else 200
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=decay_epochs,
                eta_min=1e-6,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        elif self.train_decoder_only or self.train_decoder_and_duration:
            warmup_epochs = 3
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.2,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            decay_epochs = max(1, self.max_epochs - warmup_epochs) if self.max_epochs > 0 else 200
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=decay_epochs,
                eta_min=1e-6,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=50,
                T_mult=2,
                eta_min=1e-6,
            )
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=5,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[5],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def on_train_epoch_start(self):
        if self.train_decoder_and_duration:
            self.freeze_encoder_modules()
            if hasattr(self.model, "duration") and self.model.duration is not None:
                self.model.duration.train(True)
            if hasattr(self.model, "decoder") and self.model.decoder is not None:
                self.model.decoder.train(True)
        elif self.train_decoder_only:
            self.freeze_non_decoder_modules()
            if hasattr(self.model, "decoder") and self.model.decoder is not None:
                self.model.decoder.train(True)
        elif self.train_duration_only:
            self.freeze_non_duration_modules()
            if hasattr(self.model, "duration") and self.model.duration is not None:
                self.model.duration.train(True)

    def _shared_step(self, batch):
        text = batch['ipa']
        text_lengths = batch['ipa_lengths']
        audio_lengths = batch['lengths']
        target_mel = batch['mel']
        durations = batch.get('durations', None)
        device = text.device

        max_text_len = text.size(1)
        max_audio_len = target_mel.size(1)

        text_mask = torch.arange(max_text_len, device=device)[None, :] < text_lengths[:, None]
        audio_mask = torch.arange(max_audio_len, device=device)[None, :] < audio_lengths[:, None]

        (v_pred, u_target), mu_expanded, alignment, dur_pred, mas_score = self.model(
            x=text,
            target_latent=target_mel,
            durations=durations,
            mask=text_mask,
            audio_mask=audio_mask,
            skip_decoder=self.train_duration_only,
        )

        # Duration Loss: Direct log1p(duration) regression against MFA ground-truth using Huber loss (beta=0.1)
        dur_target = torch.log1p(durations.float().clamp(min=0.0))
        dur_loss = masked_smooth_l1_loss(dur_pred, dur_target, text_mask, beta=self.duration_beta)

        if self.train_duration_only:
            total_loss = dur_loss
            losses = {
                "cfm_loss": torch.tensor(0.0, device=device),
                "prior_loss": torch.tensor(0.0, device=device),
                "dur_loss": dur_loss,
            }
        elif self.train_decoder_and_duration:
            cfm_loss = masked_mse_loss(v_pred, u_target, audio_mask)
            prior_loss = masked_smooth_l1_loss(mu_expanded, target_mel, audio_mask, beta=0.5)
            total_loss = cfm_loss + self.duration_weight * dur_loss
            losses = {
                "cfm_loss": cfm_loss,
                "prior_loss": prior_loss.detach(),
                "dur_loss": dur_loss,
            }
        elif self.train_decoder_only:
            cfm_loss = masked_mse_loss(v_pred, u_target, audio_mask)
            prior_loss = masked_smooth_l1_loss(mu_expanded, target_mel, audio_mask, beta=0.5)
            total_loss = cfm_loss
            losses = {
                "cfm_loss": cfm_loss,
                "prior_loss": prior_loss.detach(),
                "dur_loss": dur_loss.detach(),
            }
        else:
            cfm_loss = masked_mse_loss(v_pred, u_target, audio_mask)
            prior_loss = masked_smooth_l1_loss(mu_expanded, target_mel, audio_mask, beta=0.5)
            total_loss = cfm_loss + self.prior_weight * prior_loss + self.duration_weight * dur_loss
            losses = {
                "cfm_loss": cfm_loss,
                "prior_loss": prior_loss,
                "dur_loss": dur_loss,
            }

        return total_loss, losses

    def training_step(self, batch, batch_idx):
        loss, losses = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        if self.train_duration_only:
            self.log("train_loss_dur", losses["dur_loss"], prog_bar=True, on_step=True, on_epoch=True)
        else:
            self.log("train_loss_cfm", losses["cfm_loss"], prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_loss_prior", losses["prior_loss"], prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_loss_dur", losses["dur_loss"], prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, losses = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        if self.train_duration_only:
            self.log("val_loss_dur", losses["dur_loss"], prog_bar=True, on_epoch=True)
        else:
            self.log("val_loss_cfm", losses["cfm_loss"], prog_bar=True, on_epoch=True)
            self.log("val_loss_prior", losses["prior_loss"], prog_bar=True, on_epoch=True)
            self.log("val_loss_dur", losses["dur_loss"], prog_bar=True, on_epoch=True)
        return loss


# Backward compatibility alias
TTSMODEL = MambaFlowTTS
