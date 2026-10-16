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
from mamba_decoder import PointwiseDenseMLPBlock, DenseMLPBlock

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
        target_dim=128,
    ):
        super().__init__()
        self.train_csv = train_file
        self.val_csv = val_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_data = prefetch_data
        self.target_dim = target_dim

    def setup(self, stage=None):
        self.train_dataset = MyDataset(
            self.train_csv,
            target_dim=self.target_dim,
            prefetch_data=self.prefetch_data,
        )
        if self.val_csv is not None and os.path.exists(self.val_csv):
            self.val_dataset = MyDataset(
                self.val_csv,
                target_dim=self.target_dim,
                prefetch_data=self.prefetch_data,
            )
        else:
            self.val_dataset = None

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
        if self.val_dataset is None:
            return None
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
    - Pure Audio Highway Flow Matching Decoder (UniMamba-2 + ConvNeXt V2 with GRN)
    """
    def __init__(
        self,
        lr=2e-4,
        lr_convnext=None,
        lr_decoder=2e-4,
        patience=4,
        factor=0.5,
        min_lr_convnext=1e-6,
        min_lr_decoder=1e-6,
        max_epochs=200,
        model=None,
        d_codec=128,
        **kwargs,
    ):
        super().__init__()
        self.d_codec = d_codec
        d_enc = kwargs.get("d_enc", 192)
        d_dec = kwargs.get("d_dec", 448)
        num_dec_blocks = kwargs.get("num_dec_blocks", 6)
        if model is None:
            self.model = MambaFlowTTSModel(
                n_vocab=179,
                d_enc=d_enc,
                d_codec=d_codec,
                d_dec=d_dec,
                num_dec_blocks=num_dec_blocks,
                cfg_dropout=kwargs.get("cfg_dropout", 0.15),
                gradient_checkpointing=kwargs.get("gradient_checkpointing", False),
                logit_normal_mean=kwargs.get("logit_normal_mean", 0.0),
                logit_normal_std=kwargs.get("logit_normal_std", 1.0),
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
        self.lr_convnext = lr_convnext if lr_convnext is not None else lr
        self.lr_decoder = lr_decoder
        self.patience = patience
        self.factor = factor
        self.min_lr_convnext = min_lr_convnext
        self.min_lr_decoder = min_lr_decoder
        self.max_epochs = max_epochs
        self.save_hyperparameters(ignore=["model"])
        self.strict_loading = False
        self.monitor = kwargs.get("monitor", "val_loss")

        if self.train_decoder_and_duration:
            self.freeze_encoder_modules()
        elif self.train_decoder_only:
            self.freeze_non_decoder_modules()
        elif self.train_duration_only:
            self.freeze_non_duration_modules()

    def on_load_checkpoint(self, checkpoint):
        """
        Adapts ConvNeXt V2 dwconv weights from kernel 7 to 3 using center slice:
        new_weight = old_weight[..., 2:5].clone()
        Filters out keys from state_dict that have shape mismatches.
        """
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_state = self.state_dict()
        for k in list(state_dict.keys()):
            if "dwconv.weight" in k and state_dict[k].shape[-1] == 7 and (k in model_state and model_state[k].shape[-1] == 3):
                print(f"[+] on_load_checkpoint: Adapting {k} shape {state_dict[k].shape} -> {model_state[k].shape} via [..., 2:5].clone()")
                state_dict[k] = state_dict[k][..., 2:5].clone()

        keys_to_remove = [k for k, v in state_dict.items() if k in model_state and v.shape != model_state[k].shape]
        for k in keys_to_remove:
            print(f"[-] Dropping mismatched key: {k} (ckpt: {state_dict[k].shape} vs model: {model_state[k].shape})")
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
        ]
        if hasattr(self.model, "to_latent") and self.model.to_latent is not None:
            frozen_modules.append(self.model.to_latent)

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
            self.model.duration,
        ]
        if hasattr(self.model, "to_latent") and self.model.to_latent is not None:
            frozen_modules.append(self.model.to_latent)

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
        ]
        if hasattr(self.model, "to_latent") and self.model.to_latent is not None:
            frozen_modules.append(self.model.to_latent)
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
            if hasattr(self.model, "decoder") and self.model.decoder is not None:
                self.model.decoder.train(mode)
        elif self.train_decoder_only:
            self.freeze_non_decoder_modules()
            if hasattr(self.model, "decoder") and self.model.decoder is not None:
                self.model.decoder.train(mode)
        elif self.train_duration_only:
            self.freeze_non_duration_modules()
            if hasattr(self.model, "duration") and self.model.duration is not None:
                self.model.duration.train(mode)
        return self

    def configure_optimizers(self):
        dense_mlp_params = []
        remaining_params = []
        mlp_param_ids = set()

        # Identify all parameters belonging to PointwiseDenseMLPBlock
        for module in self.model.modules():
            if isinstance(module, (PointwiseDenseMLPBlock, DenseMLPBlock)):
                for p in module.parameters():
                    if p.requires_grad and id(p) not in mlp_param_ids:
                        mlp_param_ids.add(id(p))
                        dense_mlp_params.append(p)

        for p in self.parameters():
            if p.requires_grad and id(p) not in mlp_param_ids:
                remaining_params.append(p)

        param_groups = []
        min_lrs = []
        if dense_mlp_params:
            param_groups.append({
                "params": dense_mlp_params,
                "lr": self.lr_convnext,
                "name": "dense_mlp",
            })
            min_lrs.append(self.min_lr_convnext)
            print(f"[Optimizer] Group 'dense_mlp': {len(dense_mlp_params)} tensors ({sum(p.numel() for p in dense_mlp_params):,} params) at LR={self.lr_convnext}, min_lr={self.min_lr_convnext}")

        if remaining_params:
            param_groups.append({
                "params": remaining_params,
                "lr": self.lr_decoder,
                "name": "remaining_decoder",
            })
            min_lrs.append(self.min_lr_decoder)
            print(f"[Optimizer] Group 'remaining_decoder': {len(remaining_params)} tensors ({sum(p.numel() for p in remaining_params):,} params) at LR={self.lr_decoder}, min_lr={self.min_lr_decoder}")

        optimizer = optim.AdamW(
            param_groups,
            weight_decay=1e-2,
            betas=(0.9, 0.98),
            eps=1e-8,
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.factor,       # Reduce to half (0.5)
            patience=self.patience,   # Higher patience (e.g. 20 epochs)
            min_lr=min_lrs if len(min_lrs) > 1 else min_lrs[0],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": getattr(self, "monitor", "val_loss"),
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def lr_scheduler_step(self, scheduler, metric):
        if metric is not None:
            scheduler.step(metric)
        else:
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

        (v_pred, u_target), cond_expanded, alignment, dur_pred, mas_score = self.model(
            x=text,
            target_latent=target_mel,
            durations=durations,
            mask=text_mask,
            audio_mask=audio_mask,
            skip_decoder=self.train_duration_only,
        )

        # Align sequence lengths across all tensors
        if v_pred is not None:
            min_T = min(v_pred.size(1), target_mel.size(1))
            v_pred = v_pred[:, :min_T, :]
            u_target = u_target[:, :min_T, :]
            step_audio_mask = audio_mask[:, :min_T] if audio_mask is not None else None
        else:
            step_audio_mask = audio_mask

        # Duration Loss: Direct log1p(duration) regression against MFA ground-truth using Huber loss (beta=0.1)
        dur_target = torch.log1p(durations.float().clamp(min=0.0))
        dur_loss = masked_smooth_l1_loss(dur_pred, dur_target, text_mask, beta=self.duration_beta)

        if self.train_duration_only:
            total_loss = dur_loss
            losses = {
                "cfm_loss": torch.tensor(0.0, device=device),
                "dur_loss": dur_loss,
            }
        elif self.train_decoder_only:
            cfm_loss = masked_mse_loss(v_pred, u_target, step_audio_mask)
            total_loss = cfm_loss
            losses = {
                "cfm_loss": cfm_loss,
                "dur_loss": dur_loss.detach(),
            }
        else:
            cfm_loss = masked_mse_loss(v_pred, u_target, step_audio_mask)
            total_loss = cfm_loss + self.duration_weight * dur_loss
            losses = {
                "cfm_loss": cfm_loss,
                "dur_loss": dur_loss,
            }

        return total_loss, losses

    def training_step(self, batch, batch_idx):
        loss, losses = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        if self.train_duration_only:
            self.log("train_loss_dur", losses["dur_loss"], prog_bar=True, on_step=True, on_epoch=True)
        elif self.train_decoder_only:
            self.log("train_loss_cfm", losses["cfm_loss"], prog_bar=True, on_step=True, on_epoch=True)
        else:
            self.log("train_loss_cfm", losses["cfm_loss"], prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_loss_dur", losses["dur_loss"], prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, losses = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        if self.train_duration_only:
            self.log("val_loss_dur", losses["dur_loss"], prog_bar=True, on_epoch=True)
        elif self.train_decoder_only:
            self.log("val_loss_cfm", losses["cfm_loss"], prog_bar=True, on_epoch=True)
        else:
            self.log("val_loss_cfm", losses["cfm_loss"], prog_bar=True, on_epoch=True)
            self.log("val_loss_dur", losses["dur_loss"], prog_bar=True, on_epoch=True)
        return loss


# Backward compatibility alias
TTSMODEL = MambaFlowTTS
