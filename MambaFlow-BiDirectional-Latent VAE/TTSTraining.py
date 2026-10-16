import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..")))

import warnings
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    ModelSummary,
    Callback,
    LearningRateMonitor,
)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from TTSDataModule import TTSDataModule, TTSMODEL

warnings.filterwarnings("ignore", category=UserWarning, module="requests")
torch.backends.cudnn.benchmark = False


class AtomicLastCheckpointCallback(Callback):
    """
    Guarantees that the last checkpoint is atomically updated periodically,
    overwriting without accumulating duplicates, so training can always seamlessly resume.
    """
    def __init__(self, dirpath="TTS_checkpoints/", filename="mambaflow_full_cfg15_last.ckpt", save_every=1):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.save_every = save_every

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        max_ep = trainer.max_epochs
        if epoch % self.save_every == 0 or epoch == max_ep:
            os.makedirs(self.dirpath, exist_ok=True)
            last_path = os.path.join(self.dirpath, self.filename)
            tmp_path = os.path.join(self.dirpath, f"{self.filename}.tmp")
            trainer.save_checkpoint(tmp_path)
            os.replace(tmp_path, last_path)



def main():
    # =========================================================================
    # User-Configured Parameters: FULL DATASET TRAINING FROM SCRATCH (128D DACAE, 448D DECODER, 15% CFG)
    # =========================================================================
    # Training Loop Configuration
    max_epochs = -1                          # -1 = Infinite epochs (stops upon EarlyStopping trigger)
    batch_size = 20                          # Per-GPU batch size (Peak VRAM ~2.5 - 2.9 GB on 4GB GPU)
    accumulate_grad_batches = 1              # Effective batch size = 20
    lr_convnext = 2e-4                       # Pointwise Dense MLP LR
    lr_decoder = 2e-4                        # 448D Decoder, Text encoder, duration predictor, BiMamba-2 LR
    lr_patience = 10                         # Higher patience (10 epochs) avoiding false drops from CFM validation noise
    lr_factor = 0.7                          # Gentle 30% reduction (0.7) rather than aggressive halving
    min_lr_convnext = 1e-5                   # Higher LR floor (1e-5) ensuring active parameter updates
    min_lr_decoder = 1e-5                    # Higher LR floor (1e-5) preventing frozen optimizer
    early_stop_patience = 60                 # High early stopping patience allowing learning down to min_lr
    num_workers = 4                          # DataLoader workers
    prefetch_data = True                     # In-memory prefetching
    d_codec = 128                            # 128-dim DACAE continuous latent representation
    d_enc = 192                              # 192-dim 8-Stack Transformer Text Encoder (4.72M params)
    d_dec = 448                              # 448-dim CFM Decoder (Cross-Attention + BiMamba-2 + Dense MLP)
    num_dec_blocks = 6                       # 6 Decoder Layers
    cfg_dropout = 0.15                       # 15% Classifier-Free Guidance (CFG dropout)
    logit_normal_mean = 0.0                  # Logit-normal timestep sampling mean
    logit_normal_std = 1.0                   # Logit-normal timestep sampling standard deviation

    # Training Stage: FULL MODEL TRAINING FROM SCRATCH
    train_decoder_only = False               # Full training: Text Encoder, Duration Predictor, CFM Decoder
    train_duration_only = False              # All modules trainable
    train_decoder_and_duration = False       # All modules trainable

    # Checkpoint Configuration & Naming
    ckpt_dir = "TTS_checkpoints/"
    last_ckpt_name = "mambaflow_ca_bimamba_448d_cfg15_last.ckpt"    # Atomic checkpoint updated every epoch
    save_top_k = 3                                                  # Save top 3 best checkpoints

    # Dataset Paths: Full LJSpeech Dataset
    train_file = "/home/monesh/ljspeech/LJSpeech-1.1/train.csv"
    val_file = "/home/monesh/ljspeech/LJSpeech-1.1/val.csv"

    # TensorBoard Logging Directory
    log_dir = "TTS_logs"
    exp_name = "mambaflow_ca_bimamba_448d_cfg15_scratch"
    # =========================================================================

    os.makedirs(ckpt_dir, exist_ok=True)

    datamodule = TTSDataModule(
        train_file=train_file,
        val_file=val_file,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_data=prefetch_data,
        target_dim=d_codec,
    )

    # Top-K best checkpoints based on val_loss
    top_k_checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="mambaflow_ca_bimamba_448d_cfg15_epoch_{epoch:03d}_loss_{val_loss:.4f}",
        auto_insert_metric_name=False,
        monitor="val_loss",
        mode="min",
        save_top_k=save_top_k,
        save_last=False,
    )

    # Dedicated atomic callback ensuring last_ckpt_name is periodically updated every epoch
    last_checkpoint_callback = AtomicLastCheckpointCallback(
        dirpath=ckpt_dir,
        filename=last_ckpt_name,
        save_every=1,
    )

    # Learning rate monitor for TensorBoard
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Early stopping configured so ReduceLROnPlateau dominates until min_lr is reached
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=early_stop_patience,
        mode="min",
        verbose=True,
    )

    # TensorBoard Logger
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=exp_name,
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        precision="bf16-mixed",
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
        limit_val_batches=100,                 # Smooth validation evaluation (1600 samples)
        num_sanity_val_steps=1,
        check_val_every_n_epoch=1,
        logger=tb_logger,
        callbacks=[
            top_k_checkpoint_callback,
            last_checkpoint_callback,
            lr_monitor,
            early_stop_callback,
            RichProgressBar(),
            ModelSummary(max_depth=-1),
        ],
    )

    # Initialize model with training mode configuration from scratch
    lightning_model = TTSMODEL(
        lr=lr_convnext,
        lr_convnext=lr_convnext,
        lr_decoder=lr_decoder,
        patience=lr_patience,
        factor=lr_factor,
        min_lr_convnext=min_lr_convnext,
        min_lr_decoder=min_lr_decoder,
        max_epochs=max_epochs,
        d_codec=d_codec,
        d_enc=d_enc,
        d_dec=d_dec,
        num_dec_blocks=num_dec_blocks,
        cfg_dropout=cfg_dropout,
        duration_weight=0.1,
        prior_weight=0.1,
        train_duration_only=train_duration_only,
        train_decoder_only=train_decoder_only,
        train_decoder_and_duration=train_decoder_and_duration,
        logit_normal_mean=logit_normal_mean,
        logit_normal_std=logit_normal_std,
        monitor="val_loss",
    )

    # Training from scratch: Pure random weight initialization (No checkpoint reload)
    total_params = sum(p.numel() for p in lightning_model.parameters())
    trainable_params = sum(p.numel() for p in lightning_model.parameters() if p.requires_grad)

    print("=" * 80)
    print(f"[+] MAMBAFLOW-TTS 128D DACAE LATENT: TRAINING FROM SCRATCH")
    print(f"[+] Target Latent Dim:    {d_codec}D (DACAE Latent CFM Velocity Matching)")
    print(f"[+] Mode:                 TRAIN FROM SCRATCH (No checkpoint reload)")
    print(f"[+] Max Epochs:           {max_epochs} (Full LJSpeech training)")
    print(f"[+] Timestep Sampling:    Logit-Normal (mean={logit_normal_mean}, std={logit_normal_std})")
    print(f"[+] Cross-Attention:      Pre-Norm RMSNorm (No timestep shifts)")
    print(f"[+] AdaLN Modulation:     6 parameters per layer (BiMamba-2 + Dense MLP)")
    print(f"[+] Dense MLP LR:         {lr_convnext} (min_lr={min_lr_convnext})")
    print(f"[+] Encoder/Decoder LR:   {lr_decoder} (min_lr={min_lr_decoder})")
    print(f"[+] LR Scheduler:         ReduceLROnPlateau (patience={lr_patience}, factor={lr_factor}, min_lr={min_lr_decoder})")
    print(f"[+] Early Stopping:       patience={early_stop_patience} (subordinate to ReduceLROnPlateau)")
    print(f"[+] Batch Size:           {batch_size} (Accumulate grads = {accumulate_grad_batches})")
    print(f"[+] CFG Dropout:          {cfg_dropout * 100:.1f}% (CFG enabled - joint conditional & unconditional velocity matching)")
    print(f"[+] Train Dataset:        {train_file} (11,790 samples)")
    print(f"[+] Validation Dataset:   {val_file} (1,310 samples, 100 batches per epoch)")
    print(f"[+] Text Encoder:         TRAINABLE ({d_enc}D, 8 layers, 6 heads, Pure Transformer)")
    print(f"[+] Duration Predictor:   TRAINABLE (3-layer high-capacity, RF=13)")
    print(f"[+] CFM Decoder:          TRAINABLE ({d_dec}D, Cross-Attention -> BiMamba-2 -> Dense MLP, {num_dec_blocks} layers)")
    print(f"[+] Total Parameters:     {total_params:,}")
    print(f"[+] Trainable Parameters: {trainable_params:,}")
    print(f"[+] Checkpoint Directory: {ckpt_dir}")
    print(f"[+] Checkpoints Pattern:  {ckpt_dir}mambaflow_ca_bimamba_448d_cfg15_epoch_{{epoch:03d}}_loss_{{val_loss:.4f}}.ckpt")
    print(f"[+] Latest Checkpoint:    {ckpt_dir}{last_ckpt_name}")
    print(f"[+] TensorBoard Logs:     {log_dir}/{exp_name}/")
    print("=" * 80)

    trainer.fit(model=lightning_model, datamodule=datamodule, ckpt_path = "TTS_checkpoints/mambaflow_ca_bimamba_448d_cfg15_last.ckpt")


if __name__ == "__main__":
    main()
