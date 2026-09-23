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
    Guarantees that the last checkpoint is atomically updated after every completed epoch,
    overwriting without accumulating duplicates, so training can always seamlessly resume.
    """
    def __init__(self, dirpath="TTS_checkpoints/", filename="mambaflow_last.ckpt"):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename

    def on_train_epoch_end(self, trainer, pl_module):
        os.makedirs(self.dirpath, exist_ok=True)
        last_path = os.path.join(self.dirpath, self.filename)
        tmp_path = os.path.join(self.dirpath, f"{self.filename}.tmp")
        trainer.save_checkpoint(tmp_path)
        os.replace(tmp_path, last_path)


def main():
    # =========================================================================
    # User-Configured Parameters (Manual Configuration)
    # =========================================================================
    # Training Loop
    max_epochs = -1                          # -1 for unlimited epochs (runs until early stopping or manual stop)
    batch_size = 32                         # Per-GPU batch size
    accumulate_grad_batches = 1              # 2 steps -> Effective batch size = 32 (fits RTX 3050 4GB)
    lr = 2e-4                                # Learning rate (AdamW with linear warmup + cosine restarts)
    early_stop_patience = 110                # Stop if validation loss does not improve for 110 epochs
    num_workers = 8                          # CPU dataloader workers
    prefetch_data = True                     # Fast prefetching

    # Checkpoint Configuration & Naming
    ckpt_dir = "TTS_checkpoints/"
    last_ckpt_name = "mambaflow_hifigan_last.ckpt"   # Atomic checkpoint updated every epoch
    save_top_k = 5                                   # Number of best validation models to save

    # Resuming & Warm-Start Control (Training from scratch)
    resume_if_exists = True                          # Resume if mambaflow_hifigan_last.ckpt exists
    custom_resume_ckpt = None                        # Or set explicit path, else None
    warm_start_encoder_duration = False              # Training all components from scratch
    warm_start_ckpt = None

    # Dataset Paths
    train_file = "/home/monesh/ljspeech/LJSpeech-1.1/train_hifigan.csv"
    val_file = "/home/monesh/ljspeech/LJSpeech-1.1/val_hifigan.csv"

    # TensorBoard Logging Directory
    log_dir = "TTS_logs"
    exp_name = "mambaflow_hifigan_80mel_scratch"
    # =========================================================================

    os.makedirs(ckpt_dir, exist_ok=True)

    # Determine checkpoint for resuming
    last_ckpt_path = os.path.join(ckpt_dir, last_ckpt_name)
    resume_ckpt = None
    if custom_resume_ckpt and os.path.exists(custom_resume_ckpt):
        resume_ckpt = custom_resume_ckpt
        print(f"[+] Resuming training from custom checkpoint: {resume_ckpt}")
    elif resume_if_exists and os.path.exists(last_ckpt_path):
        resume_ckpt = last_ckpt_path
        print(f"[+] Found existing last checkpoint '{resume_ckpt}'. Resuming automatically.")
    else:
        print("[+] Starting training from scratch (no checkpoint loaded).")

    datamodule = TTSDataModule(
        train_file=train_file,
        val_file=val_file,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_data=prefetch_data,
    )

    # Top-K best checkpoints based on validation loss
    top_k_checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="mambaflow_hifigan_val_epoch_{epoch:03d}_loss_{val_loss:.4f}",
        auto_insert_metric_name=False,
        monitor="val_loss",
        mode="min",
        save_top_k=save_top_k,
        save_last=False,
    )

    # Dedicated atomic callback ensuring mambaflow_hifigan_last.ckpt is always the newest epoch
    last_checkpoint_callback = AtomicLastCheckpointCallback(
        dirpath=ckpt_dir,
        filename=last_ckpt_name,
    )

    # Early stopping waiting early_stop_patience epochs without validation improvement
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=early_stop_patience,
        verbose=True,
        mode="min",
    )

    # Learning rate monitor for TensorBoard
    lr_monitor = LearningRateMonitor(logging_interval="step")

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
        logger=tb_logger,
        callbacks=[
            top_k_checkpoint_callback,
            last_checkpoint_callback,
            early_stop_callback,
            lr_monitor,
            RichProgressBar(),
            ModelSummary(max_depth=-1),
        ],
    )

    # Initialize model: Training from scratch (all modules trainable)
    lightning_model = TTSMODEL(
        lr=lr,
        max_epochs=max_epochs,
        duration_weight=0.1,
        prior_weight=0.1,
        train_duration_only=False,
        train_decoder_only=False,
        train_decoder_and_duration=False,
    )

    if resume_ckpt and os.path.exists(resume_ckpt):
        ckpt_data = torch.load(resume_ckpt, map_location="cpu")
        state_dict = ckpt_data.get("state_dict", ckpt_data)
        lightning_model.load_state_dict(state_dict, strict=False)
        print("=" * 80)
        print(f"[+] SUCCESSFULLY LOADED MODEL WEIGHTS FROM: {resume_ckpt}")
        print(f"    (Text Encoder, Duration Predictor, and CFM Decoder loaded)")
        print(f"    Starting fresh optimizer and LR cycle on corrected 22.05 kHz durations.")
        print("=" * 80)
        fit_ckpt_path = None
    elif warm_start_encoder_duration and warm_start_ckpt and os.path.exists(warm_start_ckpt):
        ckpt_data = torch.load(warm_start_ckpt, map_location="cpu")
        state = ckpt_data.get("state_dict", ckpt_data)
        model_state = lightning_model.state_dict()
        matched = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
        lightning_model.load_state_dict(matched, strict=False)
        print("=" * 80)
        print(f"[+] WARM-START: Successfully loaded {len(matched)} matching parameters")
        print("=" * 80)

    total_params = sum(p.numel() for p in lightning_model.parameters())
    trainable_params = sum(p.numel() for p in lightning_model.parameters() if p.requires_grad)

    print("=" * 80)
    print(f"[+] STARTING MAMBAFLOW-TTS HiFi-GAN TRAINING ON LJ-SPEECH")
    print(f"[+] Max Epochs:           {max_epochs} (Unlimited / Continuous training)")
    print(f"[+] Learning Rate:        {lr} (Warmup 3 epochs + CosineAnnealingLR)")
    print(f"[+] Early Stopping:       Patience = {early_stop_patience} epochs")
    print(f"[+] Batch Size:           {batch_size} (Accumulate grads = {accumulate_grad_batches} -> Effective: {batch_size * accumulate_grad_batches})")
    print(f"[+] Train Dataset:        {train_file}")
    print(f"[+] Validation Dataset:   {val_file}")
    print(f"[+] Text Encoder:         6-Stack Pure Transformer (TRAINABLE from scratch)")
    print(f"[+] Duration Predictor:   3-Layer ResNet-Conv1D (TRAINABLE from scratch)")
    print(f"[+] Decoder:              Pure Mamba-2 + ConvNeXt (256-dim, 4 layers, 80 mel channels, TRAINABLE from scratch)")
    print(f"[+] Conditioning:         Input Highway Concat([x_t, mu]) + Timestep FiLM")
    print(f"[+] Loss Configuration:   1.0 * CFM Velocity MSE + 0.5 * Prior L1 + 0.1 * Duration Huber Loss")
    print(f"[+] Total Parameters:     {total_params:,}")
    print(f"[+] Trainable Parameters: {trainable_params:,}")
    print(f"[+] Checkpoint Directory: {ckpt_dir}")
    print(f"[+] Best Checkpoints:     {ckpt_dir}mambaflow_hifigan_val_epoch_{{epoch:03d}}_loss_{{val_loss:.4f}}.ckpt")
    print(f"[+] Latest Checkpoint:    {ckpt_dir}{last_ckpt_name}")
    print(f"[+] Resume Checkpoint:    {resume_ckpt if resume_ckpt else 'None (Clean start from scratch)'}")
    print(f"[+] TensorBoard Logs:     {log_dir}/{exp_name}/")
    print("=" * 80)

    trainer.fit(model=lightning_model, datamodule=datamodule, ckpt_path="TTS_checkpoints/mambaflow_hifigan_last.ckpt")


if __name__ == "__main__":
    main()
