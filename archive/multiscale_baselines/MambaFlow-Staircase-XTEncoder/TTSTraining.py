import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# pyrefly: ignore [missing-import]
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, ModelSummary, Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from TTSDataModule import TTSDataModule, TTSMODEL
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="requests")

torch.backends.cudnn.benchmark = False

class AtomicLastCheckpointCallback(Callback):
    """
    Guarantees that the last checkpoint is atomically updated after every completed epoch,
    overwriting without accumulating duplicates, so training can always seamlessly resume.
    """
    def __init__(self, dirpath="TTS_checkpoints/", filename="last_decoder_only.ckpt"):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename

    def on_train_epoch_end(self, trainer, pl_module):
        os.makedirs(self.dirpath, exist_ok=True)
        last_path = os.path.join(self.dirpath, self.filename)
        tmp_path = os.path.join(self.dirpath, f"{self.filename}.tmp")
        trainer.save_checkpoint(tmp_path)
        os.replace(tmp_path, last_path)

def get_latest_checkpoint(ckpt_dir="TTS_checkpoints/"):
    if not os.path.exists(ckpt_dir):
        return None
    ckpts = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith(".ckpt") and not f.endswith(".tmp")
    ]
    if not ckpts:
        return None
    
    # Prioritize last*.ckpt if present, selecting the most recently modified
    last_ckpts = [f for f in ckpts if os.path.basename(f).startswith("last")]
    if last_ckpts:
        return max(last_ckpts, key=os.path.getmtime)
    
    # Otherwise return the newest .ckpt by modification time
    return max(ckpts, key=os.path.getmtime)

def main():
    datamodule = TTSDataModule(
        train_file="/home/monesh/ljspeech/LJSpeech-1.1/train.csv",
        val_file="/home/monesh/ljspeech/LJSpeech-1.1/val.csv",
        batch_size=32,
        num_workers=8,
        prefetch_data=True,
    )

    ckpt_dir = "TTS_checkpoints/"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Top-5 best checkpoints based on validation CFM loss
    top_k_checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="Tetra-Decoder-epoch={epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=False,
    )

    # Dedicated atomic callback ensuring last_tetra_decoder.ckpt is always the most recent completed epoch
    last_checkpoint_callback = AtomicLastCheckpointCallback(
        dirpath=ckpt_dir,
        filename="last_tetra_decoder.ckpt",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=50,
        verbose=True,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=200,
        accelerator="auto",
        precision="bf16-mixed",
        accumulate_grad_batches=1,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
        callbacks=[
            top_k_checkpoint_callback,
            last_checkpoint_callback,
            early_stop_callback,
            RichProgressBar(),
            ModelSummary(max_depth=-1),
        ],
    )

    # Initialize model configured with new Tetra Mamba-2 Sequential Decoder
    lightning_model = TTSMODEL(
        lambda_latent=0.0,
        lambda_cfm=1.0,
        lambda_dur_pred=0.0,
        d_enc=192,
        d_dec=384,  # Scaled internal capacity (384-dim)
        d_codec=100,
        gradient_checkpointing=False,
        train_decoder_only=True,  # Freezes text encoder, duration predictor, and latent projection
        freeze_latent_only=False,
        decoder_type="tetra",
        lr=5e-4,  # Fresh decoder training lr decaying to 1e-6 (0.000001) over 200 epochs
        max_epochs=200,
    )

    # Load pre-trained weights from specified checkpoint (epoch reset 0 to 200)
    resume_ckpt_path = os.path.join(ckpt_dir, "XT-Staircase-E2E-epoch=145-val_loss=0.5805.ckpt")
    if not os.path.exists(resume_ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {resume_ckpt_path}")

    print(f"[+] Loading pre-trained text encoder and duration weights from: {resume_ckpt_path}")
    ckpt = torch.load(resume_ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Filter out old decoder weights to initialize the new Tetra decoder cleanly from scratch
    pretrained_encoder_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith("model.decoder.")
    }
    missing, unexpected = lightning_model.load_state_dict(pretrained_encoder_dict, strict=False)
    print(f"[+] Successfully loaded {len(pretrained_encoder_dict)} pre-trained weights into encoder & duration predictor")

    # Enforce train/eval modes
    lightning_model.train(True)

    print("=" * 80)
    print("[+] STARTING TETRA MAMBA-2 SEQUENTIAL DECODER TRAINING (Epochs: 0 to 200)")
    print(f"[+] Pretrained Base: {resume_ckpt_path} (Frozen: Text Encoder, Latent Proj, Duration Predictor)")
    print("[+] Architecture: Tetra Sequential Full-Resolution Decoder (3x BiMamba-2 + 1x ConvNeXt k=17)")
    print("[+] Conditioning: Timestep t on all 4 layers; mu only on Layer 1 and Layer 3")
    print("[+] Residuals: Layer 1 & 4 WITH residual; Layer 2 & 3 NO residual (anti-dual-audio)")
    print("[+] Hidden Capacity: 384-dim, State: 64, Headdim: 64")
    print("[+] Multi-Task Loss: 1.0 * CFM Velocity MSE (Latent & Duration frozen)")
    print("[+] Learning Rate: 1e-4 -> 1e-6 over 200 epochs via CosineAnnealingLR")
    print(f"[+] Checkpoint Dir: {ckpt_dir} | Pattern: Tetra-Decoder-*.ckpt")
    print("=" * 80)

    # Fit without ckpt_path so trainer runs fresh from epoch 0 to 200
    trainer.fit(model=lightning_model, datamodule=datamodule)

if __name__ == "__main__":
    main()
