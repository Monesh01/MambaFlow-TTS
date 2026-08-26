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


class AtomicLastCheckpointCallback(Callback):
    """
    Guarantees that 'last.ckpt' is atomically updated after every completed epoch,
    overwriting any previous 'last.ckpt' without accumulating duplicates,
    so training can always seamlessly resume from the most recent completed epoch.
    """
    def __init__(self, dirpath="TTS_checkpoints/"):
        super().__init__()
        self.dirpath = dirpath

    def on_train_epoch_end(self, trainer, pl_module):
        os.makedirs(self.dirpath, exist_ok=True)
        last_path = os.path.join(self.dirpath, "last.ckpt")
        tmp_path = os.path.join(self.dirpath, "last.ckpt.tmp")
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
        batch_size=8,
        num_workers=8,
        prefetch_data=True,
    )

    ckpt_dir = "TTS_checkpoints/"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Top-5 best checkpoints based on validation loss
    top_k_checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="TTSmodel-cfm-v13-continue-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=False,
    )

    # Dedicated atomic callback ensuring last.ckpt is always the most recent completed epoch
    last_checkpoint_callback = AtomicLastCheckpointCallback(dirpath=ckpt_dir)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=40,
        verbose=True,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=200,
        accelerator="auto",
        precision="bf16-mixed",
        accumulate_grad_batches=4,
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

    # Load weights from the best v3 checkpoint
    best_ckpt = "/home/monesh/TTSModel/TTS_checkpoints/last.ckpt" #TTSmodel-cfm-v13-epoch=82-val_loss=0.3412.ckpt"  #"/home/monesh/TTSModel/TTS_checkpoints/the last best/TTSmodel-v3-epoch=104-val_loss=0.6117.ckpt"
    
    # Initialize model with new LR
    lightning_model = TTSMODEL(
        lambda_latent=1.0,
        lambda_cfm=1.0,
        lambda_dur_pred=1.0,
        d_codec=100,
        gradient_checkpointing=False,
        train_decoder_only=True,
        freeze_latent_only=False,
        lr=2e-4,
        max_epochs=200,
    )

    print("Loading pretrained weights (excluding decoder)...")
    checkpoint = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    
    # Filter out decoder weights entirely so it initializes from scratch
    # This automatically prevents shape mismatch errors for the new kernel sizes
    #filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("model.decoder.")}
    #lightning_model.load_state_dict(filtered_state_dict, strict=False)

    print("Training Decoder from scratch with [1,1,1,1,1,1] dilations and [7..9] kernels. LR=2e-4.")
    # Not passing ckpt_path to trainer.fit() to start optimizer from scratch (epoch 0)
    trainer.fit(model=lightning_model, datamodule=datamodule, ckpt_path = best_ckpt)

if __name__ == "__main__":
    main()
