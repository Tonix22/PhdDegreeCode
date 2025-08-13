# train_pl_zf_mnv3.py
# Zero-Forcing phase corrector w/ MobileNetV3 (Lightning)
# Input:  H (48x48 complex) -> [mag,phase] => (2,48,48)
# Output: theta_hat (48,) per subcarrier; loss = wrapped MSE on phase(Y * e^{jθ̂}) vs angle(X)

import os
import glob
import argparse
import numpy as np
from typing import List, Tuple
import h5py
import torch
torch.set_float32_matmul_precision("high")
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small
from scipy.io import loadmat

import pytorch_lightning as pl


# ---------------- utils ----------------

def complex_to_magphase(C: np.ndarray) -> np.ndarray:
    mag = np.abs(C)
    phs = np.angle(C)
    return np.stack([mag, phs], axis=0).astype(np.float32)

def to_torch_complex_ri(v: np.ndarray) -> torch.Tensor:
    # (N,) complex -> (N,2) real/imag
    return torch.from_numpy(
        np.stack([v.real.astype(np.float32), v.imag.astype(np.float32)], axis=-1)
    )

def complex_mul_by_phase(y_ri: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    # y_ri: (B,48,2), theta: (B,48)
    c, s = torch.cos(theta), torch.sin(theta)
    yr, yi = y_ri[..., 0], y_ri[..., 1]
    out_r = yr * c - yi * s
    out_i = yr * s + yi * c
    return torch.stack([out_r, out_i], dim=-1)

def angle_from_ri(ri: torch.Tensor) -> torch.Tensor:
    # (B,48,2) -> (B,48)
    return torch.atan2(ri[..., 1], ri[..., 0])

def wrapped_phase_mse(pred_phase: torch.Tensor, tgt_phase: torch.Tensor) -> torch.Tensor:
    d = pred_phase - tgt_phase
    d_wrapped = torch.atan2(torch.sin(d), torch.cos(d))
    return torch.mean(d_wrapped ** 2)



def _h5_to_complex(arr_or_group):
    """
    Converts MATLAB v7.3 storage to a complex numpy array.
    Handles either:
      - group with datasets 'real' and 'imag'
      - dataset with fields ('real','imag') or ('r','i')
      - plain real dataset (returns float array)
    """
    if isinstance(arr_or_group, h5py.Group):
        # Common MATLAB layout: <var>/real and <var>/imag
        real = np.array(arr_or_group["real"])
        imag = np.array(arr_or_group["imag"])
        return real + 1j * imag
    elif isinstance(arr_or_group, h5py.Dataset):
        dt = arr_or_group.dtype
        if dt.fields:  # compound dtype
            fields = dt.fields.keys()
            if "real" in fields and "imag" in fields:
                data = np.array(arr_or_group)
                return data["real"] + 1j * data["imag"]
            if "r" in fields and "i" in fields:
                data = np.array(arr_or_group)
                return data["r"] + 1j * data["i"]
        # plain real dataset
        return np.array(arr_or_group)
    else:
        raise ValueError("Unsupported HDF5 node type for complex conversion.")

def loadmat_any(path):
    """
    Loads variables H_all, y_n_all, X_all from either:
      - MATLAB v7 (scipy.io.loadmat)
      - MATLAB v7.3 (HDF5 via h5py)
    Returns dict with numpy arrays (complex where appropriate).
    """
    try:
        mat = loadmat(path)
        # SciPy returns real/imag combined as complex already
        return {
            "H_all": mat["H_all"],
            "y_n_all": mat["y_n_all"],
            "X_all": mat["X_all"],
        }
    except NotImplementedError:
        pass

    # v7.3 path
    with h5py.File(path, "r") as f:
        # MATLAB stores names in HDF5 with column-major dims, so we’ll transpose later
        H_node = f["H_all"]
        y_node = f["y_n_all"]
        X_node = f["X_all"]

        H = _h5_to_complex(H_node)
        y = _h5_to_complex(y_node)
        X = _h5_to_complex(X_node)

        # MATLAB -> NumPy dimension order fix:
        # MATLAB saves as Fortran-order; h5py gives C-order arrays with reversed dims.
        # Usually we need to transpose each to match (48,48,frames) and (48,frames).
        # If your shapes look reversed (e.g., frames first), swap axes accordingly.

        # Try to coerce to expected shapes:
        def fixH(A):
            # Want (48,48,frames)
            if A.ndim == 3 and A.shape[0] in (1,48) and A.shape[1] in (1,48):
                return np.array(A)  # likely already correct
            # common swapped: (frames, 48, 48)
            if A.ndim == 3 and A.shape[-1] == 48 and A.shape[-2] == 48:
                return np.transpose(A, (1, 2, 0))
            return A

        def fixV(A):
            # Want (48, frames)
            if A.ndim == 2 and A.shape[0] == 48:
                return A
            if A.ndim == 2 and A.shape[1] == 48:
                return A.T
            return A

        H = fixH(H)
        y = fixV(y)
        X = fixV(X)

        return {"H_all": H, "y_n_all": y, "X_all": X}


# ---------------- dataset ----------------

class HEqualizerDataset(Dataset):
    def __init__(self, data_dir: str):
        super().__init__()
        self.files: List[str] = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
        if not self.files:
            raise FileNotFoundError(f"No .mat files found in {data_dir}")
        self.index_map: List[Tuple[int, int]] = []
        self.cache = []

        for fi, f in enumerate(self.files):
            mat = loadmat_any(f)                  # <-- NEW
            H_all = mat["H_all"]                  # complex (48,48,frames)
            y_all = mat["y_n_all"]                # complex (48,frames)
            X_all = mat["X_all"]                  # complex (48,frames)

            # sanity checks (optional)
            assert H_all.ndim == 3 and H_all.shape[0] == H_all.shape[1], f"Bad H shape in {f}: {H_all.shape}"
            assert y_all.ndim == 2 and X_all.ndim == 2 and y_all.shape[0] == X_all.shape[0] == H_all.shape[0], \
                f"Bad Y/X shapes in {f}: {y_all.shape}, {X_all.shape}, H:{H_all.shape}"

            frames = H_all.shape[2]
            self.cache.append((H_all, y_all, X_all))
            for k in range(frames):
                self.index_map.append((fi, k))


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        fi, k = self.index_map[idx]
        H_all, y_all, X_all = self.cache[fi]

        H = H_all[:, :, k]                 # (48,48) complex
        Y = y_all[:, k]                    # (48,)   complex  [NO NORMALIZATION]
        X = X_all[:, k]                    # (48,)   complex

        # ---- H -> [mag, phase] and normalize only H ----
        H_mp = complex_to_magphase(H)      # (2,48,48) with radians in [-pi, pi]
        # magnitude -> [0,1] per-sample
        H_mag = H_mp[0]
        mmin, mmax = H_mag.min(), H_mag.max()
        H_mag = (H_mag - mmin) / (mmax - mmin + 1e-12) if mmax > mmin else (H_mag * 0.0)
        # phase [-pi,pi] -> [0,1]
        H_phase = (H_mp[1] + np.pi) / (2 * np.pi)
        H_mp_norm = np.stack([H_mag.astype(np.float32), H_phase.astype(np.float32)], axis=0)

        # ---- Y as real/imag (keep raw; no min-max!) ----
        Y_ri = to_torch_complex_ri(Y).float()   # (48,2)

        # ---- X phase (keep raw radians) ----
        X_phase = torch.from_numpy(np.angle(X).astype(np.float32))  # (48,), in [-pi, pi]

        return torch.from_numpy(H_mp_norm), Y_ri, X_phase



# ---------------- model ----------------

class MobileNetV3PhaseHead(nn.Module):
    def __init__(self, in_ch: int = 2, out_dim: int = 48):
        super().__init__()
        m = mobilenet_v3_small(weights=None)
        # Patch first conv to accept 2 channels (mag, phase)
        first_conv = m.features[0][0]
        m.features[0][0] = nn.Conv2d(
            in_ch, first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )
        # Replace classifier head
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, out_dim)
        self.backbone = m

    def forward(self, x):  # x: (B,2,48,48)
        return self.backbone(x)  # (B,48) radians (unbounded)


# ---------------- LightningModule ----------------

class ZFMobileNetLightning(pl.LightningModule):
    def __init__(self, lr: float = 2e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = MobileNetV3PhaseHead(in_ch=2, out_dim=48)
        self.validation_losses = []  # store losses between steps

    def forward(self, H_mp: torch.Tensor) -> torch.Tensor:
        return self.model(H_mp)

    def common_step(self, batch):
        H_mp, Y_ri, X_phase = batch
        theta_hat = self(H_mp)
        Y_rot_ri = complex_mul_by_phase(Y_ri, theta_hat)
        pred_phase = angle_from_ri(Y_rot_ri)
        loss = wrapped_phase_mse(pred_phase, X_phase)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.validation_losses.append(loss.detach())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        if self.validation_losses:
            avg_loss = torch.stack(self.validation_losses).mean()
            self.log("avg_val_loss", avg_loss, prog_bar=True)
            self.validation_losses.clear()

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log("test_loss", loss, prog_bar=True)
        return {"test_loss": loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


# ---------------- DataModule ----------------

class HEqualizerDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 64, num_workers: int = 4, val_split: float = 0.1, seed: int = 42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

    def setup(self, stage=None):
        ds = HEqualizerDataset(self.data_dir)
        n = len(ds)
        n_val = max(1, int(n * self.val_split))
        n_train = n - n_val
        gen = torch.Generator().manual_seed(self.seed)
        self.ds_train, self.ds_val = torch.utils.data.random_split(ds, [n_train, n_val], generator=gen)
        self.ds_test = self.ds_val  # reuse for quick testing, or split separately

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)


# ---------------- main ----------------

def _load_weights_only(lightning_module, ckpt_path: str):
    """Load only model weights from a .ckpt or .pt into the LightningModule (no optimizer/epoch)."""
    import torch
    sd = torch.load(ckpt_path, map_location="cpu")

    # Lightning .ckpt usually has a 'state_dict' key
    if isinstance(sd, dict) and "state_dict" in sd:
        state = sd["state_dict"]
    else:
        state = sd  # raw state_dict

    # Try strict load first; if that fails, relax & strip common prefixes
    try:
        lightning_module.load_state_dict(state, strict=True)
        return
    except Exception:
        pass

    # Common prefix fix: remove "model." if present (we keep a self.model inside LM)
    fixed = {}
    for k, v in state.items():
        if k.startswith("model."):
            fixed[k[len("model."):]] = v
        else:
            fixed[k] = v

    # If still mismatched, load into inner model only
    try:
        lightning_module.load_state_dict(fixed, strict=False)
    except Exception:
        if hasattr(lightning_module, "model"):
            try:
                lightning_module.model.load_state_dict(fixed, strict=False)
            except Exception as e:
                print(f"[WARN] Could not load weights (even relaxed): {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--precision", type=str, default="32-true")  # e.g., "16-mixed"
    # NEW:
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="Resume training from this Lightning .ckpt (restores optimizer/epoch).")
    parser.add_argument("--init_ckpt", type=str, default=None,
                        help="Warm start from weights in this ckpt/.pt (does NOT restore optimizer/epoch).")
    args = parser.parse_args()

    dm = HEqualizerDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
    )
    model = ZFMobileNetLightning(lr=args.lr)

    # Warm start option (weights only)
    if args.init_ckpt and not args.resume_ckpt:
        print(f"[INFO] Warm starting from weights: {args.init_ckpt}")
        _load_weights_only(model, args.init_ckpt)

    ckpt_dir = "checkpoints_pl"
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir, filename="zf_mnv3-{epoch:02d}-{val_loss:.6f}",
            save_top_k=3, monitor="val_loss", mode="min"
        )
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        default_root_dir=".",
        callbacks=callbacks,
        log_every_n_steps=25,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
    )

    # If resume_ckpt is provided, pass it to trainer.fit so it resumes optimizer/epoch state
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume_ckpt if args.resume_ckpt else None)
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
