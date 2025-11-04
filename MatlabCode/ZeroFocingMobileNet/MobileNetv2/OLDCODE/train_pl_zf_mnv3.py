# train_pl_zf_mnv3_multitask_v2.py
# Etapa 1: regresión de fase (θ_hat) desde H
# Etapa 2: clasificación QPSK por subportadora usando Z = (H^H y_n)·e^{jθ_hat}
# Pérdida total: L = λ_phase * L_phase + λ_smooth * L_smooth + λ_cls * CE(48×4)

import os, glob, argparse
import numpy as np
from typing import List, Tuple
import h5py
import torch
torch.set_float32_matmul_precision("high")
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small
from scipy.io import loadmat
import pytorch_lightning as pl

# ---------------- utils ----------------

def wrapped_phase(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))

def wrapped_phase_mse(pred_phase: torch.Tensor, tgt_phase: torch.Tensor) -> torch.Tensor:
    d = pred_phase - tgt_phase
    d_wrapped = torch.atan2(torch.sin(d), torch.cos(d))
    return torch.mean(d_wrapped ** 2)

def wrapped_smoothness_loss(theta: torch.Tensor) -> torch.Tensor:
    diff = theta[..., 1:] - theta[..., :-1]
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return (diff * diff).mean()

def complex_to_ri(y: np.ndarray) -> np.ndarray:
    return np.stack([y.real.astype(np.float32), y.imag.astype(np.float32)], axis=-1)  # (...,2)

def complex_mul_by_phase_ri(y_ri: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    # y_ri: (B,48,2); theta: (B,48)
    c, s = torch.cos(theta), torch.sin(theta)
    yr, yi = y_ri[..., 0], y_ri[..., 1]
    out_r = yr * c - yi * s
    out_i = yr * s + yi * c
    return torch.stack([out_r, out_i], dim=-1)  # (B,48,2)

def angle_from_ri(ri: torch.Tensor) -> torch.Tensor:
    return torch.atan2(ri[..., 1], ri[..., 0])  # (...,)

def _h5_to_complex(arr_or_group):
    if isinstance(arr_or_group, h5py.Group):
        real = np.array(arr_or_group["real"])
        imag = np.array(arr_or_group["imag"])
        return real + 1j * imag
    elif isinstance(arr_or_group, h5py.Dataset):
        dt = arr_or_group.dtype
        if dt.fields:
            fields = dt.fields.keys()
            data = np.array(arr_or_group)
            if "real" in fields and "imag" in fields:
                return data["real"] + 1j * data["imag"]
            if "r" in fields and "i" in fields:
                return data["r"] + 1j * data["i"]
        return np.array(arr_or_group)
    else:
        raise ValueError("Unsupported HDF5 node type.")

def loadmat_any(path):
    import numpy as np
    try:
        mat = loadmat(path)
        out = {
            "H_all":   mat["H_all"],
            "y_n_all": mat["y_n_all"],
            "X_all":   mat["X_all"],
        }
        # NUEVO: si viene en v7 (no h5py), no suele necesitar fix, pero normalizamos dtype
        if "txcls_all" in mat:
            out["txcls_all"] = mat["txcls_all"].astype(np.uint8)
        if "txbits_all" in mat:
            out["txbits_all"] = mat["txbits_all"].astype(np.bool_)
        return out
    except NotImplementedError:
        pass

    with h5py.File(path, "r") as f:
        H_node = f["H_all"]
        y_node = f["y_n_all"]
        X_node = f["X_all"]

        H = _h5_to_complex(H_node)
        y = _h5_to_complex(y_node)
        X = _h5_to_complex(X_node)

        # Helpers para corregir orientación
        def fixH(A):
            # (48,48,frames)
            if A.ndim == 3 and A.shape[0] in (1,48) and A.shape[1] in (1,48):
                return np.array(A)
            if A.ndim == 3 and A.shape[-1] == 48 and A.shape[-2] == 48:
                return np.transpose(A, (1, 2, 0))
            return A

        def fixV(A):
            # (48, frames)
            if A.ndim == 2 and A.shape[0] == 48:
                return A
            if A.ndim == 2 and A.shape[1] == 48:
                return A.T
            return A

        H = fixH(H)
        y = fixV(y)
        X = fixV(X)

        out = {"H_all": H, "y_n_all": y, "X_all": X}

        # NUEVO: corregir orientación de txcls_all (y txbits_all si existe)
        if "txcls_all" in f:
            txc = np.array(f["txcls_all"])
            txc = fixV(txc)
            out["txcls_all"] = txc.astype(np.uint8)

        if "txbits_all" in f:
            txb = np.array(f["txbits_all"])
            # bits suelen ser (M*bitsPerSym, frames) -> no usar fixV aquí
            out["txbits_all"] = txb.astype(np.bool_)

        return out


# ---------------- dataset ----------------

class HEqualizerMultiTaskDataset(Dataset):
    """
    Devuelve:
      - H_enc:   (3,48,48)  = [ zscore(log1p(|H|)), cos(angle(H)), sin(angle(H)) ]
      - Ymf_ri:  (48,2)     = matched-filter output H^H y_n  (en .mat: y_n_all)
      - theta_tgt: (48,)    = wrap(angle(X) - angle(Ymf))
      - cls_idx: (48,) int64 = clase QPSK de X (0..3) desde .mat (txcls_all)
    """
    def __init__(self, data_dir: str):
        super().__init__()
        self.files: List[str] = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
        if not self.files:
            raise FileNotFoundError(f"No .mat files found in {data_dir}")
        self.index_map: List[Tuple[int, int]] = []
        self.cache = []

        # Acumular estadísticas globales de log|H|
        acc_H_sum = acc_H_sumsq = 0.0
        nH = 0

        for fi, f in enumerate(self.files):
            mat = loadmat_any(f)
            H_all  = mat["H_all"]      # (48,48,frames)
            Ymf_all= mat["y_n_all"]    # (48,frames) = H^H y_n
            X_all  = mat["X_all"]      # (48,frames)
            txcls_all = mat.get("txcls_all", None)

            assert H_all.ndim == 3 and H_all.shape[:2] == (48,48)
            assert Ymf_all.ndim == 2 and X_all.ndim == 2 and Ymf_all.shape[0] == X_all.shape[0] == 48
            if txcls_all is None:
                raise ValueError(f'Archivo {f} no contiene "txcls_all". Re-genera el dataset con el script MATLAB nuevo.')

            # stats globales
            logH = np.log1p(np.abs(H_all)).astype(np.float64)
            acc_H_sum += logH.sum(); acc_H_sumsq += (logH**2).sum(); nH += logH.size

            frames = H_all.shape[2]
            self.cache.append((H_all, Ymf_all, X_all, txcls_all))
            for k in range(frames):
                self.index_map.append((fi, k))

        self.logH_mean = float(acc_H_sum / max(nH,1))
        varH = max(acc_H_sumsq / max(nH,1) - self.logH_mean**2, 1e-12)
        self.logH_std  = float(np.sqrt(varH))

    def __len__(self): return len(self.index_map)

    def __getitem__(self, idx):
        fi, k = self.index_map[idx]
        H_all, Ymf_all, X_all, txcls_all = self.cache[fi]

        H    = H_all[:,:,k]           # (48,48) complex
        Ymf  = Ymf_all[:,k]           # (48,)   complex
        X    = X_all[:,k]             # (48,)   complex
        cls  = txcls_all[:,k].astype(np.int64)  # (48,), 0..3

        # Codificación H (3 canales)
        H_mag  = np.abs(H).astype(np.float32)
        H_ph   = np.angle(H).astype(np.float32)
        H_log  = (np.log1p(H_mag) - self.logH_mean) / (self.logH_std + 1e-6)
        H_enc  = np.stack([H_log, np.cos(H_ph), np.sin(H_ph)], axis=0).astype(np.float32)  # (3,48,48)

        # Target de fase: wrap(angle(X) - angle(Ymf))
        theta_tgt = np.angle(X).astype(np.float32) - np.angle(Ymf).astype(np.float32)
        theta_tgt = np.arctan2(np.sin(theta_tgt), np.cos(theta_tgt)).astype(np.float32)     # (48,)

        Ymf_ri = complex_to_ri(Ymf)    # (48,2)

        return (torch.from_numpy(H_enc),
                torch.from_numpy(Ymf_ri),
                torch.from_numpy(theta_tgt),
                torch.from_numpy(cls))

# ---------------- modelos ----------------

class MobileNetV3PhaseHead(nn.Module):
    def __init__(self, in_ch: int = 3, out_dim: int = 48):
        super().__init__()
        m = mobilenet_v3_small(weights=None)
        first_conv = m.features[0][0]
        m.features[0][0] = nn.Conv2d(in_ch, first_conv.out_channels,
                                     kernel_size=first_conv.kernel_size,
                                     stride=first_conv.stride,
                                     padding=first_conv.padding,
                                     bias=False)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, out_dim)
        self.backbone = m

    def forward(self, x):  # (B,in_ch,48,48)
        return self.backbone(x)  # (B,48)

class GlobalFCClassifier(nn.Module):
    """
    Clasificador global que modela dependencias entre TODAS las subportadoras.
    Entrada:  feats (B, C=3, M=48) con C=[log1p|Z|, cos φ(Z), sin φ(Z)]
    Salida:   logits (B, 4, M) para QPSK (4 clases por subportadora)
    """
    def __init__(self, in_ch=3, M=48, hid=512, num_classes=4, dropout=0.1):
        super().__init__()
        self.M = M
        self.num_classes = num_classes
        in_dim = in_ch * M
        out_dim = num_classes * M

        self.fc1   = nn.Linear(in_dim, hid)
        self.norm1 = nn.LayerNorm(hid)
        self.fc2   = nn.Linear(hid, hid)
        self.norm2 = nn.LayerNorm(hid)
        self.fc_out= nn.Linear(hid, out_dim)
        self.act   = nn.SiLU()
        self.drop  = nn.Dropout(dropout)

    def forward(self, feats):  # feats: (B,3,48)
        B, C, M = feats.shape
        assert M == self.M, f"Esperaba M={self.M}, llegó M={M}"
        x = feats.reshape(B, C * M)          # (B, C*M) -> concatena todas las subportadoras
        x = self.drop(self.act(self.norm1(self.fc1(x))))
        x = self.drop(self.act(self.norm2(self.fc2(x))))
        x = self.fc_out(x)                   # (B, 4*M)
        logits = x.view(B, self.num_classes, self.M)  # (B,4,48)
        return logits

# ---------------- LightningModule ----------------

class ZFMobileNetMultiTask(pl.LightningModule):
    def __init__(self, lr: float = 2e-3,
                 lambda_phase: float = 1.0,
                 lambda_smooth: float = 1e-2,
                 lambda_cls: float = 1.0):
        super().__init__()
        self.save_hyperparameters()
        self.phase_net = MobileNetV3PhaseHead(in_ch=3, out_dim=48)
        self.cls_head  = GlobalFCClassifier(in_ch=3, M=48, hid=512, num_classes=4, dropout=0.1)
        self.validation_losses = []

    def forward(self, H_enc: torch.Tensor):
        return self.phase_net(H_enc)  # θ_hat (B,48)

    def _build_Z_feats(self, Ymf_ri: torch.Tensor, theta_hat: torch.Tensor) -> torch.Tensor:
        # Z = Ymf * e^{j θ_hat}  -> features: [log1p(|Z|), cos φ(Z), sin φ(Z)]
        Z_ri   = complex_mul_by_phase_ri(Ymf_ri, theta_hat)   # (B,48,2)
        Z_ang  = angle_from_ri(Z_ri)                          # (B,48)
        Z_mag  = torch.linalg.norm(Z_ri, dim=-1)              # (B,48)
        feats  = torch.stack([torch.log1p(Z_mag), torch.cos(Z_ang), torch.sin(Z_ang)], dim=1)  # (B,3,48)
        return feats

    def common_step(self, batch):
        H_enc, Ymf_ri, theta_tgt, cls_idx = batch   # (B,3,48,48), (B,48,2), (B,48), (B,48)

        # --- Etapa 1: regresión de fase
        theta_hat = self(H_enc)  # (B,48)
        loss_phase = wrapped_phase_mse(theta_hat, theta_tgt)
        loss_smooth = wrapped_smoothness_loss(theta_hat)

        # --- Etapa 2: clasificación QPSK (CrossEntropy por subportadora)
        Z_feats = self._build_Z_feats(Ymf_ri, theta_hat)        # (B,3,48)
        logits  = self.cls_head(Z_feats)                        # (B,4,48)
        loss_cls = F.cross_entropy(logits, cls_idx)             # CE en 48 posiciones

        loss = (self.hparams.lambda_phase * loss_phase +
                self.hparams.lambda_smooth * loss_smooth +
                self.hparams.lambda_cls   * loss_cls)

        # Métricas
        with torch.no_grad():
            err = wrapped_phase(theta_hat - theta_tgt).abs()
            mae_deg = err.mean() * (180.0/np.pi)
            pred = logits.argmax(dim=1)                         # (B,48)
            acc  = (pred == cls_idx).float().mean()

        return loss, loss_phase, loss_smooth, loss_cls, mae_deg, acc

    def training_step(self, batch, batch_idx):
        loss, lph, lsm, lcl, mae_deg, acc = self.common_step(batch)
        self.log_dict({
            "train_loss": loss, "train_l_phase": lph, "train_l_smooth": lsm,
            "train_l_cls": lcl, "train_mae_deg": mae_deg, "train_acc": acc
        }, prog_bar=True, on_step=True, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, lph, lsm, lcl, mae_deg, acc = self.common_step(batch)
        self.validation_losses.append(loss.detach())
        self.log_dict({
            "val_loss": loss, "val_l_phase": lph, "val_l_smooth": lsm,
            "val_l_cls": lcl, "val_mae_deg": mae_deg, "val_acc": acc
        }, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        if self.validation_losses:
            avg_loss = torch.stack(self.validation_losses).mean()
            self.log("avg_val_loss", avg_loss, prog_bar=True)
            self.validation_losses.clear()

    def test_step(self, batch, batch_idx):
        loss, lph, lsm, lcl, mae_deg, acc = self.common_step(batch)
        self.log_dict({
            "test_loss": loss, "test_l_phase": lph, "test_l_smooth": lsm,
            "test_l_cls": lcl, "test_mae_deg": mae_deg, "test_acc": acc
        }, prog_bar=True)
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
        ds = HEqualizerMultiTaskDataset(self.data_dir)
        n = len(ds)
        n_val = max(1, int(n * self.val_split))
        n_train = n - n_val
        g = torch.Generator().manual_seed(self.seed)
        self.ds_train, self.ds_val = torch.utils.data.random_split(ds, [n_train, n_val], generator=g)
        self.ds_test = self.ds_val

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
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    try:
        lightning_module.load_state_dict(state, strict=True); return
    except Exception:
        pass
    fixed = {}
    for k,v in state.items():
        fixed[k] = v
    try:
        lightning_module.load_state_dict(fixed, strict=False)
    except Exception as e:
        print(f"[WARN] Could not load weights: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--lambda_phase", type=float, default=1.0)
    parser.add_argument("--lambda_smooth", type=float, default=1e-2)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--init_ckpt", type=str, default=None)
    args = parser.parse_args()

    dm = HEqualizerDataModule(args.data_dir, batch_size=args.batch_size,
                              num_workers=args.num_workers, val_split=args.val_split)
    model = ZFMobileNetMultiTask(lr=args.lr,
                                 lambda_phase=args.lambda_phase,
                                 lambda_smooth=args.lambda_smooth,
                                 lambda_cls=args.lambda_cls)

    if args.init_ckpt and not args.resume_ckpt:
        print(f"[INFO] Warm starting from: {args.init_ckpt}")
        _load_weights_only(model, args.init_ckpt)

    ckpt_dir = "checkpoints_pl"
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        pl.callbacks.ModelCheckpoint(dirpath=ckpt_dir,
            filename="zf_mnv3_mt_v2-{epoch:02d}-{val_loss:.6f}",
            save_top_k=3, monitor="val_loss", mode="min")
    ]
    trainer = pl.Trainer(max_epochs=args.epochs, default_root_dir=".",
                         callbacks=callbacks, log_every_n_steps=25,
                         accelerator="auto", devices="auto", precision=args.precision)
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume_ckpt if args.resume_ckpt else None)
    trainer.test(model, datamodule=dm, ckpt_path="best")

if __name__ == "__main__":
    main()
