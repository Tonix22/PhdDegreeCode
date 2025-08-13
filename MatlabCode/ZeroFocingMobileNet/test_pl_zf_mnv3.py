# test_pl_zf_mnv3.py
import argparse, os, glob, re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small
from scipy.io import loadmat
import h5py
import matplotlib.pyplot as plt
import csv
import pytorch_lightning as pl

torch.set_float32_matmul_precision("high")  # use Tensor Cores if available

# ---------- IO helpers (v7 & v7.3) ----------

def _h5_to_complex(node):
    if isinstance(node, h5py.Group):
        return np.array(node["real"]) + 1j * np.array(node["imag"])
    if isinstance(node, h5py.Dataset):
        dt = node.dtype
        if dt.fields:
            fields = dt.fields.keys()
            data = np.array(node)
            if "real" in fields and "imag" in fields:
                return data["real"] + 1j * data["imag"]
            if "r" in fields and "i" in fields:
                return data["r"] + 1j * data["i"]
        return np.array(node)
    raise ValueError("Unsupported HDF5 node type.")

def loadmat_any(path):
    # Try v7
    try:
        mat = loadmat(path)
        out = {k: mat[k] for k in mat.keys() if not k.startswith("__")}
        return out
    except NotImplementedError:
        pass
    # v7.3
    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            node = f[k]
            try:
                arr = _h5_to_complex(node)
            except Exception:
                arr = np.array(node)
            out[k] = arr
    return out

def fixH(A):
    # want (48,48,frames)
    if A.ndim == 3 and A.shape[0] == A.shape[1]:
        return A
    if A.ndim == 3 and A.shape[-1] == A.shape[-2]:
        return np.transpose(A, (1, 2, 0))
    return A

def fixV(A):
    # want (48, frames)
    if A.ndim == 2 and A.shape[0] == 48:
        return A
    if A.ndim == 2 and A.shape[1] == 48:
        return A.T
    return A

# ---------- DSP helpers ----------

def complex_to_magphase(C: np.ndarray) -> np.ndarray:
    return np.stack([np.abs(C), np.angle(C)], axis=0).astype(np.float32)

def to_torch_ri(v: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(
        np.stack([v.real.astype(np.float32), v.imag.astype(np.float32)], axis=-1)
    )

def complex_mul_by_phase(y_ri: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    yr, yi = y_ri[..., 0], y_ri[..., 1]
    return torch.stack([yr * c - yi * s, yr * s + yi * c], dim=-1)

# QPSK (M=4) Gray constellation, UnitAveragePower
def qpsk_constellation():
    s = 1/np.sqrt(2)
    pts = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) * s
    # Gray bits for indices 0..3
    bits = np.array([[0,0],[0,1],[1,1],[1,0]], dtype=np.uint8)
    return pts, bits  # (4,), (4,2)

def qpsk_demod_to_bits(x: np.ndarray) -> np.ndarray:
    """x: complex (N,) -> bits (N,2) using nearest neighbor to Gray QPSK"""
    pts, bits = qpsk_constellation()
    x = x.reshape(-1, 1)
    d2 = np.abs(x - pts[np.newaxis, :])**2
    idx = np.argmin(d2, axis=1)
    return bits[idx]  # (N,2)

def normalize_H_mp(H_mp: np.ndarray) -> np.ndarray:
    # H_mp: (2,48,48) with [mag, phase(rad)]
    H_mag  = H_mp[0]
    H_phase= H_mp[1]

    # Example: per-sample min–max on magnitude -> [0,1]
    mmin, mmax = H_mag.min(), H_mag.max()
    if mmax > mmin:
        H_mag = (H_mag - mmin) / (mmax - mmin)
    else:
        H_mag = H_mag * 0.0

    # Map phase from [-pi, pi] -> [0,1]
    H_phase = (H_phase + np.pi) / (2 * np.pi)

    return np.stack([H_mag.astype(np.float32), H_phase.astype(np.float32)], axis=0)

# ---------- Model ----------

class MobileNetV3PhaseHead(nn.Module):
    def __init__(self, in_ch: int = 2, out_dim: int = 48):
        super().__init__()
        m = mobilenet_v3_small(weights=None)
        first_conv = m.features[0][0]
        m.features[0][0] = nn.Conv2d(
            in_ch, first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, out_dim)
        self.backbone = m

    def forward(self, x):
        return self.backbone(x)  # theta_hat (B,48)

class ZFMobileNetLightning(pl.LightningModule):
    def __init__(self, lr: float = 2e-3):
        super().__init__()
        self.model = MobileNetV3PhaseHead(in_ch=2, out_dim=48)
    def forward(self, x):
        return self.model(x)

# ---------- Dataset ----------

class HEqualizerTestSet(Dataset):
    """
    Expects each .mat to have: H_all (48,48,F), y_n_all (48,F), X_all (48,F)
    Also tries to read SNR (scalar) or parse from filename.
    """
    def __init__(self, data_dir: str):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
        if not self.files:
            raise FileNotFoundError(f"No .mat files in {data_dir}")
        self.index_map = []  # (file_idx, frame_idx)
        self.cache = []      # list of dicts per file: {'H':, 'Y':, 'X':, 'SNR':, 'modorder':}
        for fi, f in enumerate(self.files):
            md = loadmat_any(f)
            H_all = fixH(md["H_all"])
            Y_all = fixV(md["y_n_all"])
            X_all = fixV(md["X_all"])
            SNR = md.get("SNR", None)
            if isinstance(SNR, np.ndarray):
                SNR = float(np.squeeze(SNR))
            if SNR is None:
                # parse from filename '..._SNR_35dB.mat'
                m = re.search(r"SNR[_\- ]*(-?\d+)\s*dB", os.path.basename(f), flags=re.IGNORECASE)
                SNR = float(m.group(1)) if m else float("nan")
            modorder = md.get("modorder", 4)
            if isinstance(modorder, np.ndarray):
                modorder = int(np.squeeze(modorder))

            frames = H_all.shape[2]
            self.cache.append(dict(H=H_all, Y=Y_all, X=X_all, SNR=SNR, modorder=modorder))
            for k in range(frames):
                self.index_map.append((fi, k))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        fi, k = self.index_map[idx]
        item = self.cache[fi]
        H = item["H"][:, :, k]      # (48,48) complex
        Y = item["Y"][:, k]         # (48,)   complex  (KEEP RAW)
        X = item["X"][:, k]         # (48,)   complex  (KEEP RAW)

        H_mp = complex_to_magphase(H)          # (2,48,48)
        H_mp = normalize_H_mp(H_mp)            # <- same as training

        Y_ri = to_torch_ri(Y).float()          # DO NOT normalize or shift
        X_c  = X.astype(np.complex64)          # raw for demod

        SNR = item["SNR"]; modorder = item["modorder"]
        return torch.from_numpy(H_mp), Y_ri, X_c, SNR, modorder

# ---------- Runner ----------

@torch.no_grad()
def evaluate_ber(model, loader, device):
    model.eval()
    snr_stats = {}  # snr -> {'err': int, 'tot': int}

    for H_mp, Y_ri, X_c, SNR, modorder in loader:
        # move tensors
        H_mp = H_mp.to(device)      # (B,2,48,48)
        Y_ri = Y_ri.to(device)      # (B,48,2)

        theta_hat = model(H_mp)     # (B,48)
        Y_rot_ri = complex_mul_by_phase(Y_ri, theta_hat)  # (B,48,2)
        # back to numpy complex for demod
        Yr = Y_rot_ri[..., 0].cpu().numpy()
        Yi = Y_rot_ri[..., 1].cpu().numpy()
        Y_hat = Yr + 1j * Yi        # (B,48)

        X_c = np.stack(X_c, axis=0) # (B,48) complex from list

        # Demod both (QPSK assumed)
        bits_hat = qpsk_demod_to_bits(Y_hat.reshape(-1))   # (B*48, 2)
        bits_ref = qpsk_demod_to_bits(X_c.reshape(-1))     # (B*48, 2)

        # XOR and count
        err_bits = np.count_nonzero(bits_hat ^ bits_ref)
        tot_bits = bits_hat.size

        # aggregate per SNR (SNR is a list/array per batch; use each item)
        for i in range(len(SNR)):
            snr = float(SNR[i])
            # split this sample's slice
            bh_i = bits_hat[i*48:(i+1)*48]
            br_i = bits_ref[i*48:(i+1)*48]
            e_i = np.count_nonzero(bh_i ^ br_i)
            t_i = bh_i.size
            d = snr_stats.setdefault(snr, {"err": 0, "tot": 0})
            d["err"] += int(e_i)
            d["tot"] += int(t_i)

    # compute BER per SNR sorted
    snrs = sorted([k for k in snr_stats.keys() if not np.isnan(k)], reverse=False)
    ber = [snr_stats[s]["err"] / snr_stats[s]["tot"] for s in snrs]
    return snrs, ber

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=None, help="Lightning .ckpt to load")
    ap.add_argument("--state_dict", type=str, default=None, help="Raw state_dict .pt to load")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_csv", type=str, default="ber_results.csv")
    ap.add_argument("--out_plot", type=str, default="ber_curve.jpg")
    args = ap.parse_args()

    # Model
    device = torch.device(args.device)
    if args.ckpt:
        model = ZFMobileNetLightning.load_from_checkpoint(args.ckpt, map_location=device).model
    else:
        model = MobileNetV3PhaseHead(in_ch=2, out_dim=48)
        if args.state_dict:
            sd = torch.load(args.state_dict, map_location="cpu")
            # allow Lightning-style 'model.backbone...' keys
            if "state_dict" in sd:
                sd = sd["state_dict"]
                sd = {k.replace("model.", "", 1): v for k, v in sd.items()}  # strip 'model.' prefix if present
            try:
                model.load_state_dict(sd, strict=False)
            except Exception as e:
                print("State dict load warning:", e)
        model = model.to(device)
    model.eval()

    # Data
    ds = HEqualizerTestSet(args.data_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True, collate_fn=None)

    # Eval
    snrs, ber = evaluate_ber(model, dl, device)

    # Save CSV
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["SNR_dB", "BER"])
        for s, b in zip(snrs, ber):
            w.writerow([s, b])
    print(f"Saved CSV -> {args.out_csv}")

    # Plot
    plt.figure()
    if len(ber) > 0:
        plt.semilogy(snrs, ber, marker="o")
        plt.grid(True, which="both")
        plt.xlabel("SNR (dB)")
        plt.ylabel("BER")
        plt.title("BER vs SNR (NN equalizer)")
        plt.savefig(args.out_plot, dpi=150, bbox_inches="tight")
        print(f"Saved plot -> {args.out_plot}")
    else:
        print("No SNR points to plot.")

if __name__ == "__main__":
    main()
