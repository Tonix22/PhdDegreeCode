# test_pl_zf_mnv3_multitask_v2_ber.py
# Eval BER coherente con ZFMobileNetMultiTask (misma ruta de inferencia que train).
# - Usa H -> [zscore(log1p|H|), cos∠H, sin∠H] (3,48,48)
# - θ̂ = phase_net(H); Z = (H^H y_n)·e^{jθ̂}; feats(Z) -> cls_head -> clases
# - Convierte clases -> bits con un LUT clase→bits APRENDIDO del .mat (si hay txcls_all+txbits_all)
# - BER contra txbits_all; CSV + plot; tqdm.

import argparse, os, glob, re, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small
from scipy.io import loadmat
import h5py
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pytorch_lightning as pl

torch.set_float32_matmul_precision("high")

# ---------- IO helpers ----------
def _h5_to_complex(node):
    if isinstance(node, h5py.Group):
        return np.array(node["real"]) + 1j * np.array(node["imag"])
    if isinstance(node, h5py.Dataset):
        dt = node.dtype
        if dt.fields:
            fields = node.dtype.fields.keys()
            data = np.array(node)
            if "real" in fields and "imag" in fields:
                return data["real"] + 1j * data["imag"]
            if "r" in fields and "i" in fields:
                return data["r"] + 1j * data["i"]
        return np.array(node)
    raise ValueError("Unsupported HDF5 node type.")

def loadmat_any(path):
    try:
        mat = loadmat(path)
        return {k: mat[k] for k in mat.keys() if not k.startswith("__")}
    except NotImplementedError:
        pass
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
    if A.ndim == 3 and A.shape[0] == A.shape[1]:
        return A
    if A.ndim == 3 and A.shape[-1] == A.shape[-2]:
        return np.transpose(A, (1, 2, 0))
    return A

def fixV(A):
    if A.ndim == 2 and A.shape[0] == 48:
        return A
    if A.ndim == 2 and A.shape[1] == 48:
        return A.T
    return A

def fixBits(A, M, bitsPerSym):
    A = np.array(A)
    assert A.ndim == 2, f"txbits_all debe ser 2D, got {A.shape}"
    expected = M * bitsPerSym
    if A.shape[0] == expected:
        return A
    if A.shape[1] == expected:
        return A.T
    raise ValueError(f"txbits_all shape {A.shape}; esperado (*,{expected}) o ({expected},*).")

# ---------- DSP helpers ----------
def to_torch_ri(v: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(
        np.stack([v.real.astype(np.float32), v.imag.astype(np.float32)], axis=-1)
    )

def complex_mul_by_phase_ri(y_ri: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    yr, yi = y_ri[..., 0], y_ri[..., 1]
    out_r = yr * c - yi * s
    out_i = yr * s + yi * c
    return torch.stack([out_r, out_i], dim=-1)

def angle_from_ri(ri: torch.Tensor) -> torch.Tensor:
    return torch.atan2(ri[..., 1], ri[..., 0])

# ---------- Modelo (idéntico al train) ----------
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
    def forward(self, x):
        return self.backbone(x)

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

class ZFMobileNetMultiTask(pl.LightningModule):
    def __init__(self, lr: float = 2e-3,
                 lambda_phase: float = 1.0,
                 lambda_smooth: float = 1e-2,
                 lambda_cls: float = 1.0):
        super().__init__()
        self.save_hyperparameters()
        self.phase_net = MobileNetV3PhaseHead(in_ch=3, out_dim=48)
        self.cls_head  = GlobalFCClassifier(in_ch=3, M=48, hid=512, num_classes=4, dropout=0.1)
    def forward(self, H_enc: torch.Tensor):
        return self.phase_net(H_enc)
    def _build_Z_feats(self, Ymf_ri: torch.Tensor, theta_hat: torch.Tensor) -> torch.Tensor:
        Z_ri   = complex_mul_by_phase_ri(Ymf_ri, theta_hat)
        Z_ang  = angle_from_ri(Z_ri)
        Z_mag  = torch.linalg.norm(Z_ri, dim=-1)
        feats  = torch.stack([torch.log1p(Z_mag), torch.cos(Z_ang), torch.sin(Z_ang)], dim=1)
        return feats

# ---------- Dataset: aprende LUT clase→bits si hay txcls_all + txbits_all ----------
class HEqualizerBERTestSet(Dataset):
    """
    Requiere: H_all (48,48,F), y_n_all (48,F) = H^H y_n
    Para BER: txbits_all (M*2,F).
    Si además hay txcls_all (M,F), se aprende LUT clase→bits por mayoría simple.
    Devuelve: H_enc (3,48,48), Ymf_ri (48,2), gt_bits (48,2), SNR
    """
    def __init__(self, data_dir: str):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
        if not self.files:
            raise FileNotFoundError(f"No .mat files in {data_dir}")
        self.index_map = []
        self.cache = []
        self.cls2bits = None  # np.array (4,2) si se puede estimar

        acc_sum = 0.0; acc_sumsq = 0.0; acc_n = 0
        # contadores para LUT
        cls_counts = np.zeros((4,4), dtype=np.int64)  # [class, code(0..3)] donde code = b0*2+b1

        for fi, f in enumerate(self.files):
            md = loadmat_any(f)
            H_all   = fixH(md["H_all"])
            Ymf_all = fixV(md["y_n_all"])
            M = H_all.shape[0]; bitsPerSym = 2

            if "txbits_all" not in md:
                raise ValueError(f'{os.path.basename(f)} no contiene "txbits_all".')
            txbits_all = fixBits(md["txbits_all"], M=M, bitsPerSym=bitsPerSym)  # (M*2, F)

            txcls_all = md.get("txcls_all", None)
            if txcls_all is not None:
                txcls_all = fixV(txcls_all)  # (M,F)

            SNR = md.get("SNR", None)
            if isinstance(SNR, np.ndarray): SNR = float(np.squeeze(SNR))
            if SNR is None:
                m = re.search(r"SNR[_\- ]*(-?\d+)\s*dB", os.path.basename(f), flags=re.IGNORECASE)
                SNR = float(m.group(1)) if m else float("nan")

            # stats H
            logH = np.log1p(np.abs(H_all)).astype(np.float64)
            acc_sum += logH.sum(); acc_sumsq += (logH**2).sum(); acc_n += logH.size

            frames = H_all.shape[2]
            self.cache.append(dict(H=H_all, Y=Ymf_all, txbits=txbits_all, SNR=SNR, M=M, txcls=txcls_all))
            for k in range(frames):
                self.index_map.append((fi, k))

            # acumular LUT si hay clases
            if txcls_all is not None:
                b0 = txbits_all[0::2, :]  # (M,F)
                b1 = txbits_all[1::2, :]  # (M,F)
                code = (b0.astype(np.int64) << 1) | b1.astype(np.int64)  # 0..3
                for c in range(4):
                    mask = (txcls_all == c)
                    if mask.any():
                        vals = code[mask]
                        binc = np.bincount(vals, minlength=4)
                        cls_counts[c] += binc

        # zscore de H
        self.logH_mean = float(acc_sum / max(acc_n,1))
        var = max(acc_sumsq / max(acc_n,1) - self.logH_mean**2, 1e-12)
        self.logH_std = float(np.sqrt(var))

        # construir LUT por mayoría si hay info
        if cls_counts.sum() > 0:
            lut = np.zeros((4,2), dtype=np.uint8)
            for c in range(4):
                if cls_counts[c].sum() == 0:
                    # fallback Gray estándar si alguna clase no apareció
                    # 0:00,1:01,2:11,3:10
                    gray = np.array([[0,0],[0,1],[1,1],[1,0]], dtype=np.uint8)
                    lut[c] = gray[c]
                else:
                    code = int(np.argmax(cls_counts[c]))  # 0..3
                    lut[c,0] = (code >> 1) & 1
                    lut[c,1] = code & 1
            self.cls2bits = lut  # (4,2)

    def __len__(self): return len(self.index_map)

    def __getitem__(self, idx):
        fi, k = self.index_map[idx]
        it = self.cache[fi]
        H   = it["H"][:,:,k]
        Ymf = it["Y"][:,k]
        txb = it["txbits"][:,k]         # (M*2,)
        SNR = it["SNR"]; M = it["M"]

        # H -> 3 canales
        H_mag = np.abs(H).astype(np.float32)
        H_ph  = np.angle(H).astype(np.float32)
        H_log = (np.log1p(H_mag) - self.logH_mean) / (self.logH_std + 1e-6)
        H_enc = np.stack([H_log, np.cos(H_ph), np.sin(H_ph)], axis=0).astype(np.float32)  # (3,48,48)

        gt_bits = np.asarray(txb, dtype=np.uint8).reshape(M, 2, order='C')  # (48,2)

        return (torch.from_numpy(H_enc),
                to_torch_ri(Ymf).float(),
                torch.from_numpy(gt_bits),
                SNR)

# ---------- Utils BER ----------
def build_bits2cls_from_lut(cls2bits: torch.Tensor) -> torch.Tensor:
    """
    cls2bits: (4,2) uint8 -> bits2cls: (2,2) long para indexar [b0,b1].
    """
    m = torch.full((2,2), -1, dtype=torch.long, device=cls2bits.device)
    for c in range(4):
        b0 = int(cls2bits[c,0].item())
        b1 = int(cls2bits[c,1].item())
        m[b0, b1] = c
    return m

@torch.no_grad()
def evaluate_ber(model: ZFMobileNetMultiTask, loader, device, cls2bits_np=None):
    model.eval()
    # LUTs
    if cls2bits_np is None:
        # Gray estándar si no pudimos estimar
        cls2bits_np = np.array([[0,0],[0,1],[1,1],[1,0]], dtype=np.uint8)
    cls2bits = torch.from_numpy(cls2bits_np).to(device=device, dtype=torch.uint8)   # (4,2)
    bits2cls = build_bits2cls_from_lut(cls2bits)                                    # (2,2)

    snr_stats = {}  # snr -> {'err': int, 'tot': int}
    tot_err = 0; tot_bits = 0
    cls_ok = 0; cls_tot = 0

    for H_enc, Ymf_ri, gt_bits, SNR in tqdm(loader, total=len(loader), desc="Testing", unit="batch", dynamic_ncols=True):
        H_enc  = H_enc.to(device)                         # (B,3,48,48)
        Ymf_ri = Ymf_ri.to(device)                        # (B,48,2)
        gt_bits = (gt_bits.to(device).to(torch.uint8) & 1)# (B,48,2)

        # θ̂ y clases
        theta_hat = model(H_enc)                          # (B,48)
        Z_feats   = model._build_Z_feats(Ymf_ri, theta_hat)   # (B,3,48)
        logits    = model.cls_head(Z_feats)                   # (B,4,48)
        pred_cls  = logits.argmax(dim=1)                      # (B,48)

        # clases -> bits con LUT aprendida
        pred_bits = cls2bits[pred_cls]                        # (B,48,2) uint8

        # BER
        err_bits_batch = (pred_bits ^ gt_bits).sum(dim=(1,2)).cpu().numpy()
        tot_bits_batch = pred_bits.size(1) * pred_bits.size(2)  # 48*2
        tot_err  += int(err_bits_batch.sum())
        tot_bits += int(tot_bits_batch * pred_bits.size(0))

        # Sanidad: acc de clase derivando clase GT desde bits via LUT inversa
        gt_cls = bits2cls[gt_bits[...,0].long(), gt_bits[...,1].long()]  # (B,48)
        cls_ok += int((pred_cls == gt_cls).sum().item())
        cls_tot += int(gt_cls.numel())

        # agregación por SNR
        if torch.is_tensor(SNR):
            snr_arr = SNR.detach().cpu().numpy().reshape(-1)
        elif isinstance(SNR, np.ndarray):
            snr_arr = SNR.reshape(-1)
        elif isinstance(SNR, (list, tuple)):
            snr_arr = np.asarray(SNR, dtype=np.float32).reshape(-1)
        else:
            snr_arr = np.asarray([SNR], dtype=np.float32)
        for i in range(pred_bits.size(0)):
            e_i = int(err_bits_batch[i]); t_i = int(tot_bits_batch); s = float(snr_arr[i])
            d = snr_stats.setdefault(s, {"err":0, "tot":0})
            d["err"] += e_i; d["tot"] += t_i

    snrs = sorted([k for k in snr_stats.keys() if not np.isnan(k)])
    ber  = [snr_stats[s]["err"] / max(1, snr_stats[s]["tot"]) for s in snrs]
    overall_ber = tot_err / max(1, tot_bits)
    overall_acc = cls_ok / max(1, cls_tot)
    return snrs, ber, overall_ber, overall_acc

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint .ckpt de ZFMobileNetMultiTask")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_csv", type=str, default="ber_results.csv")
    ap.add_argument("--out_plot", type=str, default="ber_curve.jpg")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Cargar EXACTAMENTE el multitask
    model = ZFMobileNetMultiTask.load_from_checkpoint(args.ckpt, map_location=device)
    assert isinstance(model, ZFMobileNetMultiTask), "El ckpt no corresponde a ZFMobileNetMultiTask"
    model = model.to(device).eval()
    print("Usando modelo:", type(model).__name__)

    # Data
    ds = HEqualizerBERTestSet(args.data_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # LUT aprendida si existe
    cls2bits_np = ds.cls2bits  # puede ser None -> Gray fallback dentro de evaluate_ber
    if cls2bits_np is not None:
        print("LUT clase→bits aprendida del dataset:", cls2bits_np.tolist())
    else:
        print("No se pudo aprender LUT; se usará Gray por defecto.")

    # Eval
    snrs, ber, overall, cls_acc = evaluate_ber(model, dl, device, cls2bits_np)

    print(f"BER global: {overall:.6e}")
    print(f"Accuracy de clase (sanidad): {cls_acc:.4f}")

    # CSV
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["SNR_dB", "BER"])
        for s, b in zip(snrs, ber): w.writerow([s, b])
    print(f"CSV guardado -> {args.out_csv}")

    # Plot
    if len(ber) > 0:
        plt.figure()
        plt.semilogy(snrs, ber, marker="o")
        plt.grid(True, which="both")
        plt.xlabel("SNR (dB)"); plt.ylabel("BER")
        plt.title("BER vs SNR (ZFMobileNetMultiTask)")
        plt.savefig(args.out_plot, dpi=150, bbox_inches="tight")
        print(f"Gráfica guardada -> {args.out_plot}")
    else:
        print("No hay puntos de SNR para graficar.")

if __name__ == "__main__":
    main()
