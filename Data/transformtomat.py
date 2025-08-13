#!/usr/bin/env python3
"""
Convertir un .npy a .mat para MATLAB
------------------------------------

Uso:
    python npy2mat.py \
        --in  /home/tonix/Documents/PhdDegreeCode/Data/kaggle_dataset/v2v80211p_LOS.npy \
        --out /home/tonix/Documents/PhdDegreeCode/Data/kaggle_dataset/v2v80211p_LOS.mat \
        --varname v2vData
"""
import argparse
import numpy as np
from scipy.io import savemat
from pathlib import Path

def npy_to_mat(npy_path: str | Path,
               mat_path: str | Path,
               varname: str = "data"):
    """Cargar npy y guardarlo en mat con la clave <varname>."""
    npy_path, mat_path = Path(npy_path), Path(mat_path)

    # 1. Cargar el archivo .npy (soporta memoria mapeada si fuera necesario)
    arr = np.load(npy_path, mmap_mode=None)  # usar mmap_mode="r" para archivos muy grandes

    # 2. Guardar en .mat.  savemat espera un dict {nombre_variable: ndarray}
    savemat(mat_path, {varname: arr})

    print(f"✓ Guardado: {mat_path}  (varname='{varname}', shape={arr.shape}, dtype={arr.dtype})")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convertir .npy → .mat")
    p.add_argument("--in",  dest="npy",   required=True, help=".npy de entrada")
    p.add_argument("--out", dest="mat",   required=True, help=".mat de salida")
    p.add_argument("--varname", default="data", help="Nombre de la variable en MATLAB")
    args = p.parse_args()

    npy_to_mat(args.npy, args.mat, args.varname)
