
"""
Big-O style benchmark (aprox. por inspección visual)
- Curvas generadas con funciones simples para aproximar el gráfico de referencia.
- Se elimina PhaseNet y se agrega una aproximación para MobileNetV3.

Notas (ajustables):
  * OSIC        ≈ a*N + b           con a=3.6, b=5
  * NearML      ≈ c*N**p            con c=30,  p=0.92   (sublineal, ajustado a la figura)
  * LMMSE       ≈ d*N**3            con d=4.5
  * MobileNetV3 ≈ e*N**2            con e=3.0  (orden cuadrático típico por convoluciones depthwise+pointwise)

Si tienes medidas reales, reemplaza estas fórmulas/constantes o usa tus propios arrays.
"""
import numpy as np
import matplotlib.pyplot as plt

# ----- Rango de N -----
N = np.arange(1, 61)  # 1..60

# ----- Modelos (ajustables) -----
a, b = 3.6, 5.0
OSIC_ops = a * N + b

c, p = 30.0, 0.92
NearML_ops = c * (N ** p)

d = 4.5
LMMSE_ops = d * (N ** 3)

e = 3.0
MNV3_ops = e * (N ** 2)   # MobileNetV3 ~ O(N^2)

# ----- Plot -----
plt.figure(figsize=(10, 6))
plt.semilogy(N, OSIC_ops, marker='o', linestyle='-', label='OSIC')
plt.semilogy(N, NearML_ops, marker='v', linestyle='--', dashes=(4,2), label='NearML')
plt.semilogy(N, LMMSE_ops, marker='s', linestyle='-.', label='LMMSE')
plt.semilogy(N, MNV3_ops, marker='D', linestyle=':', label='MobileNetV3 (≈ O(N^2))')

# Línea vertical de referencia (N ≈ 48)
plt.axvline(48, linestyle='--')

plt.title('Big O Benchmark')
plt.xlabel('N')
plt.ylabel('Operations (log scale)')
plt.grid(True, which='both', linestyle=':', linewidth=0.8)
plt.legend()
plt.tight_layout()
plt.show()

# ----- (Opcional) tabla pequeña de muestra para revisar magnitudes -----
sample = [5, 10, 20, 40, 60]
def pick(arr): 
    return [float(arr[i-1]) for i in sample]  # índices base-1 -> base-0
print("\nMuestras (aprox.)")
print("N       :", sample)
print("OSIC    :", pick(OSIC_ops))
print("NearML  :", pick(NearML_ops))
print("LMMSE   :", pick(LMMSE_ops))
print("MNV3    :", pick(MNV3_ops))
