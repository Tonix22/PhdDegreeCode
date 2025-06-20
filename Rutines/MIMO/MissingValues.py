import numpy as np
import matplotlib.pyplot as plt

# Datos (sin ceros)
x_data = np.array([3.01, 5.01, 7.01, 9.01, 11.01])
y_data = np.array([0.115041208791209, 0.0476461655277145, 0.00676643954142332,
                   0.00201385532463348, 1.87999398401925e-05])

# Convertimos y a log(y)
log_y = np.log(y_data)

# Ajuste lineal: log(y) = m * x + c
coef = np.polyfit(x_data, log_y, 1)
m, c = coef

# Modelo exponencial: y = exp(m * x + c)
def model(x): return np.exp(m * x + c)

# Nuevos x
x_all = np.array([3.01, 5.01, 7.01, 9.01, 11.01, 13.01, 15.01])
y_pred = model(x_all)

# Mostrar predicciones
print(f"Predicción para x=13.01: {y_pred[5]}")
print(f"Predicción para x=15.01: {y_pred[6]}")

# Graficar
plt.scatter(x_data, y_data, label="Datos conocidos")
plt.plot(x_all, y_pred, label="Ajuste exponencial (log)", color="orange")
plt.scatter([13.01, 15.01], y_pred[5:], color="red", label="Predicciones")
plt.yscale('log')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.title("Regresión Exponencial (vía log)")
plt.savefig("Exponential_Regression.png", dpi=300, bbox_inches='tight')

