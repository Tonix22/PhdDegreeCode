import numpy as np
import matplotlib.pyplot as plt
import pywt

# Parámetros
num_samples = 64  # Número de subportadoras
wavelet = 'db4'   # Familia de wavelet Daubechies-4
snr_db = 10       # Relación señal a ruido en dB

# 1. Generar datos aleatorios de 0 a 3 (simulando una señal de transmisión)
data = np.random.randint(0, 4, num_samples)

# 2. Aplicar Transformada Wavelet Discreta (DWT)
coeffs = pywt.wavedec(data, wavelet, level=3)

# 3. Convertir los coeficientes en un solo array
coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

# 4. Agregar ruido AWGN
snr_linear = 10 ** (snr_db / 10)  # Convertir SNR de dB a lineal
power_signal = np.mean(coeff_arr**2)  # Potencia de la señal
power_noise = power_signal / snr_linear  # Potencia del ruido
noise = np.random.normal(0, np.sqrt(power_noise), coeff_arr.shape)
noisy_coeff_arr = coeff_arr + noise

# 5. Reconstruir la señal usando la Transformada Wavelet Inversa (IDWT)
noisy_coeffs = pywt.array_to_coeffs(noisy_coeff_arr, coeff_slices, output_format='wavedec')
reconstructed_data = pywt.waverec(noisy_coeffs, wavelet)

# 6. Graficar la señal transmitida y recibida
plt.figure(figsize=(10, 5))

# Señal transmitida
plt.subplot(2, 1, 1)
plt.stem(data, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.title("Señal Transmitida (Datos Originales)")
plt.xlabel("Muestras")
plt.ylabel("Valor")

# Señal recibida con ruido
plt.subplot(2, 1, 2)
plt.stem(reconstructed_data[:num_samples], linefmt='g-', markerfmt='go', basefmt='r-')
plt.title("Señal Recibida después de AWGN y Reconstrucción")
plt.xlabel("Muestras")
plt.ylabel("Valor")

plt.tight_layout()
plt.show()
