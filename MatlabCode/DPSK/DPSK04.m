% Definir el rango de SNR (en dB)
SNR_dB = 0:5:25;


M = 4;  

% Número de bits por símbolo en DPSK
k = log2(M);  

% Número de subportadoras en OFDM
FFTsize = 64;  

% === OPCIÓN 1: Canal AWGN (OFDM con DPSK se comporta como DPSK convencional)
Eb_N0_dB = SNR_dB - 10*log10(k);  % NO dividimos por N

% === OPCIÓN 2: Canal dispersivo (Multipath, Fading) ===
% Eb_N0_dB = SNR_dB - 10*log10(k * FFTsize);  % Se divide por N si hay dispersión en frecuencia

% Calcular el BER teórico utilizando la función berawgn para DPSK
BER_theoretical = berawgn(Eb_N0_dB, 'dpsk', M);

% Crear el gráfico
figure;
semilogy(SNR_dB, BER_theoretical, 'o-', 'LineWidth', 2);  % Gráfico semilogarítmico
grid on;
xlabel('SNR (dB)');
ylabel('BER Teórico');
title('Curva BER vs SNR para DPSK en OFDM');

% Agregar leyenda para indicar el caso utilizado
legend({'Canal AWGN'}, 'Location', 'SouthWest');

saveas(gcf, 'GoodTheorical.png'); % Save as PNG file

