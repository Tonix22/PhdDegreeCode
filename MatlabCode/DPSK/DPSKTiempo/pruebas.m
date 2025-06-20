close all
clear all
clc



% Parámetros del sistema
N = 48; % Número de subportadoras
realizaciones = 512; % Número de realizaciones del canal (puedes cambiarlo)

% % Inicializar el canal Rayleigh
% rayleighChannel = comm.RayleighChannel(...
%     'SampleRate', 1e6, ...       % Frecuencia de muestreo (ajustar según el sistema)
%     'PathDelays', [0 1e-6], ...  % Retrasos de las trayectorias (en segundos)
%     'AveragePathGains', [0 -3], ... % Ganancias promedio de las trayectorias
%     'MaximumDopplerShift', 100, ... % Desplazamiento Doppler máximo
%     'NormalizePathGains', true); ... % Normalizar las ganancias de las trayectorias
%    % 'NumTransmitAntennas', 1, ... % Número de antenas de transmisión
%    % 'NumReceiveAntennas', 1);    % Número de antenas de recepción
% 
% % Simulación del canal Rayleigh y creación de la matriz de canal
%matrizDeCanal = zeros(realizaciones, N); % Matriz donde guardaremos las diagonales

% % Generar realizaciones del canal Rayleigh y extraer las diagonales
% for i = 1:realizaciones
%     % Generar una realización del canal (debe ajustarse con el tamaño adecuado)
%     canalRealizado = rayleighChannel(ones(N,1)); % Canal para N muestras
%     matrizDeCanal(i, :) = canalRealizado.'; % Guardamos la realización en la matriz (transpuesta)
% end


load('../../../Data/kaggle_dataset/v2v80211p_LOS.mat')

 H1 = vectReal32b;

 load('../../../Data/kaggle_dataset/v2v80211p_NLOS.mat')

 H2 = vectReal32b;


 
 
 
 channelCont=1;


 for i = 1:N
 G1 = H1(:,:,channelCont);
 G2 = H2(:,:,channelCont);
 matrizDeCanal (i,:) = diag(G1 + G2);
 channelCont = channelCont +1;
 end



% F = mydftMat(N,1);
% Finv =mydftMat(N,0);
% 
% G2 = F*G*Finv;


% Visualización de la matriz con las realizaciones del canal
figure
imagesc(mag2db(abs(matrizDeCanal))); % Usamos abs() para ver la magnitud del canal
colorbar;
xlabel('Subportadoras');
ylabel('Realizaciones');




figure
surf(mag2db(abs(matrizDeCanal))); % Usamos abs() para ver la magnitud del canal
colorbar;
xlabel('Subportadoras');
ylabel('Realizaciones');
%title('Variabilidad del Canal Rayleigh (Matriz de Realizaciones)');

figure
imagesc(mag2db(imag(matrizDeCanal)));
colorbar
xlabel('Subportadoras');
ylabel('Realizaciones');
title('Imaginario')


figure
imagesc(mag2db(real(matrizDeCanal)));
colorbar
xlabel('Subportadoras');
ylabel('Realizaciones');
title('Real')
