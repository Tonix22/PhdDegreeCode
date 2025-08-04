function receivedFrame = transmitThroughChannel( ...
        txFrame,           ... % [fftSize × fftSize] símbolos DPSK ya mapeados a subportadoras
        fftSize,           ... % = 48
        SNRdB,             ... % valor de SNR en dB para este punto de simulación
        chanLOS, chanNLOS, ... % cubos (Ns × Ns × Nsnap) con respuesta LOS/NLOS
        channelIndex)          % snapshot actual (se mantiene fuera de la función)

    % Prealocación
    receivedFrame = zeros(fftSize, fftSize);

    % ─── Canal LOS+NLOS único durante TODO el frame ──────────────────────────
    %H = chanLOS(:,:,channelIndex) + chanNLOS(:,:,channelIndex);
    H = chanNLOS(:,:,channelIndex);

    % ─── Procesar cada símbolo OFDM del frame ────────────────────────────────
    for sym = 1:fftSize
        % 1. Modulación OFDM (IFFT)
        txOFDM = ifft(txFrame(:,sym), fftSize);

        % 2. AWGN: calcular varianza a partir de la potencia del símbolo TX
        signalPow_dB = 10*log10(var(txOFDM));
        noiseVar     = 10.^((signalPow_dB - SNRdB)/10);   % misma fórmula que el original
        noise        = sqrt(noiseVar/2) * ...
                    (randn(size(txOFDM)) + 1j*randn(size(txOFDM)));

        % 3. Canal + FFT de recepción
        rxOFDM = fft(txOFDM + noise, fftSize);

        % 4. Aplicar el mismo canal a las 48 subportadoras
        receivedFrame(:,sym) = H * rxOFDM;
    end
end