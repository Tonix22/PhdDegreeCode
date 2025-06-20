function [errFrame, demodulatedFrame, channelIndex] = decodeAndCountErrors( ...
        rxFrame, txBitsFrame, modulationOrder, bitsPerSymbol, channelIndex)
%  - rxFrame          : [fftSize × fftSize]  – símbolos recibidos (ya FFT)
%  - txBitsFrame      : [fftSize × bitsPerOFDM] – bits transmitidos
%  - channelIndex     : snapshot actual (se actualiza aquí adentro)
%  ◁ errFrame         : errores de bit acumulados en el frame
%  ◁ demodulatedFrame : [fftSize × fftSize] – símbolos DPSK demodulados
%  ◁ channelIndex     : snapshot incrementado

    fftSize            = size(rxFrame,1);
    errFrame           = 0;
    demodulatedFrame   = zeros(fftSize, fftSize);   % ← almacena todo el frame

    for sc = 1:fftSize
        demodulated = dpskdemod(rxFrame(sc,:), modulationOrder, pi/4);
        demodulatedFrame(sc,:) = demodulated;      % guarda la fila completa

        rxBits  = int2bit(demodulated', bitsPerSymbol);
        errFrame = errFrame + biterr(txBitsFrame(sc,:)', rxBits);

        channelIndex = mod(channelIndex, 9999) + 1; % avanza snapshot
    end
end
