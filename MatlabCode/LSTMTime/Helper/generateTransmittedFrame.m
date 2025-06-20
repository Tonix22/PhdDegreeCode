function [bitsFrame, symbolFrame, modulatedFrame] = ...
    generateTransmittedFrame(fftSize, bitsPerOFDMSymbol, bitsPerSymbol, modulationOrder)

    bitsFrame       = randi([0 1], fftSize, bitsPerOFDMSymbol);       % 48 × 96
    numSymbols      = bitsPerOFDMSymbol / bitsPerSymbol;              % = 48
    symbolFrame     = zeros(fftSize, numSymbols);                     % 48 × 48
    modulatedFrame  = zeros(fftSize, numSymbols);                     % 48 × 48

    for idx = 1:fftSize
        bits              = bitsFrame(idx,:)';                        % columna de 96 bits
        symbols           = bit2int(bits, bitsPerSymbol);            % 48 × 1
        symbolFrame(idx,:) = symbols';                               % sin transponer el resultado final
        modulatedFrame(idx,:) = dpskmod(symbols, modulationOrder, pi/4);  % DPSK modulación
    end
end
