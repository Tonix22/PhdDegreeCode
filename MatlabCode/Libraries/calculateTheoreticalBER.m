function berTheoretical = calculateTheoreticalBER(SNR_dB, M)
    k = log2(M); 
    EbNo_dB = SNR_dB - 10*log10(k);
    berTheoretical = berawgn(EbNo_dB, 'dpsk', M, 'nondiff'); % Theoretical BER
end
