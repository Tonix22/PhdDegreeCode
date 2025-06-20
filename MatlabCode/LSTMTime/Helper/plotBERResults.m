function plotBERResults(EbNo_dB, bitErrorRate, modulationOrder)
    figure;
    berTheoretical = berawgn(EbNo_dB, 'dpsk', modulationOrder);
    berDPSKFD = [0.3327 0.2815 0.2278 0.1939 0.1594 0.1314 0.1122 0.0934 0.0908];
    berLMMSEFD = [0.2846 0.1919 0.1157 0.0511 0.0161 0.0028 0.0005 0.0003 0.0002];

    semilogy(EbNo_dB, berLMMSEFD, 'kd-', 'LineWidth', 1.5);
    hold on;
    semilogy(EbNo_dB, berDPSKFD, 'k*-', 'LineWidth', 1.5);
    semilogy(EbNo_dB, bitErrorRate, 'bo--', 'LineWidth', 1.5);
    xlabel('Eb/No (dB)');
    ylabel('BER');
    legend('DPSK-OFDM coherent FD, LMMSE', 'DPSK-OFDM non-coherent FD', 'DPSK-OFDM non-coherent TD');
    grid on;
    title('DPSK-OFDM non-coherent');
end