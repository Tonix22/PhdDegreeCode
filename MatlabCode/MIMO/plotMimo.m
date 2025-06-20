% filepath: /home/tonix/Documents/PhdDegreeCode/MatlabCode/LSTM/PlotLstmResult.m
M = 4;                   % Modulation order (QPSK)
EbNo = 0:2:12;
channelAWGN = 1;

outputFileName = 'AWGN_DPSK_MIMO_NET.png';

csvFileName = 'DPSK_SNR_MIMONET.csv';
% Read the CSV file

data = readmatrix(csvFileName);
% Extract SNR, BER (Raw), and BER (NN)
berNN = data(:, 2);           % BER for NN+DPSK

berTheorical = berawgn(EbNo,'dpsk',M);
% Plot the results
figure;
semilogy(EbNo,berTheorical, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Theoretical BER','Color', 'g');
hold on;
semilogy(EbNo, berNN, '-s', 'LineWidth', 1.5, 'DisplayName', 'DPSK MIMONET','Color', 'b');

hold off;

% Add labels, legend, and title
xlabel('Eb/No (dB)')
ylabel('BER')
legend('Location', 'southwest');
grid on;
if channelAWGN
    title('AWGN NN MIMONET vs Theoretical DPSK');
else
    title('V2V NN+DPSK vs DPSK-only');
end

saveas(gcf, outputFileName);

fprintf('Plot saved as: %s\n', outputFileName);