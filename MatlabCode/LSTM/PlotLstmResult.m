% filepath: /home/tonix/Documents/PhdDegreeCode/MatlabCode/LSTM/PlotLstmResult.m
M = 4;                   % Modulation order (QPSK)
% Read the CSV file
csvFileName = 'LSTMResults/LSTM_DPSK_Network.csv';
data = readmatrix(csvFileName);

% Extract SNR, BER (Raw), and BER (NN)
snrValues = data(:, 1);       % SNR values
berRaw = data(:, 2);          % BER for DPSK-only
berNN = data(:, 3);           % BER for NN+DPSK

berTheoretical = calculateTheoreticalBER(snrValues, M); % Optional calculation of theoretical BER
% Plot the results
figure;
semilogy(snrValues, berRaw, '-o', 'LineWidth', 1.5, 'DisplayName', 'DPSK-only','Color', 'b');
hold on;
semilogy(snrValues, berNN, '-s', 'LineWidth', 1.5, 'DisplayName', 'NN+DPSK','Color', 'r');
semilogy(snrValues, berTheoretical, '--', 'LineWidth', 1.5, 'DisplayName', 'Theoretical BER','Color', 'g');

hold off;

% Add labels, legend, and title
xlabel('SNR (dB)');
ylabel('BER');
legend('Location', 'southwest');
grid on;
title('NN+DPSK vs DPSK-only');

% Save the plot as a .png file
outputFileName = 'NN_DPSK_vs_DPSK_only.png';
saveas(gcf, outputFileName);

fprintf('Plot saved as: %s\n', outputFileName);