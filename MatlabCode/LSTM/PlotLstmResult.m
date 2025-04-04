% filepath: /home/tonix/Documents/PhdDegreeCode/MatlabCode/LSTM/PlotLstmResult.m
M = 4;                   % Modulation order (QPSK)
EbNo = 0:2:12;
channelAWGN = 1;

if channelAWGN
    csvFileName = 'LSTMResultsAwgn/LSTM_DPSK_Network.csv';
    outputFileName = 'AWGN NN+DPSK Known SNR vs NN Ber Vote Unknown SNR.png';
    csvVote = 'LSTMResultsAwgn/BER_vs_SNR_vote.csv';
else
    csvFileName = 'LSTMResultsV2VChannel/LSTM_DPSK_Network.csv';
    % Save the plot as a .png file
    outputFileName = 'V2V_NN_DPSK_vs_DPSK_only.png';
end
% Read the CSV file

data = readmatrix(csvFileName);
datavote = readmatrix(csvVote);
% Extract SNR, BER (Raw), and BER (NN)
berRaw = data(:, 2);          % BER for DPSK-only
berNN = data(:, 3);           % BER for NN+DPSK
berVote = datavote(:, 2);    % BER for voting

berTheorical = berawgn(EbNo,'dpsk',M);
% Plot the results
figure;
semilogy(EbNo,berTheorical, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Theoretical BER','Color', 'g');
hold on;
semilogy(EbNo, berNN, '-s', 'LineWidth', 1.5, 'DisplayName', 'NN+DPSK Known SNR','Color', 'b');
semilogy(EbNo, berRaw, '-o', 'LineWidth', 1.5, 'DisplayName', 'DPSK-only','Color', 'black');

semilogy(EbNo, berVote, '-+', 'LineWidth', 1.5, 'DisplayName', 'NN Ber Vote Unknown SNR','Color', 'r');

hold off;

% Add labels, legend, and title
xlabel('Eb/No (dB)')
ylabel('BER')
legend('Location', 'southwest');
grid on;
if channelAWGN
    title('AWGN NN+DPSK Known SNR vs NN Ber Vote Unknown SNR');
else
    title('V2V NN+DPSK vs DPSK-only');
end

saveas(gcf, outputFileName);

fprintf('Plot saved as: %s\n', outputFileName);