function plotBER(SNR_dB, ber, M, numBitSymbol, plotName, berTheoretical)
    % Create the plot
    figure;
    
    % Plot theoretical BER if provided
    if nargin >= 6 && ~isempty(berTheoretical)
        semilogy(SNR_dB, berTheoretical, 'k-', 'LineWidth', 1.5); % Theoretical BER
        hold on;
    end
    
    % Plot estimated BER
    semilogy(SNR_dB, ber, 'b--', 'LineWidth', 1.5); % Estimated BER
    xlabel('SNR (dB)');
    ylabel('BER');
    legend('Theoretical BER', 'Estimated BER');
    grid on;
    
    % Save the plot as a file in the current directory
    if nargin < 5 || isempty(plotName)
        plotName = 'BER_plot_SNR'; % Default name if not provided
    end
    saveas(gcf, [plotName, '.png']); % Save as PNG file

    % Close the figure to prevent unnecessary display
    close(gcf);
end
