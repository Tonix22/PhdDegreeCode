classdef Equalizers
    properties
        name     % Current Method Name
        isLinear % Linear method or not.
        handler  % Current equalizer function
    end
    
    methods
        % Constructor method to initialize properties
        function obj = Equalizers(name, isLinear,handler)
            if nargin > 0  % Check if input arguments are provided
                obj.name = name;
                obj.isLinear = isLinear;
                obj.handler = handler;
            end
        end
        
        % Method to calculate the area
        function obj = PlotResult(obj, SNR, berEst)
            semilogy(SNR, berEst, 'b*-');
            legendName = sprintf('OFDM-%s', obj.name);
            legend(legendName);
            xlabel('SNR (dB)', 'Interpreter', 'latex'); 
            ylabel('BER', 'Interpreter', 'latex');
            title('Binary QAM over V2V Channel');
            grid on;

            set(gca, 'fontsize', 14);  % Set font size
            set(groot, 'defaultAxesTickLabelInterpreter', 'latex'); 
            set(groot, 'defaultLegendInterpreter', 'latex');

            % Get the current date and time
            timestamp = datestr(datetime('now'), 'yyyy-mm-dd_HH-MM-SS');
            % Save the plot with the timestamp in the filename
            pictureFileName = sprintf('./Results/BER_Plot_%s_%s.png', obj.name, timestamp);
            csvFileName = sprintf('./Results/BER_Plot_%s_%s.csv', obj.name, timestamp);

            saveas(gcf, pictureFileName);
            csvwrite(csvFileName, berEst);

        end
    
    end
end
