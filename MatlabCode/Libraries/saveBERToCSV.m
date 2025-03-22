function saveBERToCSV(SNR_dB, ber, csvFileName)
    % Combine SNR and BER data into a single matrix
    dataToSave = [SNR_dB(:), ber(:)];
    
    % If no file name is provided, use a default name
    if nargin < 3 || isempty(csvFileName)
        csvFileName = 'ber_data_SNR.csv';
    end
    
    % Save the data to a CSV file
    writematrix(dataToSave, csvFileName);

    % Display a message indicating the data was saved
    fprintf('Data saved to %s\n', csvFileName);
end
