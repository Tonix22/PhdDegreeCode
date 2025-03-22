
function processAndRebuildPicture(symbols, M, numSC, snr, imagePath)
    % Iterate over rows of symbols and process each row
    [numRows, ~] = size(symbols);
    processedSymbols = zeros(size(symbols));

    for i = 1:numRows
        rowSymbols = symbols(i, :);

        [signalEstimate] = processChannelAndTransmit(rowSymbols, M, numSC, snr, numSC);

        %% Ensure integer symbols after processing
        processedSymbols(i, :) = signalEstimate;
    end

    % Decode the processed symbols and rebuild the image
    rebuiltImage = decodeAndRebuildImage(processedSymbols, M, numSC, imagePath);

    % Display the rebuilt image
    imwrite(rebuiltImage, 'noise_image.jpg');
    title('Rebuilt Image After Processing');
end