function rebuiltImage = decodeAndRebuildImage(symbols, M, numSC, imagePath)
    % symbols: Encoded symbols
    % M: Number of levels for decoding (related to log2(M) bits)
    % numSC: Number of blocks of log2(M) bits
    % imagePath: Path to the image (to retrieve original dimensions)

    % Read the image to get the original size
    img = imread(imagePath);
    imageSize = size(img);

    % Calculate the number of bits per symbol
    bitsPerSymbol = log2(M);

    % Convert symbols back to binary
    reshapedSymbols = reshape(symbols.', [], 1); % Reshape back to original order
    binaryData = dec2bin(reshapedSymbols, bitsPerSymbol)'; % Transpose for easier manipulation
    binaryData = binaryData(:)'; % Convert to single binary row vector

    % Remove padding bits (based on the original image size)
    totalBits = prod(imageSize) * 8; % Total bits required for the image
    binaryData = binaryData(1:totalBits); % Trim extra padding

    % Reshape binary data into 8-bit groups for pixels
    pixelBinary = reshape(binaryData, 8, []); % Each column is one pixel
    pixelValues = uint8(bin2dec(pixelBinary')); % Convert binary to uint8

    % Reshape to the original image dimensions
    rebuiltImage = reshape(pixelValues, imageSize);
end