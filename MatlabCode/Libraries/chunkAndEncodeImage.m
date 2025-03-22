function symbols = chunkAndEncodeImage(M, numSC, image)
    % M: Number of levels for encoding (related to log2(M) bits)
    % numSC: Number of blocks of log2(M) bits
    % imagePath: Path to the image

    % Read the image
    if ischar(image) || isstring(image)
        img = imread(image);
    else
        img = image;
    end
    % Ensure the image is in uint8 format
    if ~isa(img, 'uint8')
        error('The image must be in uint8 format.');
    end

    % Flatten the image into a 1D array (preserving RGB channels if any)
    imgData = img(:); % 1D array of all pixel values

    % Calculate the number of bits required for M
    bitsPerSymbol = log2(M);
    if mod(bitsPerSymbol, 1) ~= 0
        error('M must be a power of 2.');
    end

    % Convert the image data to binary
    imgBinary = dec2bin(imgData, 8); % Each pixel as an 8-bit binary string
    imgBinary = imgBinary'; % Transpose for easier manipulation
    imgBinary = imgBinary(:)'; % Convert to a single binary row vector

    % Pad binary data to ensure it is divisible by bitsPerSymbol * numSC
    totalBits = bitsPerSymbol * numSC;
    paddingBits = mod(-numel(imgBinary), totalBits);
    imgBinary = [imgBinary, repmat('0', 1, paddingBits)];

    % Split the binary data into chunks of bitsPerSymbol
    reshapedBinary = reshape(imgBinary, bitsPerSymbol, []); % Each column is one symbol
    symbols = bin2dec(reshapedBinary'); % Convert binary to decimal symbols
    symbols = reshape(symbols, numSC, []).';

    % Output the symbols and number of padding bits
    %fprintf('Image chunked and encoded into %d symbols.\n', size(symbols));
end
