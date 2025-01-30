addpath('Libraries');
imagePath = '/home/tonix/Documents/PhdDegreeCode/Data/Picture/ILSVRC_train/*.JPEG';
imds = imageDatastore(imagePath);
auds = augmentedImageDatastore([256 256], imds, 'ColorPreprocessing', 'gray2rgb');

while hasdata(auds)
    % Read the next mini-batch of augmented images
    dataBatch = read(auds); % Returns a table with images as cells in the first column
    
    % Process each image in the table
    for i = 1:height(dataBatch) % Loop over the rows of the table
        augmentedImage = dataBatch{i, 1}; % Extract the image (cell content)
        augmentedImage = augmentedImage{1}; % Access the actual image data
        symbols = chunkAndEncodeImage(4, 48, augmentedImage);
        processAndRebuildPicture(symbols, 4, 48, 20, augmentedImage);
        imwrite(augmentedImage, 'original_image.jpg'); % Display the image
        %title(['Image ', num2str(i)]);
        break; % Pause to visualize
    end
    break;
end
