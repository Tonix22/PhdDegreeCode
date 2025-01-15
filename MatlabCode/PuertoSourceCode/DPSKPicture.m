addpath('Libraries');
imagePath = 'Pictures/Retsuko.jpeg'
symbols = chunkAndEncodeImage(4, 48, imagePath);
processAndRebuildPicture(symbols, 4, 48, 20, imagePath);