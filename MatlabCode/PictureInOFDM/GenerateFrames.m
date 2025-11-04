clear; clc;

% --- Parámetros del prefijo ---
prefix = "Seq05VD_";    % <-- cambia aquí tu prefijo

% --- Modo de prueba / subconjunto ---
testMode  = false;        % true = usar solo subconjunto, false = todas
subsetIdx = 1:5;         % índices de imágenes a procesar cuando testMode=true

% --- Carga de canales ---
loadmatlab;
addpath('.');

%% Parámetros OFDM / QPSK
modorder   = 4;                    % QPSK
bitsPerSym = log2(modorder);
numTones   = size(NLOS,1);
payloadBitsPerFrame = numTones * bitsPerSym;

% --- Carpetas ---
inDir   = '/home/tonix/Documents/PhdDegreeCode/MatlabCode/PictureInOFDM/701_StillsRaw_full';
outRoot = fullfile(inDir, 'StillRawOFDM');
if ~exist(outRoot, 'dir'), mkdir(outRoot); end

% --- SNRs ---
SNRVECT_IMG = [15 5 0];

% --- Tamaño de imagen ---
targetSize  = [720 960];   % [alto ancho]

% --- Buscar imágenes que empiecen con el prefijo ---
exts = {'*.png','*.jpg','*.jpeg','*.bmp','*.tif','*.tiff'};
files = [];
for e = 1:numel(exts)
    f = dir(fullfile(inDir, exts{e}));
    f = f(startsWith({f.name}, prefix, 'IgnoreCase', true));
    files = [files; f]; %#ok<AGROW>
end

if isempty(files)
    error('No se encontraron imágenes que empiecen con "%s" en %s', prefix, inDir);
end

fprintf('Imágenes encontradas con prefijo "%s": %d\n', prefix, numel(files));

% --- Aplicar subconjunto si está en modo test ---
if testMode
    subsetIdx = subsetIdx(subsetIdx >= 1 & subsetIdx <= numel(files));
    files = files(subsetIdx);
    fprintf('Modo TEST activado. Procesando solo %d imágenes (índices %s)\n', ...
        numel(files), mat2str(subsetIdx));
else
    fprintf('Procesando todas las %d imágenes disponibles.\n', numel(files));
end

% --- Cargar todas las imágenes seleccionadas en RAM ---
numImgs = numel(files);
I_all = zeros([targetSize 3 numImgs], 'uint8');

for k = 1:numImgs
    fname = fullfile(inDir, files(k).name);
    I = imread(fname);
    I = imresize(I, targetSize);
    if size(I,3) ~= 3
        I = repmat(I,1,1,3);
    end
    I_all(:,:,:,k) = uint8(I);
end

fprintf('Imágenes cargadas y vectorizadas en RAM: %.2f GB\n', numel(I_all)/1e9);

% --- Procesamiento OFDM por imagen ---
for s = 1:numel(SNRVECT_IMG)
    SNR = SNRVECT_IMG(s);
    outDir = fullfile(outRoot, sprintf('SNR_%ddB', SNR));
    if ~exist(outDir,'dir'), mkdir(outDir); end
    fprintf('\n=== Procesando SNR = %d dB ===\n', SNR);

    for k = 1:numImgs
        % Canal distinto por imagen
        chanIdx = 1 + mod(k-1, size(NLOS,3));
        H = NLOS(:,:,chanIdx);

        % Imagen actual
        Iu8 = I_all(:,:,:,k);
        imgBytes = Iu8(:);
        imgBits  = reshape(de2bi(imgBytes,8,'left-msb').',[],1);
        padBits  = mod(-numel(imgBits), payloadBitsPerFrame);
        bitsTx   = [imgBits; zeros(padBits,1,'uint8')];

        % Vectorizado por frames
        X = qammod(bitsTx, modorder, 'gray','InputType','bit','UnitAveragePower',true);
        X = reshape(X, numTones, []);
        Y = H * X;
        Ps = mean(abs(Y(:)).^2);
        Pn = Ps / (10^(SNR/10));
        n  = sqrt(Pn/2) * (randn(size(Y)) + 1j*randn(size(Y)));
        y_n = H'*(Y+n);

        rxBits = qamdemod(y_n(:), modorder, 'gray','OutputType','bit','UnitAveragePower',true);
        rxBits = rxBits(1:numel(imgBits));
        rxBytes = uint8(bi2de(reshape(rxBits,8,[]).','left-msb'));
        I_rx = reshape(rxBytes, size(Iu8));

        % Guardar
        fout = fullfile(outDir, files(k).name);
        imwrite(I_rx, fout);
    end
end

fprintf('\n✅ Listo. Imágenes simuladas en: %s\n', outRoot);
