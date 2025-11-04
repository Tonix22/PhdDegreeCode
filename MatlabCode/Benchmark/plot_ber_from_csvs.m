clear; clc;

% === Rutas donde buscar ===
roots = {
    pwd, ...
    fullfile(pwd, 'csv_results'), ...
    '/home/tonix/Documents/PhdDegreeCode/MatlabCode/ZeroFocingMobileNet/Results/BLER', ...
    '/home/tonix/Documents/PhdDegreeCode/MatlabCode/GoldenModeling/csv_BLER'
};

% === Mostrar rutas a escanear ===
fprintf('📁 Escaneando rutas:\n');
for r = 1:numel(roots)
    fprintf('  - %s %s\n', roots{r}, ternary(isfolder(roots{r}), '', '(NO existe)'));
end

% === Recolectar .csv de forma recursiva ===
fileList = {};
for r = 1:numel(roots)
    if isfolder(roots{r})
        D = dir(fullfile(roots{r}, '**', '*.csv'));  % R2016b+
        for k = 1:numel(D)
            fileList{end+1} = fullfile(D(k).folder, D(k).name); %#ok<AGROW>
        end
    end
end

% === Log de archivos encontrados ===
if isempty(fileList)
    error('No se encontraron archivos .csv en las rutas indicadas.');
else
    fprintf('✅ Se encontraron %d archivos CSV.\n', numel(fileList));
    nshow = min(10, numel(fileList));
    for i = 1:nshow
        fprintf('   [%2d] %s\n', i, fileList{i});
    end
    if numel(fileList) > nshow
        fprintf('   ... y %d más.\n', numel(fileList)-nshow);
    end
end

% === Paletas de "texturas" (linea + marcador) ===
markers    = {'o','s','d','^','v','>','<','p','h','x','+'};
linestyles = {'-','--',':','-.'};

% ========== FIGURA ÚNICA: BER/BLER vs SNR (Y log) ==========
figure1 = figure('Color','w'); hold on; grid on; box on;
ax1 = gca; set(ax1, 'YScale', 'log');
labels   = {};
numPlotted = 0;

for f = 1:numel(fileList)
    fp = fileList{f};
    [~, base, ~] = fileparts(fp);
    isBLERfile = strncmpi(base, 'BLER_', 5);  % ¿nombre empieza con BLER_?

    % Intentar leer CSV
    try
        T = readtable(fp, 'Delimiter', ',', 'VariableNamingRule','preserve');
    catch
        warning('No se pudo leer: %s', fp);
        continue;
    end

    % Normalizar nombres de columnas
    names = string(T.Properties.VariableNames);
    namesLower = lower(strrep(names, '_',''));

    % Detectar SNR
    snrIdx = find(namesLower=="snrdb" | namesLower=="snr" | namesLower=="snrdbm" | contains(namesLower,"snr"), 1);
    if isempty(snrIdx)
        numericCols = varfun(@isnumeric, T, 'OutputFormat','uniform');
        cand = find(numericCols);
        if ~isempty(cand), snrIdx = cand(1); end
    end
    if isempty(snrIdx)
        warning('No SNR column in: %s', fp);
        continue;
    end
    snr = T{:, snrIdx};

    % Elegir métrica: si el nombre empieza con BLER_, priorizamos BLER; si no, BER.
    yIdx = [];
    if isBLERfile
        yIdx = find(namesLower=="bler" | contains(namesLower,"bler"), 1);
        if isempty(yIdx)
            % fallback: si no hubiera BLER, intenta BER
            yIdx = find(namesLower=="ber" | contains(namesLower,"ber") | contains(namesLower,"errorrate") | contains(namesLower,"pe"), 1);
        end
    else
        yIdx = find(namesLower=="ber" | contains(namesLower,"ber") | contains(namesLower,"errorrate") | contains(namesLower,"pe"), 1);
        if isempty(yIdx)
            % fallback: si no hubiera BER, intenta BLER
            yIdx = find(namesLower=="bler" | contains(namesLower,"bler"), 1);
        end
    end

    if isempty(yIdx)
        % Heurística: segunda numérica
        numericCols = varfun(@isnumeric, T, 'OutputFormat','uniform');
        cand = find(numericCols);
        if numel(cand) >= 2
            yIdx = cand(2);
        else
            warning('No BER/BLER column in: %s', fp);
            continue;
        end
    end

    y = T{:, yIdx};

    % Limpiar/ordenar
    mask = ~(isnan(snr) | isnan(y));
    snr = snr(mask); y = y(mask);
    if isempty(snr) || isempty(y), continue; end
    [snr, idx] = sort(snr(:), 'ascend');
    y = y(idx);

    % Validar rango (0<y<1) para log
    if all(y <= 0) || all(y >= 1)
        % Si todo está fuera de (0,1) no es ploteable en log de forma sensata
        continue;
    end

    % Textura única por curva
    mkr = markers{mod(numPlotted, numel(markers)) + 1};
    lst = linestyles{mod(floor(numPlotted/numel(markers)), numel(linestyles)) + 1};
    mkIdx = unique(round(linspace(1, numel(snr), min(10, numel(snr)))));

    % Graficar como semilogy (BER o BLER, el que haya tocado)
    semilogy(snr, y, 'LineWidth', 1.8, ...
        'LineStyle', lst, 'Marker', mkr, 'MarkerIndices', mkIdx);

    labels{end+1} = base; %#ok<AGROW>
    numPlotted = numPlotted + 1;
end

if numPlotted == 0
    error(['Se encontraron CSV pero ninguno tenía columnas reconocibles de SNR y BER/BLER en (0,1). ' ...
           'Verifica encabezados (p.ej., "SNR_dB,BER" o "SNR_dB,BLER").']);
end

xlabel('SNR (dB)');
ylabel('BER / BLER');
title('BER / BLER vs SNR (log scale)');
legend(labels, 'Location','northeast', 'Interpreter','none');

% Guardar JPG
outDir = 'plots';
if ~exist(outDir, 'dir'), mkdir(outDir); end
t = datetime('now','Format','MMM_dd_yyyy-HH_mm_ss');
jpg = fullfile(outDir, "BER_BLER_from_CSVs_" + string(t) + ".jpg");
saveas(figure1, jpg);
fprintf('📸 Gráfico guardado en: %s\n', jpg);

% ========== helper inline ==========
function out = ternary(cond, a, b), if cond, out = a; else, out = b; end, end
