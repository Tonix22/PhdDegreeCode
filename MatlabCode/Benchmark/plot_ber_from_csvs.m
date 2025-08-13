clear; clc;

% === Rutas donde buscar (agregué tu ruta exacta) ===
roots = {
    pwd, ...
    fullfile(pwd, 'csv_results'), ...
    '/home/tonix/Documents/PhdDegreeCode/MatlabCode/ZeroFocingMobileNet', ...
    '/home/tonix/Documents/PhdDegreeCode/MatlabCode/GoldenModeling/csv_results'  % <-- tu ruta
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
        % Busca *.csv en subcarpetas
        D = dir(fullfile(roots{r}, '**', '*.csv'));  % requiere R2016b+
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
    % opcional: listar algunos
    nshow = min(10, numel(fileList));
    for i = 1:nshow
        fprintf('   [%2d] %s\n', i, fileList{i});
    end
    if numel(fileList) > nshow
        fprintf('   ... y %d más.\n', numel(fileList)-nshow);
    end
end

% === Figura y estilo ===
figure1 = figure('Color','w'); hold on; grid on; box on;
set(gca, 'YScale', 'log');   % eje Y logarítmico

labels = {};
numPlotted = 0;

for f = 1:numel(fileList)
    fp = fileList{f};
    % Intentar leer CSV
    try
        T = readtable(fp, 'Delimiter', ',', 'VariableNamingRule','preserve');
    catch
        warning('No se pudo leer: %s', fp);
        continue;
    end

    % Normalizar nombres de columnas (para detectar SNR/BER)
    names = string(T.Properties.VariableNames);
    namesLower = lower(strrep(names, '_',''));
    % candidatos típicos
    snrIdx = find( ...
        namesLower == "snrdb" | namesLower == "snr" | namesLower == "snrdbm" | ...
        contains(namesLower, "snr"), 1);
    berIdx = find( ...
        namesLower == "ber" | contains(namesLower, "ber") | ...
        contains(namesLower, "errorrate") | contains(namesLower, "pe"), 1);

    if isempty(snrIdx) || isempty(berIdx)
        % Si no hay columnas claras, intenta heurística: 1ª numérica como SNR y 2ª numérica como BER
        numericCols = varfun(@isnumeric, T, 'OutputFormat','uniform');
        cand = find(numericCols);
        if numel(cand) >= 2
            snrIdx = cand(1);
            berIdx = cand(2);
        else
            % No ploteable
            continue;
        end
    end

    snr = T{:, snrIdx};
    ber = T{:, berIdx};

    % Limpiar y ordenar
    mask = ~(isnan(snr) | isnan(ber));
    snr = snr(mask); ber = ber(mask);
    if isempty(snr) || isempty(ber), continue; end
    [snr, idx] = sort(snr(:), 'ascend');
    ber = ber(idx);

    % Validar rango
    if all(ber <= 0) || all(ber >= 1)
        % datos sospechosos, lo saltamos
        continue;
    end

    % Graficar
    semilogy(snr, ber, 'LineWidth', 1.8);
    [~, base, ~] = fileparts(fp);
    labels{end+1} = base; %#ok<AGROW>
    numPlotted = numPlotted + 1;
end

if numPlotted == 0
    error(['Se encontraron CSV pero ninguno tenía columnas reconocibles de SNR/BER. ' ...
           'Verifica encabezados (p.ej., "SNR_dB,BER").']);
end

xlabel('SNR (dB)');
ylabel('BER');
title('BER vs SNR');
legend(labels, 'Location','northeast', 'Interpreter','none');

% Guardar JPG
outDir = 'plots';
if ~exist(outDir, 'dir'), mkdir(outDir); end
t = datetime('now','Format','MMM_dd_yyyy-HH_mm_ss');
jpg = fullfile(outDir, "BER_from_CSVs_" + string(t) + ".jpg");
saveas(figure1, jpg);
fprintf('📸 Gráfico guardado en: %s\n', jpg);

% ========== helper inline (sin funciones externas) ==========
function out = ternary(cond, a, b), if cond, out = a; else, out = b; end, end
