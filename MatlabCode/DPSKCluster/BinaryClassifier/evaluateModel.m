function metrics = evaluateModel(YTest, YPred)
    % Convertir a formato binario (0 o 1)
    YPredBin = YPred > mean(YPred);

    % Calcular Verdaderos Positivos (TP), Verdaderos Negativos (TN), 
    % Falsos Positivos (FP) y Falsos Negativos (FN)
    TP = sum((YPredBin == 1) & (YTest == 1), 'all');
    TN = sum((YPredBin == 0) & (YTest == 0), 'all');
    FP = sum((YPredBin == 1) & (YTest == 0), 'all');
    FN = sum((YPredBin == 0) & (YTest == 1), 'all');

    % Calcular métricas
    accuracy = (TP + TN) / (TP + TN + FP + FN);
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1_score = 2 * (precision * recall) / (precision + recall);

    % Manejo de valores NaN en precision y recall (cuando TP + FP o TP + FN es 0)
    if isnan(precision)
        precision = 0;
    end
    if isnan(recall)
        recall = 0;
    end
    if isnan(f1_score)
        f1_score = 0;
    end

    % Guardar métricas en una estructura
    metrics = struct(...
        'Accuracy', accuracy, ...
        'FalsePositives', FP, ...
        'FalseNegatives', FN, ...
        'Recall', recall, ...
        'Precision', precision, ...
        'F1Score', f1_score);

    % También mostrar en consola para revisión rápida
    fprintf('Accuracy: %.4f\n', metrics.Accuracy);
    fprintf('False Positives (FP): %d\n', metrics.FalsePositives);
    fprintf('False Negatives (FN): %d\n', metrics.FalseNegatives);
    fprintf('Recall: %.4f\n', metrics.Recall);
    fprintf('Precision: %.4f\n', metrics.Precision);
    fprintf('F1 Score: %.4f\n', metrics.F1Score);
end
