function metrics = testNetwork(XTest, YTest, net)
    % 🔥 **Predicción**
    YPred = forward(net, XTest);
    YPredBin = extractdata(YPred); % Convertir de dlarray a matriz numérica

    % 🔹 **Datos de prueba (ground truth)**
    YTestBin = extractdata(YTest); % Convertir a matriz numérica

    % 📊 **Evaluar rendimiento y obtener métricas**
    metrics = evaluateModel(YTestBin, YPredBin);
end