net = load("BinaryNet.mat").net;
% Datos de entrenamiento
XTrain = dlarray(randn(48, 1000), 'CB'); % 1000 muestras (48 features)
YTrain = dlarray(randi([0,1], 48, 1000), 'CB'); % Salidas binarias
% 🔹 **Generar datos de prueba**
XTest = dlarray(randn(48, 10), 'CB'); % 10 muestras de prueba
YTest = dlarray(randi([0,1], 48, 10), 'CB'); % 10 muestras de salida esperada

% Hiperparámetros
numEpochs = 10;
miniBatchSize = 16;
learningRate = 0.001;

net = trainNetworkCustom(net, XTrain, YTrain, numEpochs, miniBatchSize, learningRate, @modelLoss);

% **Probar la red y obtener métricas**
metrics = testNetwork(XTest, YTest, net);

% Acceder a métricas individualmente
fprintf("F1 Score obtenido: %.4f\n", metrics.F1Score);
