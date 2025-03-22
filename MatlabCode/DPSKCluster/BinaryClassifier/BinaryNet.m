% Definir la arquitectura de la red neuronal
layers = [
    featureInputLayer(48, 'Name', 'input') % 48 entradas
    fullyConnectedLayer(144, 'Name', 'fc1')
    batchNormalizationLayer()
    reluLayer('Name', 'relu1')

    fullyConnectedLayer(144, 'Name', 'fc2')
    batchNormalizationLayer()
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(48, 'Name', 'fc3')
    batchNormalizationLayer()
    sigmoidLayer('Name', 'sigmoid') % Activación sigmoide para clasificación binaria
];

% Convertir arquitectura en dlnetwork
net = dlnetwork(layers);

save('BinaryNet.mat',"net")
