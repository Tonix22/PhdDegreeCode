% Definir la arquitectura de la red neuronal
layers = [
    featureInputLayer(48, 'Name', 'input') % 48 entradas
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(48, 'Name', 'fc3')
    sigmoidLayer('Name', 'sigmoid') % Activación sigmoide para clasificación binaria
];

% Convertir arquitectura en dlnetwork
net = dlnetwork(layers);

save('BinaryNet.mat',"net")
