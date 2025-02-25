function net = trainNetworkCustom(net, XTrain, YTrain, numEpochs, miniBatchSize, learningRate, modelLoss)
    % Inicializar variables para Adam optimizer
    trailingAvg = [];
    trailingAvgSq = [];
    gradientDecayFactor = 0.9;
    squaredGradientDecayFactor = 0.999;

    % 🔥 **Loop de entrenamiento con BCE**
    for epoch = 1:numEpochs
        for i = 1:miniBatchSize:size(XTrain, 2)
            idx = i:min(i+miniBatchSize-1, size(XTrain, 2));
            XBatch = XTrain(:, idx);
            YBatch = YTrain(:, idx);

            % 🔹 **Calcular pérdida y gradientes usando dlfeval() con la función de pérdida personalizada**
            [loss, gradients] = dlfeval(modelLoss, net, XBatch, YBatch);

            % 🔹 **Actualizar pesos con Adam**
            [net.Learnables, trailingAvg, trailingAvgSq] = ...
                adamupdate(net.Learnables, gradients, ...
                           trailingAvg, trailingAvgSq, ...
                           epoch, learningRate, gradientDecayFactor, ...
                           squaredGradientDecayFactor);
        end
        fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
    end
end
