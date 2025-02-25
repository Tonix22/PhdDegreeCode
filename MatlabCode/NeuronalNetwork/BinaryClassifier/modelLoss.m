function [loss, gradients] = modelLoss(net, X, Y)
    YPred = forward(net, X); % Forward pass
    loss = crossentropy(YPred, Y); % Binary Cross Entropy
    gradients = dlgradient(loss, net.Learnables); % Backpropagation
end