function [XTrain, XTest, YTrain, YTest] = splitDlData(X, Y)
    % splitDlData Splits dlarray data into training and test sets **without shuffle**.
    %   [XTrain, XTest, YTrain, YTest] = splitDlData(X, Y) splits the input
    %   dlarrays X and Y (with samples along the second dimension) into
    %   80% training and 20% test sets, keeping the original order.
    %
    %   Inputs:
    %       X - dlarray for predictors with size [features, samples]
    %       Y - dlarray for responses with size [features, samples]
    %
    %   Outputs:
    %       XTrain - Training subset of X (first 80%)
    %       XTest  - Test subset of X (last 20%)
    %       YTrain - Training subset of Y (first 80%)
    %       YTest  - Test subset of Y (last 20%)
    
    % Determine the number of samples (assumed to be along dimension 2)
    numSamples = size(Y, 2);

    % Define 80% of the data for training
    numTrain = round(0.8 * numSamples);

    % Split without shuffling (first 80% training, last 20% test)
    XTrain = X(:, 1:numTrain);
    XTest  = X(:, numTrain+1:end);
    YTrain = Y(:, 1:numTrain);
    YTest  = Y(:, numTrain+1:end);
end
