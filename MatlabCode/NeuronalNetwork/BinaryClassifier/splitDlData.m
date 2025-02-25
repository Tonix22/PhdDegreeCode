function [XTrain, XTest, YTrain, YTest] = splitDlData(X, Y)
    % splitDlData Splits dlarray data into training and test sets.
    %   [XTrain, XTest, YTrain, YTest] = splitDlData(X, Y) randomly splits the
    %   input dlarrays X and Y (with samples along the second dimension) into
    %   80% training and 20% test sets.
    %
    %   Inputs:
    %       X - dlarray for predictors with size [features, samples]
    %       Y - dlarray for responses with size [features, samples]
    %
    %   Outputs:
    %       XTrain - Training subset of X
    %       XTest  - Test subset of X
    %       YTrain - Training subset of Y
    %       YTest  - Test subset of Y
    
        % Determine the number of samples (assumed to be along dimension 2)
        numSamples = size(Y, 2);
        
        % Random permutation of sample indices
        idx = randperm(numSamples);
        
        % Define 80% of the data for training
        numTrain = round(0.8 * numSamples);
        
        % Split indices into training and test sets
        trainIdx = idx(1:numTrain);
        testIdx = idx(numTrain+1:end);
        
        % Split the dlarrays based on the computed indices
        XTrain = X(:, trainIdx);
        XTest  = X(:, testIdx);
        YTrain = Y(:, trainIdx);
        YTest  = Y(:, testIdx);
end
    