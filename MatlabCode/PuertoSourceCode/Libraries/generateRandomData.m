function signalTx = generateRandomData(M, numSC)
    signalTx = randi([0 M-1], numSC, 1);
end
