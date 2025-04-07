filepath = "LSTMResultsAwgn/LSTM_Network_SNR_1.101030e+01dB.mat";
network = load(filepath).net;
hPC = optimizeConfigurationForNetwork(dlhdl.ProcessorConfig, network);
Estimated = hPC.estimateResources('IncludeReferenceDesign',false)