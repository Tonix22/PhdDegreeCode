from graphviz import Digraph

# Crear el objeto del diagrama de flujo
dot = Digraph("Sequential Diagram for processChannelAndTransmit", format="png")

# Agregar nodos
dot.node("Start", "Start", shape="oval")
dot.node("PSK", "applyPSKModulation(signalTx, M)", shape="rectangle")
dot.node("DPSK_Enc", "applyDPSKEncoding(pskSignal)", shape="rectangle")
dot.node("CheckVarargin", "length(varargin) == 1?", shape="diamond")
dot.node("ProcessChannel", "G = processChannel(varargin{1})", shape="rectangle")
dot.node("ApplyG", "DPSKsignalTx = G * DPSKsignalTx", shape="rectangle")
dot.node("OFDM_Mod", "ofdmModulate(DPSKsignalTx, FFTSize)", shape="rectangle")
dot.node("AWGN", "awgn(OFDMsignalTx, SNR_dB)", shape="rectangle")
dot.node("OFDM_Demod", "ofdmDemodulate(signalRx, FFTSize)", shape="rectangle")
dot.node("DPSK_Dec", "applyDPSKDecoding(OFDMsignalRx, numSC)", shape="rectangle")
dot.node("PSK_Demod", "applyPSKDemodulation(DPSKsignalRx, M)", shape="rectangle")
dot.node("End", "End", shape="oval")

# Agregar conexiones
dot.edge("Start", "PSK")
dot.edge("PSK", "DPSK_Enc")
dot.edge("DPSK_Enc", "CheckVarargin")

# Rama si `varargin == 1`
dot.edge("CheckVarargin", "ProcessChannel", label="Yes")
dot.edge("ProcessChannel", "ApplyG")
dot.edge("ApplyG", "OFDM_Mod")

# Rama si `varargin != 1`
dot.edge("CheckVarargin", "OFDM_Mod", label="No")

# Continuación del flujo
dot.edge("OFDM_Mod", "AWGN")
dot.edge("AWGN", "OFDM_Demod")
dot.edge("OFDM_Demod", "DPSK_Dec")
dot.edge("DPSK_Dec", "PSK_Demod")
dot.edge("PSK_Demod", "End")

# Guardar y renderizar el diagrama
dot.render("processChannelAndTransmit_diagram", format="png", cleanup=False)
