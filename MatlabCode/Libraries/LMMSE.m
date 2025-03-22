function [yp] = LMMSE(H, NoiseVar, rxSig, fftSize)
    yp = inv((H'*H + NoiseVar*eye(fftSize))) * H'*rxSig;