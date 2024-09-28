import commpy.channels as chan
import numpy as np
import os

#DataPath = os.getcwd() + "../../Data/kaggle_dataset/"
DataPath = "/home/tonix/Documents/PhdDegreeCode/Data/kaggle_dataset/"

class Channel():
    def __init__(self, snr, LOS = True):
        self.channelLOS = LOS
        if(self.channelLOS):
            # Load the .npy file into a NumPy array
            self.channel = np.load(DataPath + 'v2v80211p_LOS.npy')
        else:
            self.channel = np.load(DataPath + 'v2v80211p_NLOS.npy')
            
        self.snr = snr
        self.index = 0
        
    def response(self,tx):
        # Generate the noisy signal with AWGN
        self.noisy_signal = tx@self.channel[:, :, self.index] + chan.awgn(tx, self.snr)
        
        self.index +=1
        if(self.index == 10000):
            self.index = 0
            
        return self.noisy_signal
