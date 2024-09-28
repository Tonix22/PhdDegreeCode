import os
import sys
import unittest
import numpy as np
from pathlib import Path
from commpy.utilities import hamming_dist

# Agrega el directorio del proyecto a sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Drivers'))
sys.path.insert(0, parent_dir)

from ConstelationCoder import ConstelationCoder, Modulation # type: ignore
from SignalChannelResponse import Channel # type: ignore

class CostelationCodingTest(unittest.TestCase):
    
    def test_ConstelationCoding(self):
        coder = ConstelationCoder(Modulation.QAM, 4)
        tx_bits = np.random.randint(0, 2, 96)
        tx = coder.Encode(tx_bits)
        
        channel = Channel(40)
        rx = channel.response(tx)
        
        rx_bits = coder.Decode(rx)
        
        ber = np.sum(tx_bits != rx_bits)/len(rx_bits)
        
        print(ber)
        
if __name__ == '__main__':
    unittest.main()