import os
import sys
import unittest
import numpy as np
from pathlib import Path

# Add sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Drivers'))
sys.path.insert(0, parent_dir)

from ConstelationCoder import ConstelationCoder, Modulation # type: ignore

class CostelationCodingTest(unittest.TestCase):
    
    def test_ConstelationCoding(self):
        constelation_sizes = (4, 16, 64, 256)
        for constelation_size in constelation_sizes:
            for modulation in Modulation:
                coder = ConstelationCoder(modulation, constelation_size)
                tx_data = np.array([0, 1, 1, 0])  # Example binary data
                encoded_signal = coder.Encode(tx_data)
                decoded_signal = coder.Decode(encoded_signal)
                self.assertEqual(tx_data.all(),decoded_signal.all())
                self.assertEqual(tx_data.any(),decoded_signal.any())
        
if __name__ == '__main__':
    unittest.main()