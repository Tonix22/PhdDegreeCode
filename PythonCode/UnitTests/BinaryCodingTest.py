import os
import sys
import unittest
import numpy as np
from pathlib import Path

# Agrega el directorio del proyecto a sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Drivers'))
sys.path.insert(0, parent_dir)

# Importa Encoder desde Driver
from DataBinEncoder import EncodeDataIntoBits # type: ignore
from DataBinDecoder import DecodeFramesToBits # type: ignore

class BinaryCodingTest(unittest.TestCase):
    
    def setUp(self):
        self.byte_array = np.array([0xA2, 0x3A,0xFE,0X30,0xA2, 0xAA,0xF0,0XF0], dtype=np.uint8)  # Example byte array
        self.slice = 2 
        self.frame_size = 48
        self.filePath = os.getcwd()
    
    def test_EnconderCreation(self):
        """
        Test codification shape and file to export
        """
        databitEncode = EncodeDataIntoBits(self.byte_array, self.slice, self.frame_size, self.filePath)
        databitEncode.process_byte_array()
        databitEncode.save_frames_to_npy()
        self.assertEqual(databitEncode.frames.shape, (2,48)) 
        self.assertTrue(Path(os.getcwd()+"/frames.npy").is_file() , True)
    
    def test_DecoderCreation(self):
        """
        Test decodification matches
        """
        databitEncode = EncodeDataIntoBits(self.byte_array, self.slice, self.frame_size, self.filePath)
        databitEncode.process_byte_array()
        # inser encoder values
        databitDecode = DecodeFramesToBits(databitEncode.frames,self.frame_size)
        decoded_output = databitDecode.flatten_frames_in_bytes()
        self.assertEqual(decoded_output.shape, self.byte_array.shape)
        self.assertEqual(decoded_output.all(), self.byte_array.all())
        
        
if __name__ == '__main__':
    unittest.main()


