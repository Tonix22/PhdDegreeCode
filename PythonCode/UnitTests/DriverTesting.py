import os
import sys
import unittest
import numpy as np
from pathlib import Path
from commpy.utilities import hamming_dist
from PIL import Image
import math

# Add sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Drivers'))
sys.path.insert(0, parent_dir)

# Import Encoder and Driver
from DataBinEncoder import EncodeDataIntoBits # type: ignore
from DataBinDecoder import DecodeFramesToBits # type: ignore

from PictureEncoder import PictureEncoder # type: ignore
from PictureDecoder import PictureDecoder # type: ignore

from ConstelationCoder import ConstelationCoder, Modulation # type: ignore
from SignalChannelResponse import Channel # type: ignore

class ChannelTesting(unittest.TestCase):
    
    def setUp(self):
        # Setup method to create a sample image or use an existing one
        self.image_path = '/home/tonix/Documents/PhdDegreeCode/PythonCode/UnitTests/dog-face.jpg'
        self.image = Image.open(self.image_path)
        self.width, self.height = self.image.size
        self.channels = len(self.image.getbands())  # Specify desired channels (1, 3, or 4)
        
        self.frame_size = 96
        self.constelation_size = 4
        self.bit_slice = int(math.log2(self.constelation_size))
    
    def test_driverSequence(self):
        
        #Encode picture
        picture_encoder = PictureEncoder(self.image_path, self.height, self.width, self.channels)  # Initialize with image path and dimensions
        encoded_bytes = picture_encoder.encode()  # Encode the image to a byte sequence

        databitEncode = EncodeDataIntoBits(encoded_bytes, self.bit_slice, self.frame_size)
        databitEncode.process_byte_array()
        
        coder = ConstelationCoder(Modulation.QAM, self.constelation_size)
        channel = Channel(5)
        rx_frames = np.empty_like(databitEncode.frames)
        
        for i in range(databitEncode.frames.shape[0]):
            tx_bits = databitEncode.frames[i, :]  # Access the i-th row
            tx = coder.Encode(tx_bits)
            rx = channel.response(tx)
            rx_bits = coder.Decode(rx)
            #save rx frame
            rx_frames[i, :] = rx_bits
            #ber = np.sum(tx_bits != rx_bits)/len(rx_bits)
            #print(ber)
        
        databitDecode = DecodeFramesToBits(rx_frames,self.frame_size)
        decoded_output = databitDecode.flatten_frames_in_bytes()
        #Decode picture
        decoder = PictureDecoder(decoded_output)  # Initialize with encoded byte sequence and dimensions
        recovered_image = decoder.hint_decode(self.height, self.width, self.channels)  # Decode the image back to its original form
        recovered_image.save(os.getcwd()+'/th-ChannelRebuild.jpg')
        self.image.close()
        
    
if __name__ == '__main__':
    unittest.main()