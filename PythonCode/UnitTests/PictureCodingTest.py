import os
import sys
import unittest
import numpy as np
from pathlib import Path
from PIL import Image

# Add sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Drivers'))
sys.path.insert(0, parent_dir)

from PictureEncoder import PictureEncoder # type: ignore
from PictureDecoder import PictureDecoder # type: ignore

class PictureCodingTest(unittest.TestCase):
    
    def setUp(self):
        # Setup method to create a sample image or use an existing one
        self.image_path = '/home/tonix/Documents/PhdDegreeCode/PythonCode/UnitTests/th-1079059952.jpeg'
        self.image = Image.open(self.image_path)
        self.width, self.height = self.image.size
        self.channels = len(self.image.getbands())  # Specify desired channels (1, 3, or 4)
    
    def test_Encoder_Decoder(self):

        encoder = PictureEncoder(self.image_path, self.height, self.width, self.channels)  # Initialize with image path and dimensions
        encoded_bytes = encoder.encode()  # Encode the image to a byte sequence

        decoder = PictureDecoder(encoded_bytes)  # Initialize with encoded byte sequence and dimensions
        recovered_image = decoder.decode()  # Decode the image back to its original form

        # To show the image (optional)
        self.assertEqual(self.image.size,recovered_image.size)
        recovered_image.save(os.getcwd()+'/th-Rebuild.jpeg')
        
        self.image.close()
        
    def test_encode_decode_chunks(self):
        # Initialize the encoder with the sample image
        encoder = PictureEncoder(self.image_path, self.height, self.width, self.channels)
        
        # Encode the image into chunks
        encoded_chunks = encoder.encode_in_chunks(chunk_size=1024)  # Encode in chunks of 1024 bytes

        # Initialize the decoder with the encoded chunks
        decoder = PictureDecoder(encoded_chunks)
        recovered_image = decoder.decode_chunks()  # Decode the chunks back to the image
        
        # Verify that the reconstructed image has the same size as the original
        self.assertEqual((self.width, self.height), recovered_image.size)

        # Optionally, save the recovered image to a file
        output_path = os.path.join(os.getcwd(), 'recovered_image_chunks.jpeg')
        recovered_image.save(output_path)

        # Close the original image to free resources
        self.image.close()    
        
    def tearDown(self):
        # Clean up after each test case
        if os.path.exists(os.getcwd()+'/recovered_image_chunks.jpeg'):
            os.remove(os.getcwd()+'/recovered_image_chunks.jpeg')
        
if __name__ == '__main__':
    unittest.main()
