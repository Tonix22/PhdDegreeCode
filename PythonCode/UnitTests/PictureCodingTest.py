import os
import sys
import unittest
import numpy as np
from pathlib import Path
from PIL import Image

# Agrega el directorio del proyecto a sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Drivers'))
sys.path.insert(0, parent_dir)

from PictureEncoder import PictureEncoder # type: ignore
from PictureDecoder import PictureDecoder # type: ignore

class TestEncoder(unittest.TestCase):
    
    def test_Encoder_Decoder(self):
        # Usage Example
        # Set your image path, desired height, width, and channels (e.g., 3 for RGB, 4 for RGBA, 1 for Grayscale)
        image_path = '/home/tonix/Documents/PhdDegreeCode/PythonCode/UnitTests/th-1079059952.jpeg'
        image = Image.open(image_path)
        width, height = image.size
        channels = len(image.getbands())  # Specify desired channels (1, 3, or 4)

        encoder = PictureEncoder(image_path, height, width, channels)  # Initialize with image path and dimensions
        encoded_bytes = encoder.encode()  # Encode the image to a byte sequence

        decoder = PictureDecoder(encoded_bytes)  # Initialize with encoded byte sequence and dimensions
        recovered_image = decoder.decode()  # Decode the image back to its original form

        # To show the image (optional)
        self.assertEqual(image.size,recovered_image.size)
        recovered_image.save(os.getcwd()+'/th-Rebuild.jpeg')
        
        image.close()
        
if __name__ == '__main__':
    unittest.main()
