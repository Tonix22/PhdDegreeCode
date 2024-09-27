import numpy as np
from PIL import Image
import zlib  # For CRC32 checksum

class PictureEncoder:
    def __init__(self, image_path, height, width, channels):
        # Load the image and resize it to the specified dimensions
        self.image = Image.open(image_path).resize((width, height))
        # Ensure the image has the required number of channels
        self.image = self.image.convert('RGB') if channels == 3 else self.image.convert('RGBA') if channels == 4 else self.image.convert('L')
        self.height = height
        self.width = width
        self.channels = channels
        self.image_array = np.array(self.image)
        
    def encode(self):
        # Convert shape info to a flat array (4 bytes for each dimension)
        header = np.array([self.height, self.width, self.channels], dtype=np.int32).tobytes()
        # Convert the image to a flat uint8 byte sequence
        byte_sequence = self.image_array.flatten().tobytes()
        # Create CRC32 checksum for the byte sequence
        checksum = zlib.crc32(byte_sequence).to_bytes(4, byteorder='big')  # 4 bytes checksum
        # Footer as a delimiter
        footer = b'END'  # A 3-byte footer
        
        # Return the combined header, byte sequence, checksum, and footer
        return header + byte_sequence + checksum + footer
    
    def encode_in_chunks(self, chunk_size=1024):
        # Convert shape info to a flat array (4 bytes for each dimension)
        header = np.array([self.height, self.width, self.channels], dtype=np.int32).tobytes()
        # Convert the image to a flat uint8 byte sequence
        byte_sequence = self.image_array.flatten().tobytes()
        
        # Split the byte sequence into chunks
        chunks = [byte_sequence[i:i + chunk_size] for i in range(0, len(byte_sequence), chunk_size)]
        encoded_chunks = []
        
        for index, chunk in enumerate(chunks):
            # Create a chunk header with index and size
            chunk_header = np.array([index, len(chunk)], dtype=np.int32).tobytes()
            # Create a CRC32 checksum for the chunk
            checksum = zlib.crc32(chunk).to_bytes(4, byteorder='big')
            # Append the header, chunk, and checksum together
            encoded_chunk = chunk_header + chunk + checksum
            encoded_chunks.append(encoded_chunk)
        
        # Add a final chunk with the END footer to indicate completion
        end_chunk = np.array([-1, 0], dtype=np.int32).tobytes() + b'' + b'END'
        encoded_chunks.append(end_chunk)
        
        # Return the header followed by all the chunks
        return header + b''.join(encoded_chunks)

