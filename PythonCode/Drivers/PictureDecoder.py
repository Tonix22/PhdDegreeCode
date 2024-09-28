import numpy as np
from PIL import Image
import zlib  # For CRC32 checksum

class PictureDecoder:
    def __init__(self, encoded_data):
        # Initialize with the encoded byte sequence
        self.encoded_data = encoded_data.tobytes()

    def hint_decode(self, height, width, channels):
        image_byte_length = height * width * channels
        image_data = self.encoded_data[12:12 + image_byte_length]
        image_array = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, channels))
        # Return the reconstructed image
        if channels == 1:
            return Image.fromarray(image_array.squeeze(), mode='L')  # Grayscale
        elif channels == 3:
            return Image.fromarray(image_array, mode='RGB')  # RGB
        elif channels == 4:
            return Image.fromarray(image_array, mode='RGBA')  # RGBA
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")

    def decode(self):
        try:
            # Extract the header (first 12 bytes for height, width, channels)
            height, width, channels = np.frombuffer(self.encoded_data[:12], dtype=np.int32)
            
            # Calculate the image byte length (height * width * channels)
            image_byte_length = height * width * channels

            # Extract the image byte sequence
            image_data = self.encoded_data[12:12 + image_byte_length]
            
            # Extract and verify the checksum (next 4 bytes after image data)
            received_checksum = int.from_bytes(self.encoded_data[12 + image_byte_length:12 + image_byte_length + 4], byteorder='big')
            calculated_checksum = zlib.crc32(image_data)
            
            if received_checksum != calculated_checksum:
                raise ValueError("Checksum mismatch! The data may be corrupted.")
            
            # Check the footer to verify the end of the message (last 3 bytes)
            footer = self.encoded_data[-3:]
            if footer != b'END':
                raise ValueError("Invalid footer! Data may be incomplete or corrupted.")
            
            # Reshape the image data to its original dimensions
            image_array = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, channels))
            
            # Return the reconstructed image
            if channels == 1:
                return Image.fromarray(image_array.squeeze(), mode='L')  # Grayscale
            elif channels == 3:
                return Image.fromarray(image_array, mode='RGB')  # RGB
            elif channels == 4:
                return Image.fromarray(image_array, mode='RGBA')  # RGBA
            else:
                raise ValueError(f"Unsupported number of channels: {channels}")
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None

    def decode_chunks(self):
        try:
            # Extract the initial header (first 12 bytes for height, width, channels)
            header_size = 12
            height, width, channels = np.frombuffer(self.encoded_data[:header_size], dtype=np.int32)
            
            # Initialize variables for chunk processing
            chunk_start = header_size
            image_data = b''

            # Process each chunk
            while chunk_start < len(self.encoded_data):
                # Extract chunk header (index and size)
                chunk_header_size = 8
                chunk_header = self.encoded_data[chunk_start:chunk_start + chunk_header_size]
                chunk_index, chunk_size = np.frombuffer(chunk_header, dtype=np.int32)
                
                # End of transmission if index is -1
                if chunk_index == -1:
                    break
                
                # Calculate the chunk data position and extract the chunk
                chunk_data_start = chunk_start + chunk_header_size
                chunk_data_end = chunk_data_start + chunk_size
                chunk_data = self.encoded_data[chunk_data_start:chunk_data_end]
                
                # Extract the checksum for the chunk
                checksum_start = chunk_data_end
                checksum_end = checksum_start + 4
                received_checksum = int.from_bytes(self.encoded_data[checksum_start:checksum_end], byteorder='big')
                
                # Calculate and verify the checksum
                calculated_checksum = zlib.crc32(chunk_data)
                if received_checksum != calculated_checksum:
                    raise ValueError(f"Checksum mismatch in chunk {chunk_index}! Data may be corrupted.")
                
                # Append the chunk data to the image data
                image_data += chunk_data

                # Move to the next chunk position
                chunk_start = checksum_end
            
            # Reshape the complete image data to its original dimensions
            image_array = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, channels))
            
            # Return the reconstructed image
            if channels == 1:
                return Image.fromarray(image_array.squeeze(), mode='L')  # Grayscale
            elif channels == 3:
                return Image.fromarray(image_array, mode='RGB')  # RGB
            elif channels == 4:
                return Image.fromarray(image_array, mode='RGBA')  # RGBA
            else:
                raise ValueError(f"Unsupported number of channels: {channels}")
        except Exception as e:
            print(f"Error decoding chunks: {e}")
            return None