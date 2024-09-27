import numpy as np

class EncodeDataIntoBits():
    def __init__(self, byte_array, slice):
        self.byte_array = byte_array
        self.slice = slice

    def bytes_to_bits(self, byte_array):
        """Convert a numpy array of uint8 bytes to a numpy array of bits (0 or 1)."""
        bits = np.unpackbits(byte_array)  # Convert uint8 to bits
        return bits

    def slice_bits(self, bits, slice_size):
        """Slice the array of bits into windows of a specific size."""
        total_bits = len(bits)
        # Calculate the number of slices
        num_slices = total_bits // slice_size
        
        # Create an array of shape (num_slices, slice_size) and pad with 0s if necessary
        bits = bits.reshape((num_slices, slice_size))
        
        return bits


    def group_slices_to_frames(self, slices, frame_size=48):
        """Concatenate slices into frames of size 'frame_size'. Pad if necessary."""
        # Flatten the slices into a continuous sequence of bits
        concatenated_slices = slices.flatten()

        total_bits = len(concatenated_slices)
        # Calculate the number of frames required
        num_frames = (total_bits + frame_size - 1) // frame_size

        # Pad the concatenated slices to fit into a multiple of frame_size
        padded_slices = np.pad(concatenated_slices, (0, num_frames * frame_size - total_bits), 'constant')
        
        # Reshape into frames of shape (num_frames, frame_size)
        frames = padded_slices.reshape((num_frames, frame_size))

        # Ensure the output is a regular integer array (not numpy arrays)
        frames = frames.astype(np.uint8)  # Convert to uint8 for clarity
        return frames.T  # Transpose to get (48, X) shape

    def save_frames_to_npy(self,filename="frames.npy"):
        """Save the frames array to an npy file."""
        np.save(filename, self.frames)
        print(f"Frames saved to {filename}")

    def process_byte_array(self):
        # Step 1: Convert byte array (uint8) to bit array
        bits = self.bytes_to_bits(self.byte_array)
        
        # Step 2: Slice the bits into the desired slice size
        slices = self.slice_bits(bits, self.slice_size)
        
        # Step 3: Group the slices into frames of size 48
        self.frames = self.group_slices_to_frames(slices)
        # Step 4: Save the frames to a .npy file


# Example usage
byte_array = np.array([0xA2, 0x3A,0xFE,0X30,0xA2, 0xAA,0xF0,0XF0], dtype=np.uint8)  # Example byte array
databitEncode = EncodeDataIntoBits(byte_array,2)
databitEncode.process_byte_array()
databitEncode.save_frames_to_npy()
# to get the each column is with frames[:, col_idx] 
