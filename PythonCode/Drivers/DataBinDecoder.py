import numpy as np

class DecodeFramesToBits:
    def __init__(self, frames, frame_size=48):
        """
        Initialize with an array of frames of size (x, frame_size) and flatten it into bits.
        """
        self.frames = frames
        self.frame_size = frame_size

    def flatten_frames_in_bytes(self):
        """
        Flatten the frame array into a continuous sequence of bits.
        Last value of sequence must never be zero
        Input: frames of shape (num_frames, frame_size)
        Output: Array of Hex without trailling zeros removed.
        """
        # Flatten the frames array into a 1D array of bits
        flattened_frames = self.frames.flatten()

        # Reshape to (48, x) preserving the order of the original frames
        # Calculate the total number of bits
        total_bits = len(flattened_frames)

        if total_bits % 8 != 0:
            padding_length = 8 - (total_bits % 8)
            bit_array = np.pad(bit_array, (0, padding_length), 'constant')

        # Reshape the bit array into groups of 8 bits
        bit_array = self.frames.reshape(-1, 8)

        # Convert each group of 8 bits to a byte (uint8)
        byte_array = np.packbits(bit_array, axis=1).flatten()
        
        last_non_zero_index = np.max(np.nonzero(byte_array)) + 1

        return byte_array[:last_non_zero_index]
