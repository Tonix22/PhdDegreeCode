import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

################################################################################
# 1. Utilities: QPSK Mod/Demod, AWGN, BER, One-hot
################################################################################

def qpsk_modulate(symbols, phase_offset):
    """
    QPSK modulator with an added phase offset.
    symbols: 1D array of size [num_symbols], each in {0,1,2,3}
    phase_offset: float, offset in radians
    Returns complex array of shape [num_symbols].
    """
    # Map {0,1,2,3} to QPSK angles {0, pi/2, pi, 3pi/2}
    base_phases = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    # Gather the base phase per symbol
    base_phase_per_symbol = base_phases[symbols]
    # Add the offset
    total_phase = base_phase_per_symbol + phase_offset
    # Convert to complex
    i = np.cos(total_phase)
    q = np.sin(total_phase)
    return i + 1j*q

def qpsk_demodulate(rx_symbols, phase_offset):
    """
    QPSK demodulator, attempting to correct by 'phase_offset'.
    rx_symbols: complex array of shape [num_symbols].
    phase_offset: float, offset in radians (our chosen correction).
    Returns array of hard-decision symbols in {0,1,2,3}.
    """
    # Apply the negative of the offset as "correction"
    corrected_symbols = rx_symbols * np.exp(-1j*phase_offset)

    # Compute angles in [0..2pi)
    angles = np.angle(corrected_symbols)
    angles = np.mod(angles, 2*np.pi)

    # Decision boundaries for QPSK: we partition [0..2pi) into 4 quadrants
    # 0 => [0..pi/2)
    # 1 => [pi/2..pi)
    # 2 => [pi..3pi/2)
    # 3 => [3pi/2..2pi)
    detected_symbols = np.zeros_like(angles, dtype=int)
    detected_symbols[(angles >= 0) & (angles < np.pi/2)] = 0
    detected_symbols[(angles >= np.pi/2) & (angles < np.pi)] = 1
    detected_symbols[(angles >= np.pi) & (angles < 3*np.pi/2)] = 2
    detected_symbols[(angles >= 3*np.pi/2) & (angles < 2*np.pi)] = 3

    return detected_symbols

def add_awgn(tx_symbols, snr_db):
    """
    Add AWGN noise to the transmit symbols according to SNR in dB.
    tx_symbols: complex array of shape [num_symbols].
    snr_db: float, signal-to-noise ratio in dB
    Returns noisy Rx symbols (complex).
    """
    # Calculate symbol power
    power = np.mean(np.abs(tx_symbols)**2)
    # Convert SNR from dB to linear
    snr_linear = 10**(snr_db/10)
    # Noise power based on SNR
    noise_power = power / snr_linear
    # Generate Gaussian noise (real + j*imag)
    noise = np.sqrt(noise_power/2) * (np.random.randn(*tx_symbols.shape) + 
                                      1j*np.random.randn(*tx_symbols.shape))
    return tx_symbols + noise

def compute_ber(true_symbols, detected_symbols):
    """
    Compute bit error rate (BER) between true_symbols and detected_symbols.
    Both are arrays of shape [num_symbols], each in {0,1,2,3}.
    We'll interpret each symbol as 2 bits. 
    """
    # Convert each symbol in {0,1,2,3} to 2-bit binary
    # Example: 0 => 00, 1 => 01, 2 => 10, 3 => 11
    def symbol_to_bits(sym):
        # sym is an int in [0..3]
        b1 = (sym >> 1) & 1
        b0 = sym & 1
        return (b1, b0)

    # Flatten bits
    true_bits = []
    detected_bits = []
    for tsym, rsym in zip(true_symbols, detected_symbols):
        tb = symbol_to_bits(tsym)
        rb = symbol_to_bits(rsym)
        true_bits.extend(tb)
        detected_bits.extend(rb)

    true_bits = np.array(true_bits)
    detected_bits = np.array(detected_bits)
    bit_errors = np.sum(true_bits != detected_bits)
    total_bits = len(true_bits)
    ber = bit_errors / total_bits
    return ber

def one_hot_encode(idx, size):
    """
    One-hot encode an integer 'idx' in [0..size-1].
    Returns a 1D torch tensor of length 'size'.
    """
    vec = torch.zeros(size)
    vec[idx] = 1.0
    return vec

################################################################################
# 2. Environment Definition
################################################################################

class CommEnv:
    """
    A toy environment modeling QPSK transmission over an AWGN channel 
    with an unknown discrete phase offset. The agent tries to guess 
    the correct offset to minimize BER.
    
    - We define a discrete set of possible offsets in {0,1,...,7}.
      That means the actual offset is offset_index * (2*pi/8).
    - The environment's state is the "true offset index" (exposed 
      artificially for demonstration).
    - The agent picks an action in the same discrete set {0,...,7} 
      as the guessed offset correction.
    - The reward is 1 - BER. (Higher = better.)
    - Single-step episode: once you pick an action, we compute BER 
      and end the episode.
    """
    def __init__(self, num_symbols=48, snr_db=10.0):
        self.num_symbols = num_symbols
        self.snr_db = snr_db
        
        # We'll define 8 possible offset indexes (0..7)
        # each offset = offset_index * 2π/8
        self.offset_size = 8
        
        self.action_space = list(range(self.offset_size))  # 0..7
        self.state_size = self.offset_size                # 8 possible states
        self.reset()

    def reset(self):
        """
        - Generate new random data in [0..3].
        - Pick a random true offset index.
        - QPSK-modulate + AWGN.
        - State is the integer offset index (in a real system we wouldn't know this, 
          but this is a toy example).
        Returns state (int).
        """
        self.data = np.random.randint(0, 4, size=(self.num_symbols,))
        self.true_offset_idx = np.random.randint(0, self.offset_size)
        true_offset = 2*np.pi * (self.true_offset_idx / self.offset_size)
        
        # Modulate
        tx_symbols = qpsk_modulate(self.data, phase_offset=true_offset)
        # Add noise
        self.rx_symbols = add_awgn(tx_symbols, snr_db=self.snr_db)
        
        self.done = False
        
        # The environment's "observation" is the index of the offset, 
        # just for demonstration
        return self.true_offset_idx

    def step(self, action):
        """
        action: integer in [0..7], the guessed offset index.
        
        - We demodulate with the guessed offset.
        - Compute BER.
        - Reward = 1 - BER.
        - Episode done = True (single step).
        """
        if self.done:
            raise ValueError("Episode has finished. Call reset() before step().")
        
        guessed_offset = 2*np.pi * (action / self.offset_size)
        
        # Demodulate with the guessed offset
        detected = qpsk_demodulate(self.rx_symbols, guessed_offset)
        self.ber = compute_ber(self.data, detected)
        reward = 1.0 - self.ber
        
        self.done = True
        
        # For a single-step scenario, the next state is meaningless. 
        # Typically we might set it to None or zero. 
        # We'll just keep returning the same offset index for demonstration.
        next_state = self.true_offset_idx
        
        return next_state, reward, self.done

################################################################################
# 3. Q-Network
################################################################################

class QNetwork(nn.Module):
    """
    A simple feed-forward network for discrete Q-values.
    Input dimension = number of states (8).
    Output dimension = number of actions (8).
    """
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.model(x)

################################################################################
# 4. Training Loop (Q-learning)
################################################################################

def train_agent(
    env,
    q_net,
    optimizer,
    num_episodes=2000,
    gamma=0.9,
    epsilon=0.3,
    print_every=200
):
    """
    Standard Q-learning training loop for the single-step environment.
    """
    
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        
        # Convert state to one-hot
        state_tensor = one_hot_encode(state, env.state_size)
        
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.choice(env.action_space)
        else:
            with torch.no_grad():
                q_values = q_net(state_tensor)
                action = torch.argmax(q_values).item()
        
        # Take a step
        next_state, reward, done = env.step(action)
        
        # Convert next_state to one-hot
        # (Though in single-step, we won't do multiple steps)
        next_state_tensor = one_hot_encode(next_state, env.state_size)
        
        # Compute the target
        with torch.no_grad():
            # For single-step, it's basically reward if done
            next_q_values = q_net(next_state_tensor)
            max_next_q = torch.max(next_q_values).item()
            target = reward + (0 if done else gamma * max_next_q)
        
        # Current Q
        q_values = q_net(state_tensor)
        q_val_action = q_values[action]
        
        # TD-loss
        loss = (q_val_action - target)**2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Optionally print
        if (episode+1) % print_every == 0:
            print(f"[Episode {episode+1}] State={state}, Action={action}, "
                  f"Reward={reward:.4f}, Loss={loss.item():.6f}, BER = {env.ber}")

    print("Training complete.")

################################################################################
# 5. Main Execution
################################################################################

if __name__ == "__main__":
    # Create environment
    env = CommEnv(num_symbols=48, snr_db=10.0)
    
    # Create Q-network and optimizer
    q_net = QNetwork(state_size=env.state_size, action_size=len(env.action_space))
    optimizer = optim.Adam(q_net.parameters(), lr=0.001)
    
    # Train
    train_agent(env, q_net, optimizer, 
                num_episodes=2000, 
                gamma=0.9, 
                epsilon=0.3,
                print_every=200)
    
    # You could now test the trained agent or save the model.
