#!/usr/bin/env python3

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

###############################################################################
# Adjust path so we can import DPSK_OFDM
###############################################################################
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../App'))
sys.path.insert(0, parent_dir)

from DPSK import DPSK_OFDM  # <-- Make sure this matches your file name & class location

###############################################################################
# 1) The Environment
###############################################################################

class FeedForwardNet(nn.Module):
    """
    Example feed-forward net that processes:
      - Real+Imag input (length=2*num_bits)
      - A single scalar offset (guessed offset in radians)
    Then outputs a length=num_bits vector (e.g., bit likelihoods).
    """
    def __init__(self, input_dim, offset_dim, output_dim):
        super(FeedForwardNet, self).__init__()
        # We'll embed the offset dimension into a small hidden layer
        self.offset_fc = nn.Linear(offset_dim, 8)  # offset => 8 features

        # Then process the real+imag with a separate linear
        self.signal_fc = nn.Linear(input_dim, 128)

        # Combine them
        self.fc_combined = nn.Linear(128 + 8, 64)
        self.fc_out = nn.Linear(64, output_dim)

    def forward(self, signal_in, offset_in):
        """
        signal_in: shape [batch_size, input_dim] (real+imag)
        offset_in: shape [batch_size, offset_dim] (scalar offset)
        Returns: shape [batch_size, output_dim]
        """
        offset_feat = torch.relu(self.offset_fc(offset_in))  
        sig_feat = torch.relu(self.signal_fc(signal_in))
        combined = torch.cat([sig_feat, offset_feat], dim=1)
        x = torch.relu(self.fc_combined(combined))
        out = torch.tanh(self.fc_out(x))
        return out

class DPSKPhaseOffsetEnv:
    """
    A single-step environment that uses your DPSK_OFDM system.
    Steps:
      1) On reset(), we:
         - generate random bits
         - force first N bits to 0
         - DPSK encode
         - apply random phase offset + noise
      2) On step(action), we:
         - interpret action as guessed offset
         - feed offset + signal into a feed-forward net (placeholder)
         - decode => get bit errors => compute reward => done=True
    """

    def __init__(self, numSC=48, N=0, snr_dB=10.0, offset_size=8):
        """
        Args:
            numSC (int): Number of subcarriers => we have 2*numSC bits.
            N (int): Number of leading bits forced to 0.
            snr_dB (float): SNR in dB for AWGN.
            offset_size (int): Number of discrete offset guesses in [0..offset_size-1].
        """
        # Create your DPSK_OFDM system
        self.ofdm_system = DPSK_OFDM(
            snr_dB_range=[snr_dB],
            modulation_order=4,   # QPSK
            fft_size=numSC,       # We'll use 'numSC' for FFT size as well
            num_subcarriers=numSC,
            channel_snr=snr_dB,
            los=True
        )
        
        self.numSC = numSC
        self.num_bits = 2 * self.numSC
        self.N = N
        self.snr_dB = snr_dB

        # We'll define discrete actions: offset_idx in [0..offset_size-1]
        # offset = 2*pi*(offset_idx/offset_size)
        self.offset_size = offset_size
        self.action_space = list(range(offset_size))

        self.done = False

        # The feed-forward net that processes (Rx_signal, guessed_offset).
        # In a real system, you might let the RL agent control this net’s parameters,
        # but here we keep it in the env as a placeholder.
        self.ff_net = FeedForwardNet(
            input_dim=self.numSC * 2,   # 48 * 2 = 96 if numSC=48
            offset_dim=1,
            output_dim=self.num_bits    # 2 * numSC = 96 bits if QPSK
        )


    def reset(self):
        """Reset environment for a new single-step episode."""
        self.done = False

        # 1) Generate random bits
        signalTx = np.random.randint(0, 2, self.num_bits).astype(np.int32)

        # 2) Force first N bits to 0
        if self.N > 0:
            signalTx[: self.N] = 0

        # 3) DPSK encode
        dpsk_signal = self.ofdm_system.DPSK_encoder(signalTx)

        # 4) Random offset + noise
        self.true_offset_idx = np.random.randint(0, self.offset_size)
        self.true_offset = 2 * np.pi * (self.true_offset_idx / self.offset_size)

        # Apply offset + AWGN manually
        rx_signal = self.apply_phase_offset_and_noise(dpsk_signal, self.true_offset, self.snr_dB)

        # Store
        self.signalTx = signalTx
        self.dpsk_signal = dpsk_signal
        self.rx_signal = rx_signal

        # Return an observation (the real+imag as a tensor),
        # though we won't necessarily use it for discrete Q-learning
        obs = self._make_observation(rx_signal)
        return obs

    def step(self, action):
        """
        Single-step:
          - Convert action => offset
          - Pass (rx_signal, offset) into the feed-forward net
          - "Decode" => measure bit errors => reward
          - Episode ends
        """
        if self.done:
            raise ValueError("Episode already ended. Call reset() first.")

        guessed_offset = 2 * np.pi * (action / self.offset_size)

        # Feed to FF net
        rx_real_imag = self._make_observation(self.rx_signal)  # shape [2*num_bits]
        offset_tensor = torch.FloatTensor([guessed_offset]).unsqueeze(0)

        net_in = rx_real_imag.unsqueeze(0)
        net_out = self.ff_net(net_in, offset_tensor)  # shape [1, num_bits]
        net_out_np = net_out.detach().numpy().squeeze()

        # We'll do a trivial "decode" of net_out => bits
        signalEstimate = (net_out_np > 0.5).astype(np.int32)

        # Count errors
        errors = np.bitwise_xor(self.signalTx, signalEstimate).sum()
        errors = int(errors)
        reward = -errors  # negative of errors

        self.done = True
        info = {
            "errors": errors,
            "BER": errors / self.num_bits,
            "true_offset_idx": self.true_offset_idx
        }

        return None, reward, self.done, info

    def apply_phase_offset_and_noise(self, dpsk_signal, offset, snr_dB):
        """Helper: multiply by e^{j*offset} and add AWGN."""
        rx_signal = dpsk_signal * np.exp(1j * offset)
        snr_linear = 10 ** (snr_dB / 10)
        power = np.mean(np.abs(dpsk_signal)**2)
        noise_power = power / snr_linear
        noise_std = np.sqrt(noise_power / 2)
        noise = noise_std * (np.random.randn(*dpsk_signal.shape) 
                             + 1j * np.random.randn(*dpsk_signal.shape))
        rx_signal += noise
        return rx_signal

    def _make_observation(self, rx_signal):
        """Convert complex array => real+imag -> torch float tensor."""
        rx_real = rx_signal.real.astype(np.float32)
        rx_imag = rx_signal.imag.astype(np.float32)
        return torch.from_numpy(np.concatenate([rx_real, rx_imag], axis=0))

###############################################################################
# 2) Q-Network and Q-Learning
###############################################################################

class QNetwork(nn.Module):
    """
    Minimal Q-network for discrete action selection with a *dummy* state.
    If you truly want to incorporate the environment's real+imag
    as input, you'd need a different approach (like DQN for continuous states).
    """
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, action_size)

    def forward(self, x):
        # x shape: [batch_size, state_size]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # shape [batch_size, action_size]
        return x

def one_hot_encode(idx, size):
    """Convert integer 'idx' to a 1D one-hot vector of length 'size'."""
    vec = torch.zeros(size)
    vec[idx] = 1.0
    return vec

def train_qlearning(env, q_net, optimizer, num_episodes=500, epsilon=0.3, gamma=0.0, print_every=50):
    """
    Single-step Q-learning loop:
      - We'll treat the 'state' as a simple integer (always 0).
      - We pick action with epsilon-greedy from q_net.
      - We step once => reward => done.
      - Q(s,a)=reward for the terminal state.
    """
    state_size = 1  # we have only one dummy state "0"
    action_size = len(env.action_space)

    for episode in range(num_episodes):
        # 1) Reset environment
        env.reset()
        state_idx = 0  
        state_vec = one_hot_encode(state_idx, state_size).unsqueeze(0)  # shape [1,1]

        # 2) Epsilon-greedy
        if random.random() < epsilon:
            action = random.choice(env.action_space)
        else:
            with torch.no_grad():
                q_values = q_net(state_vec)
            action = torch.argmax(q_values, dim=1).item()

        # 3) Step
        _, reward, done, info = env.step(action)

        # single-step => done=True => target=reward
        target = reward

        # current Q
        q_values = q_net(state_vec)[0]  
        q_val_action = q_values[action]
        loss = (q_val_action - target)**2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if (episode+1) % print_every == 0:
            errors = info["errors"]
            ber = info["BER"]
            print(f"Ep {episode+1} | Action={action} | Reward={reward} "
                  f"| Errors={errors} | BER={ber:.4f} | Loss={loss.item():.4f}")

    print("Training complete.")

###############################################################################
# 3) Main Script
###############################################################################

if __name__ == "__main__":
    # 1) Create environment
    env = DPSKPhaseOffsetEnv(
        numSC=48,
        N=0,
        snr_dB=10.0,
        offset_size=8
    )

    # 2) Create Q-network
    #    state_size=1 (dummy), action_size=8
    q_net = QNetwork(state_size=1, action_size=len(env.action_space))

    # 3) Optimizer
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)

    # 4) Train
    train_qlearning(env, q_net, optimizer,
                    num_episodes=10000,
                    epsilon=0.9,
                    gamma=0.5,
                    print_every=100)
