# =======================================================================================================================
# =======================================================================================================================
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import hdf5storage


# =======================================================================================================================
# =======================================================================================================================


class SE_Loss(nn.Module):
    def __init__(self, snr=10, device=torch.device("cuda:0")):
        super().__init__()
        self.SNR = snr
        self.device = device

    def forward(self, h, h0):
        # input : h:  B, Nt, Nr (complex)      h0: B, Nt, Nr (complex)
        # 1. prepare data
        SNR = self.SNR
        B, Nt, Nr = h.shape
        H = h.to(self.device)  # B * Nr * Nt
        H0 = h0.to(self.device)  # B * Nr * Nt
        if Nr != 1:
            S_real = torch.diag(torch.ones(Nr, 1).squeeze()).unsqueeze(0).repeat([B, 1, 1])  # b,2 * 2
        elif Nr == 1:
            S_real = torch.diag(torch.ones(Nr, 1)).unsqueeze(0).repeat([B, 1, 1])  # b,2 * 2
        S_imag = torch.zeros([B, Nr, Nr])
        S = torch.complex(S_real, S_imag).to(device=self.device)
        matmul0 = torch.matmul(H0, S)
        fro = torch.norm(matmul0, p='fro', dim=(1, 2))  # B,1
        noise_var = (torch.pow(fro, 2) / (Nt * Nr)) * pow(10, (-SNR / 10))
        # 2. get D and D0
        D = torch.adjoint(H)
        D = torch.div(D, torch.norm(D, p=2, dim=(1, 2), keepdim=True))
        D0 = torch.adjoint(H0)
        D0 = torch.div(D0, torch.norm(D0, p=2, dim=(1, 2), keepdim=True))
        # 3. get SE and SE0
        matmul1 = torch.matmul(D, H0)
        matmul2 = torch.matmul(D0, H0)

        noise_var = noise_var.unsqueeze(1).unsqueeze(1)  # B,1,1
        SE = -torch.log2(torch.det(torch.div(torch.pow(torch.abs(matmul1), 2), noise_var) + S))  # B
        SE = torch.mean(SE.real)

        SE0 = -torch.log2(torch.det(torch.div(torch.pow(torch.abs(matmul2), 2), noise_var) + S))  # B
        SE0 = torch.mean(SE0.real)

        return SE, SE0


def NMSE_cuda(x_hat, x):
    power = torch.sum(x ** 2)
    mse = torch.sum((x - x_hat) ** 2)
    nmse = mse / power
    return nmse


class NMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x_hat, x)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse)
        else:
            nmse = torch.sum(nmse)
        return nmse


class BER_Loss(nn.Module):
    """
    Bit Error Rate (BER) calculator for 4-QAM modulation.

    BER describes the communication reliability at a certain transmission rate.
    4-QAM (quadrature amplitude modulation) is adopted.
    The communication SNR is set as specified (default 10 dB).
    """
    def __init__(self, snr=10, device=torch.device("cuda:0")):
        super().__init__()
        self.SNR = snr
        self.device = device

    def qam4_modulate(self, bits):
        """
        Modulate bits to 4-QAM symbols.
        00 -> -1-1j, 01 -> -1+1j, 10 -> 1-1j, 11 -> 1+1j
        """
        # Reshape bits to pairs: (batch, n_symbols, 2)
        bits = bits.reshape(-1, 2)
        # Convert to 4-QAM symbols
        symbols = torch.zeros(bits.shape[0], dtype=torch.complex64, device=self.device)
        symbols[(bits[:, 0] == 0) & (bits[:, 1] == 0)] = -1 - 1j
        symbols[(bits[:, 0] == 0) & (bits[:, 1] == 1)] = -1 + 1j
        symbols[(bits[:, 0] == 1) & (bits[:, 1] == 0)] = 1 - 1j
        symbols[(bits[:, 0] == 1) & (bits[:, 1] == 1)] = 1 + 1j
        return symbols

    def qam4_demodulate(self, symbols):
        """
        Demodulate 4-QAM symbols to bits (vectorized).
        -1-1j -> 00, -1+1j -> 01, 1-1j -> 10, 1+1j -> 11
        """
        # Vectorized demodulation: (batch * num_symbols,) -> (batch * num_symbols * 2,)
        real = symbols.real  # (N,)
        imag = symbols.imag  # (N,)

        # bit 0: real < 0 -> 0, real >= 0 -> 1
        bit0 = (real >= 0).long()

        # bit 1: imag < 0 -> 0, imag >= 0 -> 1
        bit1 = (imag >= 0).long()

        # Interleave bits: [bit0, bit1, bit0, bit1, ...]
        bits = torch.stack([bit0, bit1], dim=-1).reshape(-1)  # (2 * N,)
        return bits

    def forward(self, h, h0, num_symbols=100):
        """
        Calculate BER using predicted CSI (h) and true CSI (h0) (vectorized).

        Args:
            h: Predicted CSI, shape (B, Nt, Nr) complex
            h0: True CSI, shape (B, Nt, Nr) complex
            num_symbols: Number of symbols to transmit for BER calculation

        Returns:
            BER: Bit error rate using predicted CSI
            BER0: Bit error rate using true CSI (baseline)
        """
        B, Nt, Nr = h.shape

        # Generate random bits - same for all batches
        total_bits = num_symbols * 2
        bits = torch.randint(0, 2, (total_bits,), device=self.device)

        # Modulate to 4-QAM symbols
        tx_symbols = self.qam4_modulate(bits)  # (num_symbols,)

        # Calculate noise power based on SNR
        signal_power = torch.mean(torch.abs(tx_symbols) ** 2)
        noise_power = signal_power / (10 ** (self.SNR / 10))
        noise_std = torch.sqrt(noise_power / 2)

        # Vectorized batch processing
        # Hermitian transpose: (B, Nt, Nr) -> (B, Nr, Nt)
        h_hermitian = torch.conj(h).transpose(-1, -2)  # (B, Nr, Nt)
        h0_hermitian = torch.conj(h0).transpose(-1, -2)  # (B, Nr, Nt)

        # Normalize precoding vectors
        h_norm = torch.norm(h_hermitian, p='fro', dim=(1, 2), keepdim=True) + 1e-10  # (B, 1, 1)
        h0_norm = torch.norm(h0_hermitian, p='fro', dim=(1, 2), keepdim=True) + 1e-10  # (B, 1, 1)

        w = h_hermitian / h_norm  # (B, Nr, Nt)
        w0 = h0_hermitian / h0_norm  # (B, Nr, Nt)

        # Expand tx_symbols for batch processing: (num_symbols,) -> (B, num_symbols, 1)
        tx_symbols_expanded = tx_symbols.unsqueeze(0).unsqueeze(-1)  # (1, num_symbols, 1)
        tx_symbols_expanded = tx_symbols_expanded.expand(B, -1, -1)  # (B, num_symbols, 1)

        # Apply precoding using einsum for clear batch broadcasting
        # w is (B, Nr, Nt) = (B, 1, 16)
        # tx_symbols_expanded is (B, num_symbols, 1)
        # Result: (B, num_symbols, 1) @ (B, 1, Nt) -> (B, num_symbols, Nt)
        tx_precoded = torch.einsum('bsi,bij->bsj', tx_symbols_expanded, w)  # (B, num_symbols, Nt)
        tx_precoded0 = torch.einsum('bsi,bij->bsj', tx_symbols_expanded, w0)  # (B, num_symbols, Nt)

        # Channel propagation: (B, num_symbols, Nt) @ (B, Nt, Nr) -> (B, num_symbols, Nr)
        rx = torch.bmm(tx_precoded, h)  # (B, num_symbols, Nr)
        rx0 = torch.bmm(tx_precoded0, h0)  # (B, num_symbols, Nr)

        # Add noise - same noise std for all
        noise = noise_std * (torch.randn_like(rx.real) + 1j * torch.randn_like(rx.imag))
        noise0 = noise_std * (torch.randn_like(rx0.real) + 1j * torch.randn_like(rx0.imag))

        rx_noisy = rx + noise  # (B, num_symbols, Nr)
        rx_noisy0 = rx0 + noise0  # (B, num_symbols, Nr)

        # Equalization: for Nr=1, squeeze to (B, num_symbols)
        rx_equalized = rx_noisy.squeeze(-1)  # (B, num_symbols)
        rx_equalized0 = rx_noisy0.squeeze(-1)  # (B, num_symbols)

        # Flatten for demodulation: (B, num_symbols) -> (B * num_symbols,)
        rx_flat = rx_equalized.reshape(-1)  # (B * num_symbols,)
        rx_flat0 = rx_equalized0.reshape(-1)  # (B * num_symbols,)

        # Demodulate all symbols at once
        rx_bits = self.qam4_demodulate(rx_flat)  # (B * num_symbols * 2,)
        rx_bits0 = self.qam4_demodulate(rx_flat0)  # (B * num_symbols * 2,)

        # Broadcast bits to all batches: (total_bits,) -> (B * total_bits,)
        bits_broadcast = bits.unsqueeze(0).expand(B, -1).reshape(-1)

        # Count errors
        ber_errors = torch.sum(bits_broadcast != rx_bits).item()
        ber_errors0 = torch.sum(bits_broadcast != rx_bits0).item()

        # Average BER over all batches
        BER = ber_errors / (B * total_bits)
        BER0 = ber_errors0 / (B * total_bits)

        return BER, BER0
