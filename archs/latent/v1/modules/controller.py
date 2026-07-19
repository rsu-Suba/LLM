import torch
import torch.nn as nn

class EntropyFusedController(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim * 3 + 1, hidden), nn.SiLU(), nn.Linear(hidden, 2))
    def forward(self, latent, previous, entropy):
        delta = latent - previous
        x = torch.cat([latent, previous, delta, entropy], dim=-1)
        logits = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        read = probs[..., 0:1]
        skip = probs[..., 1:2]
        return read, skip

class PointerTracker(nn.Module):
    def __init__(self, dim, max_delta=32):
        super().__init__()
        self.max_delta = max_delta
        self.delta = nn.Linear(dim, 1)
    def forward(self, latent, pointer, seq_len):
        move = torch.tanh(self.delta(latent))
        move = move * self.max_delta
        pointer = pointer + move
        pointer = pointer.clamp(0, seq_len - 1)
        return pointer

class WindowPredictor(nn.Module):
    def __init__(self, dim, minimum=4, maximum=128):
        super().__init__()
        self.minimum = minimum
        self.maximum = maximum
        self.linear = nn.Linear(dim, 1)
    def forward(self, latent):
        w = torch.sigmoid(self.linear(latent))
        return self.minimum + (self.maximum - self.minimum) * w

