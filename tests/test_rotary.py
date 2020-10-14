import torch

from lib.modules.rotary import get_auxiliary_tensors, RotaryEmbeddings


def test_rotary_embeddings():
    batch_size = 11
    seq_len = 50
    nhead = 4
    dim = 1024
    dtype = torch.float32
    device = torch.device('cpu')
    base = 10 ** 4

    torch.use_deterministic_algorithms(True)

    # auxiliary tensors
    a, b = get_auxiliary_tensors(seq_len, dim, dtype, device, base)
    x, y = Rotary(dim, base).forward(torch.randn(1, seq_len, dim, device=device))
    assert torch.allclose(a.view_as(x), x, atol=1e-4, rtol=0)
    assert torch.allclose(b.view_as(y), y, atol=1e-4, rtol=0)

    # full layer outputs
    ref_layer = Rotary(dim, base)
    our_layer = RotaryEmbeddings(dim, base)
    q = torch.randn(batch_size, seq_len, nhead, dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seq_len, nhead, dim, dtype=dtype, device=device)

    q_ref, k_ref = apply_rotary_pos_emb(q.permute(1, 0, 2, 3), k.permute(1, 0, 2, 3), *ref_layer(k))
    q_our, k_our = our_layer(q), our_layer(k)
    assert torch.allclose(q_ref.permute(1, 0, 2, 3), q_our, atol=1e-4, rtol=0)
    assert torch.allclose(k_ref.permute(1, 0, 2, 3), k_our, atol=1e-4, rtol=0)

    # check rotation equivariance of dot product
    original_dot = (q[0, :, 0] * k[0, :, 0]).sum(-1)
    rotated_dot = (our_layer(q)[0, :, 0] * our_layer(k)[0, :, 0]).sum(-1)
    assert torch.allclose(original_dot, rotated_dot, atol=1e-4, rtol=0)


class Rotary(torch.nn.Module):
    """ Reference implementation by ElutherAI """
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in torch < 1.8.0


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)