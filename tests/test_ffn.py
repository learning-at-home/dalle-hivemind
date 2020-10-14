import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.modules.ffn import LeanFFN


class ReferenceFFN(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 activation=F.gelu,
                 layer_norm_eps=1e-12,
                 dropout: float = 0.0):
        super().__init__()
        self.dense_i2h = nn.Linear(hidden_size, intermediate_size)
        self.dense_h2o = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.activation = activation
        self.dropout = dropout

    def forward(self, input):
        output = self.dense_i2h(self.layer_norm(input))
        output = self.activation(output)
        output = self.dense_h2o(output)
        output = F.dropout(output, self.dropout)
        return output + input


def test_ffn_exact_match():
    torch.use_deterministic_algorithms(True)

    batch_size = 4
    seq_len = 128
    dim = 32
    num_layers = 4

    baseline_ffn = ReferenceFFN(dim, 4 * dim)
    our_ffn = LeanFFN(dim, 4 * dim)

    assert our_ffn.load_state_dict(baseline_ffn.state_dict())

    x = torch.rand(batch_size, seq_len, dim, device='cpu', requires_grad=True)

    # test outputs
    out_ref = x
    for i in range(num_layers):
        out_ref = baseline_ffn.forward(out_ref)

    out_our = x
    for i in range(num_layers):
        out_our = our_ffn(out_our)

    assert torch.allclose(out_our, out_ref)

    # test grad inputs
    obj = (out_ref * (out_ref + 1)).square().mean()
    grad_ref, = torch.autograd.grad(obj, x)

    obj = (out_our * (out_our + 1)).square().mean()
    grad_our, = torch.autograd.grad(obj, x)
    assert torch.allclose(grad_ref, grad_our)

    # test grad params
    x = torch.rand(batch_size, seq_len, dim, device='cpu', requires_grad=True)

    out_ref = x
    for i in range(num_layers):
        out_ref = baseline_ffn.forward(out_ref)

    out_our = x
    for i in range(num_layers):
        out_our = our_ffn(out_our)

    obj = (out_ref * (out_ref + 1)).square().mean()
    grad_params_ref = torch.autograd.grad(obj, list(baseline_ffn.parameters()))

    obj = (out_our * (out_our + 1)).square().mean()
    grad_params_our = torch.autograd.grad(obj, list(our_ffn.parameters()))

    for grad_ref, grad_our in zip(grad_params_ref, grad_params_our):
        assert torch.allclose(grad_ref, grad_our)
