import math
import functools
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
import torch.distributed as dist


def matmul_a_logb(A, B):
    bd = len(B.shape) - 2
    B_max = torch.amax(B, dim=bd, keepdim=True)
    B = B - B_max
    B.exp_()
    C = torch.matmul(A, B)
    C.log_()
    C.add_(B_max)

    return C


def ib_ib_bj_to_ij(pf, pp, cp):
    ll = torch.amax(cp, dim=-1)
    pp = torch.exp(pp - ll[None, :])
    cp = torch.exp(cp - ll[:, None])

    ratio = pf / pp
    ratio[pp == 0.0] = 0.0
    ef = torch.matmul(ratio, cp)

    return ef


class ProdLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        return x + y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, pf):
        return pf, pf


class InputLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, input_ids):
        hidden_size, _ = weights.shape
        outputs = weights[
            torch.arange(0, hidden_size, device=weights.device)[:, None],
            input_ids[None, :]].contiguous() # hidden_size * batch_size
        ctx.save_for_backward(weights, input_ids)
        return outputs

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, pf):
        weights, input_ids = ctx.saved_tensors
        hidden_size, _ = weights.shape
        input_ids = input_ids[None, :].expand(hidden_size, -1)
        weights.grad.scatter_add_(1, input_ids, pf.view(hidden_size, -1))
        return None, None


class InputLayerMissing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, outputs_shape):
        outputs = torch.zeros(outputs_shape, device=weights.device)
        ctx.save_for_backward(weights)
        return outputs

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, pf):
        weights = ctx.saved_tensors[0]
        weights_exp = torch.softmax(weights, dim=-1)
        pf = torch.sum(pf.squeeze(), dim=1)
        weights.grad.add_(weights_exp * pf[:, None])
        return None, None


class LinearLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, inputs):
        outputs = matmul_a_logb(weights, inputs)
        ctx.save_for_backward(weights, inputs, outputs)
        return outputs

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, pf):
        weights_exp, cp, pp = ctx.saved_tensors

        ef = ib_ib_bj_to_ij(pf, pp,
            torch.transpose(cp, 0, 1).contiguous()) * weights_exp

        pp_max = torch.amax(pp, dim=0, keepdim=True) # 1 * batch_size
        ratio = pf / torch.exp(pp - pp_max)
        ratio.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        cf = torch.matmul(
            torch.transpose(weights_exp, 0, 1).contiguous(),
            ratio) * torch.exp(cp - pp_max)

        return ef, cf


class HMM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, hidden_size, vocab_size, eos_token_id, sep_token_id=None):
        super().__init__()

        alpha_exp = torch.softmax(torch.randn(hidden_size, hidden_size), dim=1)
        beta = torch.log_softmax(torch.randn(hidden_size, vocab_size), dim=1)
        gamma_exp = torch.softmax(torch.randn(1, hidden_size), dim=0)

        if sep_token_id is not None:
            ################# SEP TOKEN INITIALIZATION #################
            beta[-1, sep_token_id] = 1e10
            beta[:-1, sep_token_id] = -1e10
            beta = torch.log_softmax(beta, dim=-1)
            alpha_exp[-1, -1] = 1e-10
            ################# SEP TOKEN INITIALIZATION #################
        else:
            ################# EOS TOKEN INITIALIZATION #################
            beta[-1, eos_token_id] = 1e10
            beta[:-1, eos_token_id] = -1e10
            beta = torch.log_softmax(beta, dim=-1)
            alpha_exp[-1, :] = 1e-10
            alpha_exp[-1, -1] = 1.0
            ################# EOS TOKEN INITIALIZATION #################

        self.alpha_exp = nn.Parameter(alpha_exp, requires_grad=True)
        self.beta = nn.Parameter(beta, requires_grad=True)
        self.gamma_exp = nn.Parameter(gamma_exp, requires_grad=True)

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.sep_token_id = sep_token_id

    def zero_grad(self):
        for name, param in self.named_parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param.data, device=param.device)
                param.node_flow = torch.zeros(param.data.shape[:-1] + (1,), device=param.device)
            else:
                param.grad.fill_(0.0)

    def step(self, lr=1.0, neginf=-1e10, eps=1e-10):
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            with torch.no_grad():
                torch.distributed.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.mul_(lr)

                if name.find('exp') == -1:
                    param.data.exp_()
                param.data.mul_((1.0 - lr) * param.node_flow)

                param.data.add_(param.grad)
                param.node_flow = torch.sum(param.data, dim=-1, keepdim=True)

                param.data.div_(param.node_flow)
                param.data.clamp_(min=eps)
                if name.find('exp') == -1:
                    param.data.log_()
                    param.data.nan_to_num_(nan=neginf, neginf=neginf)

    # bottom-up circuit pass
    def forward(self, input_ids):
        hidden_size = self.hidden_size
        batch_size, seq_len = input_ids.shape

        y = torch.zeros((hidden_size, batch_size), device=self.alpha_exp.device)
        for t in range(seq_len-1, -1, -1):
            if t != seq_len - 1:
                y = LinearLayer.apply(self.alpha_exp, y)
            input_prob = InputLayer.apply(self.beta, input_ids[:, t])
            y = ProdLayer.apply(input_prob, y)
        y = LinearLayer.apply(self.gamma_exp, y)

        return y