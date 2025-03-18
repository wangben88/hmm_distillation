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

        alpha_exp = torch.softmax(torch.randn(hidden_size, hidden_size), dim=-1)
        beta = torch.log_softmax(torch.randn(hidden_size, vocab_size), dim=-1)
        gamma_exp = torch.softmax(torch.randn(1, hidden_size), dim=-1)

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

    def compute_forward_probability(self, input_ids):
        batch_size, m = input_ids.size()
        x_1 = input_ids[:, 0]
        emission = self.beta[:, x_1].transpose(0, 1) 
        gamma = torch.log(self.gamma_exp)
        alpha_prev = gamma + emission 

        for t in range(1, m):
            temp = alpha_prev.unsqueeze(2) + torch.log(self.alpha_exp + 1e-12).unsqueeze(0)
            alpha_prev = torch.logsumexp(temp, dim=1)
            x_t = input_ids[:, t]
            emission = self.beta[:, x_t].transpose(0, 1)
            alpha_prev = alpha_prev + emission

        return alpha_prev
    
    def generate(self, input_ids, max_length=20, temperature=1.0):
        """
        Generate text sequences from the HMM model using autoregressive token generation.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (1, seq_len).
            max_length (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature to control diversity.

        Returns:
            List[int]: Generated token IDs.
        """
        device = self.alpha_exp.device
        generated_tokens = input_ids.tolist()[0]  # Assuming batch_size=1

        # Initialize α_prev using the input_ids
        with torch.no_grad():
            alpha_prev = self.compute_forward_probability(input_ids)  # Shape: (1, H)
            print(alpha_prev.shape)

        for _ in range(max_length):
            # Compute logits for the next token
            logits = self.compute_next_token_logits(alpha_prev)  # Shape: (1, V)
            logits = logits / temperature

            # Sample next token from the logits
            probs = torch.softmax(logits, dim=-1)  # Shape: (1, V)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Append the sampled token to the generated sequence
            generated_tokens.append(next_token)

            # Break if end-of-sequence token is generated
            if next_token == self.eos_token_id:
                break

            # Update α_prev with the newly generated token
            next_token_tensor = torch.tensor([next_token], device=device)  # Shape: (1,)
            alpha_prev = self.update_alpha_prev_with_token(alpha_prev, next_token_tensor)  # Shape: (1, H)

        return generated_tokens


    def compute_next_token_logits(self, alpha_prev):
        """
        Compute the logits for the next token (t = m+1) given the current forward probabilities α_m.

        Args:
            alpha_prev (torch.Tensor): Tensor of shape (batch_size, hidden_states) representing α_m.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, vocab_size) representing logits for the next token.
        """
        # Compute the predictive state distribution at time m+1
        # temp = α_prev(i) + log(a_{ij}) for all i, j
        temp = alpha_prev.unsqueeze(2) + torch.log(self.alpha_exp + 1e-12).unsqueeze(0)  # Shape: (batch_size, H, H)
        print(temp.shape)
        gamma_m1 = torch.logsumexp(temp, dim=1)  # Shape: (batch_size, H)

        # Compute logits = logsumexp over hidden states of [gamma_m1(j) + beta_j(x)]
        logits = torch.logsumexp(gamma_m1.unsqueeze(2) + self.beta.unsqueeze(0), dim=1)  # Shape: (batch_size, V)

        return logits  # Shape: (batch_size, vocab_size)

    def update_alpha_prev_with_token(self, alpha_prev, next_token):
        """
        Update the forward probabilities α_prev with the newly generated token.

        Args:
            alpha_prev (torch.Tensor): Tensor of shape (batch_size, hidden_states) representing α_m.
            next_token (torch.Tensor): Tensor of shape (batch_size,) containing the next token IDs.

        Returns:
            torch.Tensor: Updated α_{m+1} of shape (batch_size, hidden_states).
        """
        # Compute the predictive state distribution at time m+1
        temp = alpha_prev.unsqueeze(2) + torch.log(self.alpha_exp + 1e-12).unsqueeze(0)  # Shape: (batch_size, H, H)
        gamma_m1 = torch.logsumexp(temp, dim=1)  # Shape: (batch_size, H)

        # Emission probabilities for the next_token
        # self.beta: (H, V), next_token: (batch_size,)
        emission = self.beta[:, next_token].transpose(0, 1)  # Shape: (batch_size, H)

        # Update α_{m+1} = γ_{m+1} + log P(x_{m+1} | s_j)
        alpha_new = gamma_m1 + emission  # Shape: (batch_size, H)

        return alpha_new  # Shape: (batch_size, H)
    

    def loglikelihood(self, input_ids, batch_size):
        device = self.alpha_exp.device
        data_size, seq_len = input_ids.shape

        ll = torch.tensor([0.0], device=device)
        for batch_idx in range(0, data_size, batch_size):
            batch_size_ = min(batch_size, data_size - batch_idx)
            input_ids_batch = input_ids[batch_idx: batch_idx + batch_size_].to(device)
            probs_ = self.forward(input_ids_batch)
            ll += torch.sum(probs_[-1])

        return ll