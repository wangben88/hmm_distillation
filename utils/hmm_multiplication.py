import torch
import argparse
import functools
from hmm.em import HMM
from hmm.em import MonarchHMM


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--checkpoint_paths', nargs='+', default=[])
    arg_parser.add_argument('--output_model', default='', type=str)
    arg_parser.add_argument('--output_path', default='', type=str)

    args = arg_parser.parse_args()

    hmm_models = [HMM.from_pretrained(checkpoint_path)
        for checkpoint_path in args.checkpoint_paths]

    block_sizes = [model.hidden_size for model in hmm_models]
    hidden_size = functools.reduce(lambda x,y: x * y, block_sizes)
    vocab_size = hmm_models[0].vocab_size
    eos_token_id = hmm_models[0].eos_token_id
    sep_token_id = hmm_models[0].sep_token_id

    if args.output_model == 'hmm':
        output_model = HMM(hidden_size, vocab_size, eos_token_id, sep_token_id)
    elif args.output_model == 'monarch-hmm':
        block_sizes = tuple([model.hidden_size for model in hmm_models])
        output_model = MonarchHMM(hidden_size, block_sizes, vocab_size, eos_token_id, sep_token_id)
    else:
        raise NotImplementedError


    # multiply input distributions (beta)
    betas = []
    for i, hmm_model in enumerate(hmm_models):
        beta_shape = (1,) * i + (hmm_model.beta.shape[0],) + (1,) * (len(block_sizes) - 1 - i) + (hmm_model.beta.shape[-1],)
        betas.append(hmm_model.beta.view(beta_shape))
    beta = betas[0]
    for beta_i in betas[1:]:
        beta = beta + beta_i
    beta = beta.view(-1, beta.shape[-1]).contiguous()
    beta = torch.log_softmax(beta, dim=-1)
    output_model.beta.data.copy_(beta)

    # multiply initial probabilities (gamma)
    gammas = []
    for i, hmm_model in enumerate(hmm_models):
        gamma_shape = (1,) * i + (hmm_model.gamma_exp.shape[1],) + (1,) * (len(block_sizes) - 1 - i)
        gammas.append(torch.log(hmm_model.gamma_exp.view(gamma_shape)))
    gamma = gammas[0]
    for gamma_i in gammas[1:]:
        gamma = gamma + gamma_i
    gamma = gamma.view(1, -1).contiguous()
    gamma_exp = torch.softmax(gamma, dim=-1)
    output_model.gamma_exp.data.copy_(gamma_exp)

    # multiply transition probabilities (alpha)
    if args.output_model == 'hmm':
        alphas = []
        for i, hmm_model in enumerate(hmm_models):
            alpha_shape = (1,) * i + (hmm_model.alpha_exp.shape[0],) + (1,) * (len(block_sizes) - 1 - i)
            alpha_shape = alpha_shape + alpha_shape
            alphas.append(torch.log(hmm_model.alpha_exp.view(alpha_shape)))

        alpha = alphas[0]
        for alpha_i in alphas[1:]:
            alpha = alpha + alpha_i
        alpha = alpha.view(hidden_size, hidden_size).contiguous()
        alpha_exp = torch.softmax(alpha, dim=-1)

        output_model.alpha_exp.data.copy_(alpha_exp)

    elif args.output_model == 'monarch-hmm':
        for i, hmm_model in enumerate(hmm_models):
            output_model.alpha_exp.weights_exp[i].data.copy_(hmm_model.alpha_exp[None, :, :])

    else:
        raise NotImplementedError

    print(f'saving to {args.output_path}')
    output_model.save_pretrained(args.output_path)