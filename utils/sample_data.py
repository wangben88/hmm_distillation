import os
import sys
import json
import argparse

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from tqdm import tqdm

import transformers
transformers.utils.logging.get_logger("transformers").setLevel(transformers.utils.logging.ERROR)


def pad_to_len(x, d, eos_token_id):
    if x.shape[1] < d:
        new_shape = x.shape[:1] + (d-x.shape[1],) + x.shape[2:]
        x = torch.cat((x, torch.full(new_shape, eos_token_id, dtype=x.dtype)), dim=1)
    return x


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--model_name_or_path', default='', type=str)
    arg_parser.add_argument('--tokenizer_name_or_path', default='', type=str)
    arg_parser.add_argument('--float16', action='store_true')
    arg_parser.add_argument('--bfloat16', action='store_true')

    arg_parser.add_argument('--prompts_file',  default='', type=str)
    arg_parser.add_argument('--train_size',  default=0, type=int)
    arg_parser.add_argument('--dev_size',  default=0, type=int)
    arg_parser.add_argument('--test_size',  default=0, type=int)
    arg_parser.add_argument('--batch_size', default=32, type=int)
    arg_parser.add_argument('--max_new_tokens', type=int, default=128)
    arg_parser.add_argument('--save_top_k_logits', type=int, default=0)
    
    arg_parser.add_argument('--top_k', type=int, default=0)
    arg_parser.add_argument('--top_p', type=float, default=1.0)    
    arg_parser.add_argument('--temperature', type=float, default=1.0)

    arg_parser.add_argument('--output_file',  default='', type=str)

    args = arg_parser.parse_args()

    assert (args.float16 == True and args.bfloat16 == True) == False
    # assert args.save_top_k_logits <= args.top_k

    dist.init_process_group('gloo')
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = f'cuda:{rank}'

    if rank == 0:        
        print(str(vars(args)))

    # load base model & tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    if args.float16:
        base_model.float16()
    if args.bfloat16:
        base_model.bfloat16()
    base_model.to(device)
    base_model.eval()

    tokenizer_name_or_path = args.tokenizer_name_or_path if args.tokenizer_name_or_path != '' else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token    

    # load input_data: a list of prompts for sampling data from the base model
    with open(args.prompts_file, 'r') as fin:
        prompts = json.load(fin)

    dataset = {
        'vocab_size': tokenizer.vocab_size,
        'eos_token_id': tokenizer.eos_token_id,
        'sequence_length': args.max_new_tokens,
    }

    for split_name, split_size in [('train', args.train_size), ('dev', args.dev_size), ('test', args.test_size)]:
        if split_size == 0:
            continue

        if rank == 0:
            print(f'generating {split_size} samples for {split_name} ...')

        ######################## DUPLICATE PROMPTS ########################
        num_samples_per_prompt = split_size // len(prompts) + 1
        prompts_ = []
        for prompt in prompts:
            prompts_.extend([prompt] * num_samples_per_prompt)
        prompts_ = prompts_[:split_size]
        prompts = prompts_
        ######################## DUPLICATE PROMPTS ########################

        num_samples_per_process = split_size // world_size + 1
        prompts_process = prompts[rank * num_samples_per_process: (rank+1) * num_samples_per_process]
        samples, top_k_logits = [], []
        for batch_idx in tqdm(range(0, len(prompts_process), args.batch_size)):
            inputs = tokenizer(
                prompts_process[batch_idx: batch_idx+args.batch_size],
                return_tensors='pt',
                padding=True)

            with torch.no_grad():
                generation_config = {
                    'do_sample': True,  
                    'top_k': args.top_k,                  
                    'top_p': args.top_p,                    
                    'temperature': args.temperature,
                }

                outputs = base_model.generate(
                    input_ids=inputs['input_ids'].to(device),
                    attention_mask=inputs['attention_mask'].to(device),
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=args.max_new_tokens,
                    output_hidden_states=False,
                    output_scores=(args.save_top_k_logits > 0),
                    return_dict_in_generate=True,
                    **generation_config,
                )

                samples_ = outputs.sequences[:, inputs['input_ids'].shape[1]:].cpu()
                samples_ = pad_to_len(samples_, args.max_new_tokens, tokenizer.eos_token_id)
                samples.append(samples_)

        samples = torch.cat(samples, dim=0)
        if rank == 0:
            samples_list = [torch.empty_like(samples, dtype=samples.dtype)
                    for idx in range(world_size)]
        else:
            samples_list = None

        dist.gather(samples, gather_list=samples_list)

        if rank == 0:
            samples = torch.cat(samples_list, dim=0)
            dataset[split_name] = samples[:split_size].clone()

    if rank == 0:
        torch.save(dataset, args.output_file)