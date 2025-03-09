
python hmm_multiplication.py \
    --checkpoint_paths \
        ../workspace/models/hmm_gpt2-large_seq-len-32_128_1/checkpoint-2440 \
        ../workspace/models/hmm_gpt2-large_seq-len-32_256_1/checkpoint-2440 \
    --output_model hmm \
    --output_path \
        ../workspace/models/hmm_gpt2-large_seq-len-32_32768/checkpoint-0