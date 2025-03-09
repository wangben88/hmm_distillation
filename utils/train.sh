
# MODEL='hmm'
# HIDDEN_SIZE=128
# DATASET='gpt2-large_seq-len-32'
# DATASET_PATH='../workspace/data/gpt2-large_seq-len-32_train-1000000_dev-10000_test-0.th'

# for HIDDEN_SIZE in 16384
# do
#     for ID in 1 2
#     do
#         MODEL_ID="${MODEL}_${DATASET}_${HIDDEN_SIZE}_${ID}"
#         SAVE_MODEL_PATH="../workspace/models/${MODEL_ID}"
#         mkdir -p ${SAVE_MODEL_PATH}

#         CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=gpu \
#             train.py \
#             --init_model_path ../workspace/models/hmm_gpt2-large_seq-len-32_16384/checkpoint-0 \
#             --checkpoint 0 \
#             --dataset_path ${DATASET_PATH} \
#             --model ${MODEL} \
#             --hidden_size ${HIDDEN_SIZE} \
#             --batch_size 2048 \
#             --grad_accum_iters 2 \
#             --em_schedule "epoch,40,1.0,0.0,linear" \
#             --save_model_path ${SAVE_MODEL_PATH} \
#             --log_file ../logs/${MODEL_ID}_log.txt \
#             --eval_per_steps 100
#     done
# done


MODEL='hmm'
HIDDEN_SIZE=32768
DATASET='gpt2-large_seq-len-32'
DATASET_PATH='../workspace/data/gpt2-large_seq-len-32_train-1000000_dev-10000_test-0.th'

MODEL_ID="${MODEL}_${DATASET}_${HIDDEN_SIZE}"
SAVE_MODEL_PATH="../workspace/models/${MODEL_ID}"
mkdir -p ${SAVE_MODEL_PATH}

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=gpu \
    train.py \
    --init_model_path ../workspace/models/hmm_gpt2-large_seq-len-32_32768/checkpoint-0 \
    --checkpoint 0 \
    --dataset_path ${DATASET_PATH} \
    --model ${MODEL} \
    --hidden_size ${HIDDEN_SIZE} \
    --batch_size 4096 \
    --grad_accum_iters 1 \
    --em_schedule "epoch,50,1.0,0.0,linear" \
    --save_model_path ${SAVE_MODEL_PATH} \
    --log_file ../logs/${MODEL_ID}_log.txt \
    --eval_per_steps 100