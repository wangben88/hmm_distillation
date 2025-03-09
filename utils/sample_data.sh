
CUDA_CORES=4,5,6,7
BASE_MODEL_NAME=gpt2-large
TOKENIZER_NAME=gpt2-large
PROMPTS_FILE=./prompts.json
TRAIN_SIZE=1000000
DEV_SIZE=10000
TEST_SIZE=0
BATCH_SIZE=1024
SEQUENCE_LENGTH=32

OUTPUT_FILE=../workspace/data/gpt2-large_train-${TRAIN_SIZE}_dev-${DEV_SIZE}_test-${TEST_SIZE}_seq-len-${SEQUENCE_LENGTH}.th

CUDA_VISIBLE_DEVICES=${CUDA_CORES} torchrun --standalone --nproc_per_node=gpu \
    sample_data.py \
    --model_name_or_path ${BASE_MODEL_NAME} \
    --tokenizer_name_or_path ${TOKENIZER_NAME} \
    --prompts_file ${PROMPTS_FILE} \
    --train_size ${TRAIN_SIZE} \
    --dev_size ${DEV_SIZE} \
    --test_size ${TEST_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --max_new_tokens ${SEQUENCE_LENGTH} \
    --output_file ${OUTPUT_FILE}