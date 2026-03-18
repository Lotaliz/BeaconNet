# !/bin/bash

lm_eval --model hf \
    --model_args pretrained=./models/pruned/llama3.1-8B-Instruct-dpo-wanda-0.6,tokenizer=./models/pruned/llama3.1-8B-Instruct-dpo-wanda-0.6 \
    --tasks toxigen,mmlu,gsm8k \
    --apply_chat_template \
    --device cuda:0 \
    --batch_size 8 \
    --output_path ./data/utility