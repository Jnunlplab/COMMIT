function finetune_codet5p () {
  prompt_with=$1
  mkdir -p ./saved_models_${prompt_with}/cache_data
  python run_gen.py    \
      --do_train \
      --do_eval \
      --do_eval_bleu \
      --do_test  \
      --save_last_checkpoints \
      --always_save_model   \
      --task summarize_prompt \
      --sub_task java \
      --model_type codet5 \
      --data_num -1    \
      --num_train_epochs 10 \
      --warmup_steps 1000 \
      --learning_rate 5e-5 \
      --patience 2   \
      --tokenizer_name=../../checkpoint/codet5p-220m \
      --tokenizer_path=../../checkpoint/codet5p-220m   \
      --model_name_or_path=../../checkpoint/codet5p-220m \
      --output_dir saved_models_${prompt_with}/  \
      --summary_dir tensorboard   \
      --data_dir ../../datasets  \
      --cache_path saved_models_${prompt_with}/cache_data \
      --res_dir saved_models_${prompt_with}/prediction \
      --res_fn saved_models_${prompt_with}/summarize_codet5_base.txt   \
      --train_batch_size 16 \
      --eval_batch_size 16 \
      --max_source_length 512 \
      --max_target_length 128   \
      --seed 2024 \
      --example_type cpg-3 \
      --soft_or_prefix soft \
      --prompt_with ${prompt_with} \
      2>&1 | tee saved_models_${prompt_with}/train.log
}

finetune_codet5p v1
finetune_codet5p v2
finetune_codet5p v3

