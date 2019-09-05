
# Prepare search Query
# python preIR.py ../data/dev.jsonl ../data/dev-labels.lst preir_dev.tsv
python preIR.py ../data/dev.jsonl preir_dev.tsv

# IR
python ir_from_aristo.py preir_dev.tsv

# RE-RANK AND CONSOLIDATION
python mergeIR.py preir_dev.tsv.out physical_merged_dev


### BEST MODEL WEIGHTED SUM physical_merged_dev.jsonl

# /scratch/kkpal/physical/noq/tiedws_128_9e-6_10865
# mcq_model bert-mcq-weighted-sum  --tie_weights_weighted_sum 


#Trainer
# python hf_trainer.py --training_data_path ../../data/physical_train_kb_noq_v2.jsonl --validation_data_path ../../data/physical_dev_kb_noq_v2.jsonl  --mcq_model bert-mcq-weighted-sum  --tie_weights_weighted_sum --bert_model /scratch/kkpal/phys_finetune_old/checkpoint-10865/ --output_dir  /scratch/kkpal/physical/noq/tiedws_128_9e-6_10865 --num_train_epochs 15 --train_batch_size 64  --do_eval --do_train --max_seq_length 128 --do_lower_case --gradient_accumulation_steps 16 --eval_freq 500 --learning_rate 9e-6  --warmup_steps 400 --weight_decay 0.001 --fp16