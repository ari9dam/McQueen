#!/usr/bin/env bash

set -e

pip install elasticsearch elasticsearch-dsl


/etc/init.d/elasticsearch start

sleep 15

cp scripts/Abductive/* .

cat ROCaug.txt | python insert_text_to_elasticsearch.py

python ai2test.py /data/anli.jsonl

python ir_from_aristo.py devfinal.tsv

/etc/init.d/elasticsearch  stop

python merge_ir.py mnli_simple

# scratch/srmishr1/Sunpower/2015_swt_5e6_001_11

tar -zxvf trained_models/anli.tar


python pytorch_transformers/models/hf_scorer.py --input_data_path  devf.jsonl   --eval_batch_size 1 --model_dir scratch/srmishr1/Sunpower/2015_swt_5e6_001_11 --bert_model bert-large-uncased-whole-word-masking --mcq_model bert-mcq-weighted-sum --tie_weights_weighted_sum --output_data_path .

python fix_preds.py

mv predictions2.txt /results/predictions.lst
#python random_baseline.py --input-file /data/anli.jsonl --output-file /results/predictions.lst