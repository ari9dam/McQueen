#!/usr/bin/env bash

set -e

pip install elasticsearch elasticsearch-dsl

/etc/init.d/elasticsearch start

sleep 15

cp scripts/Physical/* .

cat wikihow_single_sent.txt | python insert_text_to_elasticsearch.py

python preIR.py /data/physicaliqa.jsonl preir_dev.tsv

# IR
python ir_from_aristo.py preir_dev.tsv

/etc/init.d/elasticsearch  stop

# RE-RANK AND CONSOLIDATION
python mergeIR.py preir_dev.tsv.out physical_merged_dev


### BEST MODEL WEIGHTED SUM physical_merged_dev.jsonl

tar -zxvf trained_models/physical.tar 


python pytorch_transformers/models/hf_scorer.py --input_data_path physical_merged_dev.jsonl --model_dir scratch/kkpal/physical/noq/tiedws_128_9e-6_10865  --bert_model bert-large-uncased-whole-word-masking --mcq_model bert-mcq-weighted-sum  --tie_weights_weighted_sum --output_data_path .

mv predictions.txt /results/predictions.lst

