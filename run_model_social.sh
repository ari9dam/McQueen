#!/usr/bin/env bash

set -e

pip install elasticsearch elasticsearch-dsl

/etc/init.d/elasticsearch start

sleep 15

cp scripts/Social/* .

cat atomic_knowledge_sentences.txt | python insert_text_to_elasticsearch_lemmatized.py

python preprare_dataset.py /data/socialiqa.jsonl

# IR
python ir_from_aristo_lemmatized.py dev_ir_lemma.tsv

/etc/init.d/elasticsearch  stop

# RE-RANK AND CONSOLIDATION
python merge_ir.py typed

python clean_dataset.py dev_typed.jsonl

### BEST MODEL WEIGHTED SUM physical_merged_dev.jsonl

tar -zxvf trained_models/social_mac.tar

python pytorch_transformers/models/hf_scorer.py --input_data_path dev_typed.jsonl   --max_number_premises 6 --max_seq_length 72 --eval_batch_size 1 --model_dir scratch/pbanerj6/social/social_mcq_mac_bertir_ranked_6_tw_50  --bert_model bert-large-uncased-whole-word-masking --mcq_model bert-mcq-mac  --output_data_path .

python fix_preds.py

mv predictions2.txt /results/predictions.lst



