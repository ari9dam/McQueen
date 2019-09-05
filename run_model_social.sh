 #!/usr/bin/env bash

set -e

pip install elasticsearch elasticsearch-dsl

/etc/init.d/elasticsearch start

sleep 15

cp scripts/Social/* .

cat atomic_knowledge_sentences.txt | python insert_text_to_elasticsearch_lemmatized.py

python preprare_dataset.py val.jsonl

# IR
python ir_from_aristo_lemmatized.py dev_ir_lemma.tsv

/etc/init.d/elasticsearch  stop

# RE-RANK AND CONSOLIDATION
python merge_ir.py typed

python clean_dataset.py dev_typed.jsonl

### BEST MODEL WEIGHTED SUM physical_merged_dev.jsonl

tar -zxvf trained_models/social2.tar

python pytorch_transformers/models/hf_scorer.py --input_data_path dev_typed.jsonl   --eval_batch_size 1 --model_dir scratch/pbanerj6/serdir_softmaxed_weighted_sum_tied_5e6_001_social  --bert_model bert-large-uncased-whole-word-masking --mcq_model bert-mcq-weighted-sum  --tie_weights_weighted_sum --output_data_path .

python fix_preds.py

mv predictions2.txt /results/predictions.lst



