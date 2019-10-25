# <img src="mcqueen.jpg" width="60"> Introduction
McQueen is a MCQ solving library that allows researchers and developers to train and test several existing MCQ solvers on custom textual mcq datasets. Related Paper: [Exploring ways to incorporate additional knowledge to improve Natural
Language Commonsense Question Answering](https://arxiv.org/pdf/1909.08855.pdf)
 

## Package Overview
<!--
### AllenNLP Modules
<table>
<tr>
    <td><b> bert_mcq </b></td>
    <td> contains a simple BERT based MCQ solver that score each choice string using BERT w.r.t. a premise string.</td>
</tr>
<tr>
    <td><b> bert_mcq_parallel </b></td>
    <td> functionality for simple BERT based MCQ solver that scores each choice string using BERT w.r.t. an array premise strings and takes the maximum score as the confidence value for the choice.</td>
</tr>
<tr>
    <td><b> esim_mcq </b></td>
    <td> functionality for simple ESIM based MCQ solver that scores each option w.r.t a premise string using ESIM </td>
</tr>
<tr>
    <td><b> esim_mcq_parallel </b></td>
    <td> functionality for simple ESIM based MCQ solver that scores each choice string using ESIM w.r.t. an array premise strings and takes the maximum score as the confidence value for the choice. </td>
</tr>
<tr>
    <td><b> decatt_mcq </b></td>
    <td> functionality for simple DecAatt based MCQ solver that scores each option w.r.t a premise string using DecAtt </td>
</tr>
<tr>
    <td><b> decatt_mcq_parallel </b></td>
    <td> functionality for simple DecAtt based MCQ solver that scores each choice string using DecAtt w.r.t. an array premise strings and takes the maximum score as the confidence value for the choice. </td>
</tr>
<tr>
    <td><b> mac_bert </b></td>
    <td> implements the MacBERT model described in the paper </td>
</tr>
<tr>
    <td><b> mac_bert_graphical </b></td>
    <td> implements the MacBERTGraphical model described in the paper </td>
</tr>
<tr>
    <td><b> mac_bert_pairwise </b></td>
    <td> implements the MacBERTPairwise model described in the paper </td>
</tr>
<tr>
    <td><b> mac_bert_no_coverage </b></td>
    <td> implements the MacBERTWithoutCoverage model described in the paper </td>
</tr>
<tr>
    <td><b> mac_seq </b></td>
    <td> implements the MacSeq model described in the paper </td>
</tr>
</table>
-->
### HuggingFace Modules
<table>
<tr>
    <td><b> bert-mcq-concat </b></td>
    <td> contains a simple BERT based MCQ solver that score each choice string using BERT w.r.t. a premise string. File: pytorch_transformers/models/hf_bert_mcq_concat.py</td>
</tr>
<tr>
    <td><b> bert-mcq-parallel-max </b></td>
    <td> functionality for simple BERT based MCQ solver that scores each choice string using BERT w.r.t. an array premise strings and takes the maximum score as the confidence value for the choice. File: pytorch_transformers/models/hf_bert_mcq_parallel.py</td>
</tr>
<tr>
    <td><b> bert-mcq-weighted-sum </b></td>
    <td> functionality for simple BERT based MCQ solver that for each choice string run bert choice string, premise string pair and perform a weighted sum over the pooled cls token vectors to score the choice. File: pytorch_transformers/models/hf_bert_weighted_sum.py</td>
</tr>
<tr>
    <td><b> bert-mac </b></td>
    <td> coming soon. </td>
</tr>
</table>

# Setup
To run the huggingface models 
1. conda create --name env-name python=3.6
2. source activate env-name
3. pip install pytorch-transformers
4. git checkout the repo in your agave host
5. run launch gpu to get a host with multiple gpus 
    
 # Training
 The training data should be in the format that is mentioned in the [doc](https://docs.google.com/document/d/1asswWYl_qG3sA97IMrv25k46Ueu4ujysDls6vIUX6Jk/)
 
 There are three models as of now, namely bert-mcq-concat, bert-mcq-parallel-max, bert-mcq-weighted-average
 
 Here is a sample run command to train bert-mcq-concat model with bert-large-whole-word-uncased :
 
1. Running bert-mcq-concat
 ```
  nohup python hf_trainer.py --training_data_path mcq_abductive_train.jsonl --validation_data_path mcq_abductive_dev.jsonl  --mcq_model bert-mcq-concat --bert_model bert-large-uncased-whole-word-masking --output_dir ./serdir_bertlgww_concat_2e5_abd --num_train_epochs 4 --train_batch_size 64  --do_eval --do_train --max_seq_length 68 --do_lower_case --gradient_accumulation_steps 1  --learning_rate 2e-6 --weight_decay 0.009  --eval_freq 1000 --warmup_steps 250 &> bertlgww_concat_2e5_009_abd.out
```
2. Running bert-mcq-parallel-max
```
nohup python hf_trainer.py --training_data_path mcq_abductive_train.jsonl --validation_data_path mcq_abductive_dev.jsonl  --mcq_model bert-mcq-parellel-max --bert_model bert-large-uncased-whole-word-masking --output_dir ./serdir_bertlgww_concat_2e5_abd --num_train_epochs 4 --train_batch_size 64  --do_eval --do_train --max_seq_length 68 --do_lower_case --gradient_accumulation_steps 1  --learning_rate 2e-6 --weight_decay 0.009  --eval_freq 1000 --warmup_steps 250
```
3. Running simple sum model
```
nohup python hf_trainer.py --training_data_path mcq_sc_sim_train.jsonl --validation_data_path mcq_sc_sim_dev.jsonl  --mcq_model bert-mcq-simple-sum  --bert_model bert-large-uncased-whole-word-masking --output_dir ./serdir_bertlgww_simple_sum_4e6_001_social --num_train_epochs 4 --train_batch_size 16  --do_eval --do_train --max_seq_length 68 --do_lower_case --gradient_accumulation_steps 1 --eval_freq 500 --learning_rate 4e-6  --warmup_steps 400 --weight_decay 0.001
 ```
 
4. Running Weighted sum model with ( with tied weights )
```
nohup python hf_trainer.py --training_data_path mcq_sc_sim_train.jsonl --validation_data_path mcq_sc_sim_dev.jsonl  --mcq_model bert-mcq-weighted-sum  --tie_weights_weighted_sum --bert_model bert-large-uncased-whole-word-masking --output_dir ./serdir_bertlgww_simple_sum_4e6_001_social --num_train_epochs 4 --train_batch_size 16  --do_eval --do_train --max_seq_length 68 --do_lower_case --gradient_accumulation_steps 1 --eval_freq 500 --learning_rate 4e-6  --warmup_steps 400 --weight_decay 0.001
 ```
 
5. Running weighted sum without tied weights
```
nohup python hf_trainer.py --training_data_path mcq_sc_sim_train.jsonl --validation_data_path mcq_sc_sim_dev.jsonl  --mcq_model bert-mcq-weighted-sum  --bert_model bert-large-uncased-whole-word-masking --output_dir ./serdir_bertlgww_simple_sum_4e6_001_social --num_train_epochs 4 --train_batch_size 16  --do_eval --do_train --max_seq_length 68 --do_lower_case --gradient_accumulation_steps 1 --eval_freq 500 --learning_rate 4e-6  --warmup_steps 400 --weight_decay 0.001
 ```


Please see the file hf_trainer.py to read the meaning and the default value of the parameters.
