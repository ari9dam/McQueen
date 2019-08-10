# <img src="mcqueen.jpg" width="60"> Introduction
McQueen is a MCQ solving library that allows researchers and developers to train and test several existing MCQ solvers on custom textual mcq datasets. McQueen is written in **[AllenNLP](https://github.com/allenai/allennlp)**

## Package Overview

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

# Setup
