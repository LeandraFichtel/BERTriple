#!/bin/bash
#evaluation of baselines languge models
python evaluation.py -lm_names bert-base-cased bert-large-cased distilbert-base-cased -vocab common
python evaluation.py -lm_names roberta-base facebook/bart-base -vocab different

#experiment1
python finetuning_and_evaluation.py -lm_name bert-base-cased -train_file AUTOPROMPT41 -sample all -epoch 3 -template LAMA -query_type obj -vocab common -lama_uhn

#experiment2
for sample in 900 800 700 600 500 400 300 200 100 50 30 10 1
do
    python finetuning_and_evaluation.py -lm_name bert-base-cased -train_file AUTOPROMPT41 -sample $sample -epoch 3 -template LAMA -query_type obj -vocab common
    python finetuning_and_evaluation.py -lm_name bert-base-cased -train_file AUTOPROMPT41 -sample $sample -epoch 3 -template label -query_type obj -vocab common
done
python finetuning_and_evaluation.py -lm_name bert-base-cased -train_file AUTOPROMPT41 -sample all -epoch 3 -template label -query_type obj -vocab common
python create_tables_for_eval.py -train_file AUTOPROMPT41 -epoch 3 -query_type obj -precision_3_runs

#experiment3
python finetuning_and_evaluation.py -lm_name bert-base-cased -train_file AUTOPROMPT41 -sample all -epoch 3 -template label -query_type obj -vocab common -transfer_learning
python create_tables_for_eval.py -train_file AUTOPROMPT41 -sample all -epoch 3 -template LAMA -query_type obj -transfer_learning

#experiment4 (appendix)
python finetuning_and_evaluation.py -lm_name bert-large-cased -train_file AUTOPROMPT41 -sample all -epoch 3 -template LAMA -query_type obj -vocab common -lama_uhn
python finetuning_and_evaluation.py -lm_name distilbert-base-cased -train_file AUTOPROMPT41 -sample all -epoch 3 -template LAMA -query_type obj -vocab common -lama_uhn
python finetuning_and_evaluation.py -lm_name roberta-base -train_file AUTOPROMPT41 -sample all -epoch 3 -template LAMA -query_type obj -vocab different -lama_uhn
python finetuning_and_evaluation.py -lm_name facebook/bart-base -train_file AUTOPROMPT41 -sample all -epoch 3 -template LAMA -query_type obj -vocab different -lama_uhn