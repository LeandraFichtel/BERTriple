#!/bin/bash
#training and evaluation

for sample in all
do
    python evaluation.py -template LAMA
    python finetuning_and_evaluation.py -lm_name bert-base-cased -train_file AUTOPROMPT41 -sample $sample -epoch 3 -template LAMA -query_type obj
    python finetuning_and_evaluation.py -lm_name distilbert-base-cased -train_file AUTOPROMPT41 -sample $sample -epoch 3 -template LAMA -query_type obj
    #python finetuning_and_evaluation.py -train_file AUTOPROMPT41 -sample $sample -epoch 3 -template label -query_type obj
done