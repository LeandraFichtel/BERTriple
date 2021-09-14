#!/bin/bash
#training and evaluation

for sample in all
do
    #python finetuning_and_evaluation.py -lm_name bert-base-cased -train_file AUTOPROMPT41 -sample $sample -epoch 3 -template LAMA -query_type obj -vocab common
    #python finetuning_and_evaluation.py -lm_name distilbert-base-cased -train_file AUTOPROMPT41 -sample $sample -epoch 3 -template LAMA -query_type obj -vocab common
    
    python evaluation.py -vocab different
    python finetuning_and_evaluation.py -lm_name roberta-base -train_file AUTOPROMPT41 -sample $sample -epoch 3 -template LAMA -query_type obj -vocab different -lama_uhn
    python finetuning_and_evaluation.py -lm_name facebook/bart-base -train_file AUTOPROMPT41 -sample $sample -epoch 3 -template LAMA -query_type obj -vocab different -lama_uhn
    python finetuning_and_evaluation.py -lm_name bert-large-cased -train_file AUTOPROMPT41 -sample $sample -epoch 3 -template LAMA -query_type obj -vocab common -lama_uhn
done