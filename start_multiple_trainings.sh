#!/bin/bash
#training and evaluation

for sample in all
do
    python evaluation -template LAMA
    python add_pretraining_bert_trainer.py -lm_name bert-base-cased -train_file AUTOPROMPT41 -sample $sample -epoch 3 -template LAMA -query_type obj
    python add_pretraining_bert_trainer.py -lm_name distilbert-base-cased -train_file AUTOPROMPT41 -sample $sample -epoch 3 -template LAMA -query_type obj
    #python add_pretraining_bert_trainer.py -train_file AUTOPROMPT41 -sample $sample -epoch 3 -template label -query_type obj
done