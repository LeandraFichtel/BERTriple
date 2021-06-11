#!/bin/bash
#training and evaluation

source bertriple_transformer/bin/activate
for sample in 800 600 500 400 300 200 100 50
do
    python add_pretraining_bert_trainer.py -train_file AUTOPROMPT41 -sample $sample -epoch 3 -template LAMA -query_type obj
done

for sample in 800 600 500 400 300 200 100 50
do
    python add_pretraining_bert_trainer.py -train_file AUTOPROMPT41 -sample $sample -epoch 3 -template label -query_type obj
done