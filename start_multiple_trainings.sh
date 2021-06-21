#!/bin/bash
#training and evaluation

for sample in all 900 800 700 600 500 400 300 200 100 50 30 10 1
do
    python add_pretraining_bert_trainer.py -train_file AUTOPROMPT41 -sample $sample -epoch 3 -template LAMA -query_type obj
    python add_pretraining_bert_trainer.py -train_file AUTOPROMPT41 -sample $sample -epoch 3 -template label -query_type obj
done