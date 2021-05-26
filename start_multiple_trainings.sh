#!/bin/bash
#training and evaluation of LAMA and label templates

python add_pretraining_bert_trainer.py -train_file LPAQAfiltered25.json -sample 500 -epoch 3 -template LAMA -string_token onetoken -perc_prop 100;
python add_pretraining_bert_trainer.py -train_file LPAQAfiltered41.json -sample 500 -epoch 3 -template LAMA -string_token onetoken -perc_prop 100;
cd LAMA
python scripts/run_experiments.py
    