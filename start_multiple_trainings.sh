#!/bin/bash
#training and evaluation

source bertriple_transformer/bin/activate
for sample in 800 700 600 500
do
    python add_pretraining_bert_trainer.py -train_file wikidata41.json -sample $sample -epoch 3 -template LAMA -query_type obj
done

source bertriple_LAMA/bin/activate 
cd LAMA
python scripts/run_experiments.py
