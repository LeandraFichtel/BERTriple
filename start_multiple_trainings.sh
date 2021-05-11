#!/bin/bash
#training and evaluation of LAMA and label templates
for sample in 1 5 10 20 30 50 100 200 300 400 500
do
    
    #python add_pretraining_bert_trainer.py -min_sample 500 -sample $sample -epoch 3 -template LAMA -string_token onetoken -perc_prop 100;
    #python add_pretraining_bert_trainer.py -min_sample 500 -sample $sample -epoch 3 -template label -string_token onetoken -perc_prop 100;
    #python fill_mask_bert.py -queries -min_sample 500 -sample $sample -epoch 3 -template LAMA -string_token onetoken -perc_prop 100;
    #python fill_mask_bert.py -queries -min_sample 500 -sample $sample -epoch 3 -template label -string_token onetoken -perc_prop 100;
    #python fill_mask_bert.py -train -min_sample 500 -sample $sample -epoch 3 -template LAMA -string_token onetoken -perc_prop 100;
    #python fill_mask_bert.py -train -min_sample 500 -sample $sample -epoch 3 -template label -string_token onetoken -perc_prop 100;
    python evaluation.py -queries -min_sample 500 -sample $sample -epoch 3 -template LAMA -string_token onetoken -perc_prop 100;
    python evaluation.py -queries -min_sample 500 -sample $sample -epoch 3 -template label -string_token onetoken -perc_prop 100;
    #python evaluation.py -train -min_sample 500 -sample $sample -epoch 3 -template LAMA -string_token onetoken -perc_prop 100;
    #python evaluation.py -train -min_sample 500 -sample $sample -epoch 3 -template label -string_token onetoken -perc_prop 100;

    if [ $sample == 100 ] || [ $sample == 500 ]; then
        for percent in 90 80 70 60 50 40 30 20 10
        do
            for i in 0 1 2 3 4
            do
                #python add_pretraining_bert_trainer.py -min_sample 500 -sample $sample -epoch 3 -template label -string_token onetoken -perc_prop "${percent}-${i}";
                #python fill_mask_bert.py -queries -min_sample 500 -sample $sample -epoch 3 -template label -string_token onetoken -perc_prop "${percent}-${i}";
                python evaluation.py -queries -min_sample 500 -sample $sample -epoch 3 -template label -string_token onetoken -perc_prop "${percent}-${i}";
            done
        done
    fi
done

#evaluation of auto templates
#for sample in 1 5 10 20 30 50 100 200 300 400 500
#do
#    python fill_mask_bert.py -queries -min_sample 500 -sample $sample -epoch 3 -template auto -string_token onetoken -perc_prop 100;
#    python evaluation.py -queries -min_sample 500 -sample $sample -epoch 3 -template auto -string_token onetoken -perc_prop 100;
#done
#
#./create_plots_tables.sh