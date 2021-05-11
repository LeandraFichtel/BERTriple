#create tables and figures
#python create_tables_for_eval.py -precision -epoch 3 -min_sample 500 -sample all -template all -string_token onetoken -perc_prop 100
python create_tables_for_eval.py -probability -epoch 3 -min_sample 500 -sample 100 -template label -string_token onetoken -perc_prop 100
#python create_tables_for_eval.py -seen_in_training -epoch 3 -min_sample 500 -sample 500 -template label -string_token onetoken -perc_prop 100
#python create_tables_for_eval.py -training_data_memorization -epoch 3 -min_sample 500 -sample 500 -template label -string_token onetoken -perc_prop 100
#python create_tables_for_eval.py -template_eval -epoch 3 -min_sample 500 -sample 200 -template label -string_token onetoken -perc_prop 100
#python create_tables_for_eval.py -random_props -epoch 3 -min_sample 500 -sample 500 -template label -string_token onetoken -perc_prop all
#python create_tables_for_eval.py -per_props_template -epoch 3 -min_sample 500 -sample 500 -template all -string_token onetoken -perc_prop 100
#python create_tables_for_eval.py -per_props_random_props -epoch 3 -min_sample 500 -sample 500 -template label -string_token onetoken -perc_prop all
#python create_tables_for_eval.py -ece -epoch 3 -min_sample 500 -sample all -template label -string_token onetoken -perc_prop 100