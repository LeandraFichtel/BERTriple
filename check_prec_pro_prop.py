import pandas as pd
import json
import numpy as np


all_props = ['P19', 'P20', 'P279', 'P37', 'P413', 'P449', 'P47', 'P138', 'P364', 'P463', 'P101', 'P106', 'P527', 'P530', 'P176', 'P27', 'P407', 'P30', 'P178', 'P1376', 'P131', 'P1412', 'P108', 'P136', 'P17', 'P39', 'P264', 'P276', 'P937', 'P140', 'P1303', 'P127', 'P103', 'P190', 'P1001', 'P31', 'P495', 'P159', 'P36', 'P740', 'P361']
my_props = ['P17', 'P19', 'P27', 'P31', 'P47', 'P106', 'P127', 'P131', 'P136', 'P138', 'P159', 'P190', 'P264', 'P276', 'P279', 'P361', 'P364', 'P407', 'P449', 'P463', 'P495', 'P527', 'P530', 'P740', 'P1303']

def calculate_avg(dict_results, props=all_props):
    print("Average over {} props".format(len(props)))
    avg = 0
    for prop in dict_results:
       if prop in props:
           avg = avg + dict_results[prop]
    return avg/len(props)

def print_prec_per_prop(list_dicts_results, props=all_props):
    print("prop BBCF_new BBCF_old")
    for prop in props:
        line = prop + " "
        for dict in list_dicts_results:
            line = line + str(dict[prop]) + " "
        print(line)
    

dictio_eval_results = json.load(open("/home/fichtel/projektarbeit/results/queries/eval_queries_100_onetoken_3_LAMA_500_500.json", "r"))
query_type = "object"
dict_prop_prec_finetuned_old_25 = {}
precision_per_prop_finetuned = dictio_eval_results[query_type]["precision@1"]["per_prop"]["finetuned"]
for i, prop in enumerate(all_props):
    if prop in precision_per_prop_finetuned:
        dict_prop_prec_finetuned_old_25[prop] = precision_per_prop_finetuned[prop]
    else:
        dict_prop_prec_finetuned_old_25[prop] = "X"
print("finetuned_old_25", calculate_avg(dict_prop_prec_finetuned_old_25, my_props))

dict_prop_prec_finetuned_25 =  dict((pd.read_csv('results/BBCF_LPAQA25_obj_3_LAMA.csv', sep = ',', header = None)).values)
print("BBCF_LPAQA25_obj_3_LAMA.csv", calculate_avg(dict_prop_prec_finetuned_25, my_props))

dict_prop_prec_finetuned_41 =  dict((pd.read_csv('results/BBCF_LPAQA41_obj_3_LAMA.csv', sep = ',', header = None)).values)
print("BBCF_LPAQA41_obj_3_LAMA.csv", calculate_avg(dict_prop_prec_finetuned_41))

dict_prop_prec_finetuned_41 =  dict((pd.read_csv('results/BBCF_wikidata41_obj_3_LAMA.csv', sep = ',', header = None)).values)
print("BBCF_wikidata41_obj_3_LAMA", calculate_avg(dict_prop_prec_finetuned_41))

dict_prop_prec_normal =  dict((pd.read_csv('results/bert_base.csv', sep = ',', header = None)).values)
print("bert_base.csv", calculate_avg(dict_prop_prec_normal))

#print_prec_per_prop([dict_prop_prec_finetuned, dict_prop_prec_finetuned_old])
