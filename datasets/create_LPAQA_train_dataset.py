import os
import json
import os.path as path
from transformers import BertTokenizer

def check_triple(label, tokenizer):
    tokenized_label = tokenizer.tokenize(label)
    #generel check, that the labels can be representend with the tokenizer
    if "[UNK]" not in tokenized_label:
        #only labels which are in vocab file (=one token)
        if len(tokenized_label) == 1:
            return True 
    return False

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    
    #all 41 props
    #props = ['P19', 'P20', 'P279', 'P37', 'P413', 'P449', 'P47', 'P138', 'P364', 'P463', 'P101', 'P106', 'P527', 'P530', 'P176', 'P27', 'P407', 'P30', 'P178', 'P1376', 'P131', 'P1412', 'P108', 'P136', 'P17', 'P39', 'P264', 'P276', 'P937', 'P140', 'P1303', 'P127', 'P103', 'P190', 'P1001', 'P31', 'P495', 'P159', 'P36', 'P740', 'P361']
    
    #my 25 props
    props = ['P17', 'P19', 'P27', 'P31', 'P47', 'P106', 'P127', 'P131', 'P136', 'P138', 'P159', 'P190', 'P264', 'P276', 'P279', 'P361', 'P364', 'P407', 'P449', 'P463', 'P495', 'P527', 'P530', 'P740', 'P1303']
    
    LPAQA_train_filtered_file = open("/data/kalo/akbc2021/training_datasets/LPAQAfiltered{}_all.json".format(len(props)), "w+")
    
    LPAQA_train_filtered = {}
    LPAQA_train_filtered["subj_queries"] = {}
    LPAQA_train_filtered["obj_queries"] = {}
    for filename in os.listdir("/data/fichtel/BERTriple/LPAQA/TREx_train"):
        prop = filename.replace(".jsonl", "")
        if prop in props:
            LPAQA_train = open("/data/fichtel/BERTriple/LPAQA/TREx_train/{}".format(filename), "r")
            count_subj_queries = 0
            count_obj_queries = 0
            for line in LPAQA_train:
                datapoint = json.loads(line)
                if check_triple(datapoint["obj_label"], tokenizer):
                    count_obj_queries = count_obj_queries + 1
                    if prop not in LPAQA_train_filtered["obj_queries"]:
                        LPAQA_train_filtered["obj_queries"][prop] = [{"subj": datapoint["sub_label"], "prop": prop, "obj": datapoint["obj_label"]}]
                    else:
                        LPAQA_train_filtered["obj_queries"][prop].append({"subj": datapoint["sub_label"], "prop": prop, "obj": datapoint["obj_label"]})
                if check_triple(datapoint["sub_label"], tokenizer):
                    count_subj_queries = count_subj_queries + 1
                    if prop not in LPAQA_train_filtered["subj_queries"]:
                        LPAQA_train_filtered["subj_queries"][prop] = [{"subj": datapoint["sub_label"], "prop": prop, "obj": datapoint["obj_label"]}]
                    else:
                        LPAQA_train_filtered["subj_queries"][prop].append({"subj": datapoint["sub_label"], "prop": prop, "obj": datapoint["obj_label"]})
            print("Added {} triples for object queries of property {}.".format(count_obj_queries, prop))
            print("Added {} triples for subject queries of property {}.".format(count_subj_queries, prop))
    print(len(LPAQA_train_filtered["obj_queries"]))
    json.dump(LPAQA_train_filtered, LPAQA_train_filtered_file, indent=4)
                
        