import os
import json
from transformers import BertTokenizer
import os.path as path

def check_triple(obj_label, tokenizer):
    tokenized_obj = tokenizer.tokenize(obj_label)
    #generel check, that the labels can be representend with the tokenizer
    if "[UNK]" not in tokenized_obj:
        #only labels which are in vocab file (=one token)
        if len(tokenized_obj) == 1:
            #triple can be used only as object query
            return True 
    return False

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    
    if path.exists("/data/fichtel/BERTriple/training_datasets/LPAQA_filtered.json"):
        print("removed /data/fichtel/BERTriple/training_datasets/LPAQA_filtered.json")
        os.remove("/data/fichtel/BERTriple/training_datasets/LPAQA_filtered.json")
    LPAQA_train_filtered_file = open("/data/fichtel/BERTriple/training_datasets/LPAQAfiltered.json", "w")

    LPAQA_train_filtered = []
    for filename in os.listdir("/data/fichtel/BERTriple/LPAQA/TREx_train"):
        prop = filename.replace(".jsonl", "")
        LPAQA_train = open("/data/fichtel/BERTriple/LPAQA/TREx_train/{}".format(filename), "r")
        count = 0
        for line in LPAQA_train:
            datapoint = json.loads(line)
            if check_triple(datapoint["obj_label"], tokenizer):
                count = count + 1
                LPAQA_train_filtered.append({"subj": datapoint["sub_label"], "prop": prop, "obj": datapoint["obj_label"]})
        print("Added {} triples of property {}.".format(count, prop))
    json.dump(LPAQA_train_filtered, LPAQA_train_filtered_file, indent=4)
                
        