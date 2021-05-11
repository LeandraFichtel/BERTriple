import os
import json
from transformers import BertTokenizer

def check_tokenizer(subj_label, obj_label, tokenizer):
    tokenized_subj = tokenizer.tokenize(subj_label)
    tokenized_obj = tokenizer.tokenize(obj_label)
    #generel check, that the labels can be representend with the tokenizer
    if "[UNK]" not in tokenized_subj and  "[UNK]" not in tokenized_obj:
        #only labels which are in vocab file (=one token) and consist of only one word are accepted
        if len(tokenized_subj) == 1 and len(tokenized_obj) == 1:
            #triple can be used as subject and object query
            return "subjobj"
        elif len(tokenized_subj) == 1 and len(tokenized_obj) > 1:
            #triple can be used only as subject query
            return None
        elif len(tokenized_subj) > 1 and len(tokenized_obj) == 1:
            #triple can be used only as object query
            return "obj"    
    #print(subj_label, tokenized_subj, obj_label, tokenized_obj)
    return None

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    
    for filename in os.listdir("/data/fichtel/BERTriple/LPAQA/TREx_train"):
        prop = filename.replace(".jsonl", "")
        print(prop)
        prop_train = open("/data/fichtel/BERTriple/LPAQA/TREx_train/{}".format(filename), "r")
        count_all = 0
        count = 0
        for line in prop_train:
            count_all = count_all + 1
            datapoint = json.loads(line)
            if check_tokenizer(datapoint["sub_label"], datapoint["obj_label"], tokenizer):
                count = count + 1
        print(count, count_all)
        