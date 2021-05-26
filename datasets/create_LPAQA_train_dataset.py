import os
import json
from transformers import BertTokenizer
import os.path as path

if __name__ == "__main__":
    vocab_file = open("../LAMA/pre-trained_language_models/common_vocab_cased.txt", "r")
    vocab = [line.rstrip() for line in vocab_file]
    
    #all 41 props
    #props = ['P19', 'P20', 'P279', 'P37', 'P413', 'P449', 'P47', 'P138', 'P364', 'P463', 'P101', 'P106', 'P527', 'P530', 'P176', 'P27', 'P407', 'P30', 'P178', 'P1376', 'P131', 'P1412', 'P108', 'P136', 'P17', 'P39', 'P264', 'P276', 'P937', 'P140', 'P1303', 'P127', 'P103', 'P190', 'P1001', 'P31', 'P495', 'P159', 'P36', 'P740', 'P361']
    #my 25 props
    props = ['P17', 'P19', 'P27', 'P31', 'P47', 'P106', 'P127', 'P131', 'P136', 'P138', 'P159', 'P190', 'P264', 'P276', 'P279', 'P361', 'P364', 'P407', 'P449', 'P463', 'P495', 'P527', 'P530', 'P740', 'P1303']
    
    if path.exists("/data/fichtel/BERTriple/training_datasets/LPAQAfiltered{}.json".format(len(props))):
        print("removed /data/fichtel/BERTriple/training_datasets/LPAQAfiltered{}.json".format(len(props)))
        os.remove("/data/fichtel/BERTriple/training_datasets/LPAQAfiltered{}.json".format(len(props)))
    LPAQA_train_filtered_file = open("/data/fichtel/BERTriple/training_datasets/LPAQAfiltered{}.json".format(len(props)), "w")
    
    LPAQA_train_filtered = []
    count_all = 0
    for filename in os.listdir("/data/fichtel/BERTriple/LPAQA/TREx_train"):
        prop = filename.replace(".jsonl", "")
        if prop in props:
            LPAQA_train = open("/data/fichtel/BERTriple/LPAQA/TREx_train/{}".format(filename), "r")
            count = 0
            for line in LPAQA_train:
                datapoint = json.loads(line)
                if datapoint["obj_label"] in vocab:
                    count = count + 1
                    count_all = count_all + 1
                    LPAQA_train_filtered.append({"subj": datapoint["sub_label"], "prop": prop, "obj": datapoint["obj_label"]})
            print("Added {} triples of property {}.".format(count, prop))
    print("Added {} triples for all {} properties.".format(count_all, len(props)))
    json.dump(LPAQA_train_filtered, LPAQA_train_filtered_file, indent=4)
                
        