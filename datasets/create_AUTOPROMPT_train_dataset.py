import os
import json

def load_trex_file(dataset, trex_data, vocab):
    counter = 0
    for triple_dict in trex_data:
        p_qid = triple_dict["predicate_id"]
        s_label = triple_dict["sub_label"]
        o_label = triple_dict["obj_label"]
        if o_label not in vocab:
            print("WARNING: object label ({}) is not in bert-base-cased vocab = is not one-token, but added triple anyway".format(o_label))
        triple = {"subj": s_label, "prop": p_qid, "obj": o_label}
        counter += 1
        if p_qid not in dataset:
            dataset[p_qid] = [triple]
        else:
            dataset[p_qid].append(triple)
    print("Added {} triples of property {}.".format(counter, p_qid))

def get_test_data(path_autoprompt_dataset, vocab):
    test_dataset = {}

    for file in os.listdir(path_autoprompt_dataset):
        with open(path_autoprompt_dataset+file+"/test.jsonl", "r") as f:
            trex_data = []
            for line in f.readlines():
                trex_data.append(json.loads(line))
            load_trex_file(test_dataset, trex_data, vocab)
    
    count_all_triples = 0
    for p_qid in test_dataset:
            count_all_triples = count_all_triples + len(test_dataset[p_qid])
    print("Lengths of test dataset {}.".format(count_all_triples))

def get_train_data(path_autoprompt_dataset, train_dataset_file, vocab):
    train_dataset = {}
    train_dataset["subj_queries"] = {}
    train_dataset["obj_queries"] = {}
    props = {}
    for file in os.listdir(path_autoprompt_dataset):
        with open(path_autoprompt_dataset+file+"/train.jsonl", "r") as f:
            trex_data = []
            for line in f.readlines():
                trex_data.append(json.loads(line))
            load_trex_file(train_dataset["obj_queries"], trex_data, vocab)
        with open(path_autoprompt_dataset+file+"/dev.jsonl", "r") as f:
            trex_data = []
            for line in f.readlines():
                trex_data.append(json.loads(line))
            load_trex_file(train_dataset["obj_queries"], trex_data, vocab)
        print(file.split(".")[0], len(train_dataset["obj_queries"][file.split(".")[0]]))
        if len(train_dataset["obj_queries"][file.split(".")[0]]) < 1000:
            props[file.split(".")[0]] = len(train_dataset["obj_queries"][file.split(".")[0]])
        print("\n")
    print(props)
    count_all_triples = 0
    for p_qid in train_dataset["obj_queries"]:
            count_all_triples = count_all_triples + len(train_dataset["obj_queries"][p_qid])
    print("Lengths of train dataset {}.".format(count_all_triples))

    #save the train dataset of AUTOPROMPT
    with open(train_dataset_file, "w+") as f:
        json.dump(train_dataset, f, indent=4, sort_keys=True)

if __name__ == "__main__":
    path_autoprompt_dataset = "/data/kalo/akbc2021/AUTOPROMPT/original/"
    vocab_file = "/home/fichtel/BERTriple/LAMA/pre-trained_language_models/bert/cased_L-12_H-768_A-12/vocab.txt"
    with open(vocab_file, "r") as f:
        vocab = set()
        for line in f:
            vocab.add(line.strip())
    
    #get_test_data(path_autoprompt_dataset, vocab)
    
    train_dataset_file = "/data/kalo/akbc2021/training_datasets/AUTOPROMPT41_all.json"
    get_train_data(path_autoprompt_dataset, train_dataset_file, vocab)