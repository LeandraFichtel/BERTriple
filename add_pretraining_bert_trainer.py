from transformers import BertForMaskedLM, BertModel, Trainer, TrainingArguments
import json
from transformers import BertTokenizer
import torch
#from torch.utils.data import DataLoader
from transformers import AdamW
import dill
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import shutil
import argparse
#from torchsummary import summary
from shutil import copyfile
import random



#functions
def read_dataset(dataset_path, dictio_prop_template, query_type, sample, template):
    queries = []
    answers = []
    #print(dataset_path)
    #dataset_file = json.load(open("/home/kalo/conferences/akbc2021/data/data/TREx/wikidata_training.json", "r"))
    dataset_file = json.load(open(dataset_path, "r"))
    print(dataset_path)
    
    #dictio = {}
    #dictio["obj_queries"] = []
    #for datapoint in dataset_file:
    #    dp = {}
    #    dp["subj"] = datapoint["subject"]
    #    dp["prop"] = datapoint["predicate"]
    #    dp["obj"] = datapoint["object"]
    #    dictio["obj_queries"].append(dp)
    #file = open("/data/fichtel/BERTriple/training_datasets/wikidata41.json", "w")
    #json.dump(dictio, file)
    #exit()
    #prepare dataset
    dictio_prop_triple = {"subj_queries": {}, "obj_queries": {}}
    for queries_type in dataset_file:
        for datapoint in dataset_file[queries_type]:
            prop =  datapoint["prop"]
            if prop not in dictio_prop_triple[queries_type]:
                dictio_prop_triple[queries_type][prop] = [datapoint]
            else:
                dictio_prop_triple[queries_type][prop].append(datapoint)
    #take only a sample of the triples if it is wanted
    if sample != "all":
        for queries_type in dictio_prop_triple:
            for prop in dictio_prop_triple[queries_type]:
                #take only random triples when there are more than the max sample number
                if len(dictio_prop_triple[queries_type][prop]) > int(sample):
                    dictio_prop_triple[queries_type][prop] = random.sample(dictio_prop_triple[queries_type][prop], int(sample))
                
    del dataset_file
    if query_type == "subjobj":
        if dictio_prop_triple["subj_queries"] == {} or dictio_prop_triple["obj_queries"] == {}:
            exit("dataset cannot be used for subjobj")
        for prop in dictio_prop_triple["subj_queries"]:
            for datapoint in dictio_prop_triple["subj_queries"][prop]:
                subj_label = datapoint["subj"]
                prop =  datapoint["prop"]
                obj_label = datapoint["obj"]
                query_template = dictio_prop_template[prop][template]
                queries.append(query_template.replace("[S]", "[MASK]").replace("[O]", obj_label))
                answers.append(query_template.replace("[S]", subj_label).replace("[O]", obj_label))
        for prop in dictio_prop_triple["obj_queries"]:
            for datapoint in dictio_prop_triple["obj_queries"][prop]:
                subj_label = datapoint["subj"]
                prop =  datapoint["prop"]
                obj_label = datapoint["obj"]
                query_template = dictio_prop_template[prop][template]
                queries.append(query_template.replace("[S]", subj_label).replace("[O]", "[MASK]"))
                answers.append(query_template.replace("[S]", subj_label).replace("[O]", obj_label))
    elif query_type == "subj":
        if dictio_prop_triple["subj_queries"] == {}:
            exit("dataset cannot be used for subj")
        for prop in dictio_prop_triple["subj_queries"]:
            for datapoint in dictio_prop_triple["subj_queries"][prop]:
                subj_label = datapoint["subj"]
                prop =  datapoint["prop"]
                obj_label = datapoint["obj"]
                query_template = dictio_prop_template[prop][template]
                queries.append(query_template.replace("[S]", "[MASK]").replace("[O]", obj_label))
                answers.append(query_template.replace("[S]", subj_label).replace("[O]", obj_label))
    elif query_type == "obj":
        if dictio_prop_triple["obj_queries"] == {}:
            exit("dataset cannot be used for obj")
        for prop in dictio_prop_triple["obj_queries"]:
            print(prop, len(dictio_prop_triple["obj_queries"][prop]))
            for datapoint in dictio_prop_triple["obj_queries"][prop]:
                subj_label = datapoint["subj"]
                prop =  datapoint["prop"]
                obj_label = datapoint["obj"]
                query_template = dictio_prop_template[prop][template]
                queries.append(query_template.replace("[S]", subj_label).replace("[O]", "[MASK]"))
                answers.append(query_template.replace("[S]", subj_label).replace("[O]", obj_label))
    return queries, answers

class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    dictio_prop_template = json.load(open("/data/fichtel/BERTriple/templates.json", "r"))

    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_file', help="filename of the training dataset")
    parser.add_argument('-sample', help="set how many triple should be used of each property at maximum (e.g. 500 (=500 triples per prop for each query type) or all (= all given triples per prop for each query type))")
    parser.add_argument('-epoch', help="set how many epoches should be executed")
    parser.add_argument('-template', help="set which template should be used (LAMA or label)")
    parser.add_argument('-query_type', help="set which queries should be used during training (subjobj= subject and object queries, subj= only subject queries, obj= only object queries)")
    
    args = parser.parse_args()
    print(args)
    train_file = args.train_file
    epoch = int(args.epoch)
    template = args.template
    sample = args.sample
    query_type = args.query_type
    assert(query_type in ["subjobj", "subj", "obj"])

    #used LM
    lm_name = 'bert-base-cased'
    #pepare training dataset
    #read datasets from path
    train_queries, train_answers = read_dataset("/data/fichtel/BERTriple/training_datasets/{}".format(train_file), dictio_prop_template, query_type, sample, template)
    print(train_queries[0], train_answers[0])
    #use tokenizer to get encodings
    tokenizer = BertTokenizer.from_pretrained(lm_name)
    train_question_encodings = tokenizer(train_queries, truncation=True, padding='max_length', max_length=256)
    train_answer_encodings = tokenizer(train_answers, truncation=True, padding='max_length', max_length=256)["input_ids"]
    #get final datasets for training
    train_dataset = MaskedDataset(train_question_encodings, train_answer_encodings)
    
    model = BertModel.from_pretrained(lm_name)

    print("start training")
    lm_name_path = lm_name.replace("-", "_")
    model_path = "/data/fichtel/BERTriple/models/{}_finetuned_{}_{}_{}_{}_{}".format(lm_name_path, train_file.split(".")[0], sample, query_type, epoch, template)
    if os.path.exists(model_path):
        print("remove dir of model")
        shutil.rmtree(model_path)
    os.mkdir(model_path)
    
    training_args = TrainingArguments(
    output_dir=model_path+'/results',          # output directory
    num_train_epochs=epoch,           # total number of training epochs
    per_device_train_batch_size=16,   # batch size per device during training
    per_device_eval_batch_size=64,    # batch size for evaluation
    warmup_steps=500,                 # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                # strength of weight decay
    logging_dir=model_path+'/logs',   # directory for storing logs
    logging_steps=10,
    save_total_limit=0
    )
    
    model = BertForMaskedLM.from_pretrained(lm_name)

    trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset          # training dataset
    )

    trainer.train()
    trainer.save_model(model_path)
    copyfile("/home/fichtel/BERTriple/LAMA/pre-trained_language_models/bert/cased_L-12_H-768_A-12/vocab.txt", model_path + "/vocab.txt")
    os.rename(model_path + "/config.json", model_path + "/bert_config.json")