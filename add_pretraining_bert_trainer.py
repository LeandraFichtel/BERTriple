import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline, Trainer, TrainingArguments
import json
import os
import shutil
import argparse
from shutil import copyfile
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain, combinations
from evaluation import start_evaluation
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


#functions
def get_queries_answers(dictio_prop_triple, lm_name_initials, query_type, template, omitted_props):
    model_path = "models/baselines/{}_{}".format(lm_name_initials, template)
    fill_mask = pipeline("fill-mask", model=model_path)
    
    queries = []
    answers = []
    print("omitted props:", omitted_props)
    if dictio_prop_triple[query_type] == {}:
        exit("dataset cannot be used for {}".format(query_type))
    for prop in dictio_prop_triple[query_type]:
        print(prop, len(dictio_prop_triple[query_type][prop]))
        for datapoint in dictio_prop_triple[query_type][prop]:
            if omitted_props and prop in omitted_props:
                print("no training data for", prop)
                break
            else:
                subj_label = datapoint["subj"]
                prop =  datapoint["prop"]
                obj_label = datapoint["obj"]
                query_template = dictio_prop_template[prop][template]
                if query_type == "subj_queries":
                    queries.append(query_template.replace("[X]", fill_mask.tokenizer.mask_token).replace("[Y]", obj_label))
                    answers.append(query_template.replace("[X]", subj_label).replace("[Y]", obj_label))
                elif query_type == "obj_queries":
                    queries.append(query_template.replace("[X]", subj_label).replace("[Y]", fill_mask.tokenizer.mask_token))
                    answers.append(query_template.replace("[X]", subj_label).replace("[Y]", obj_label))
    del fill_mask
    return queries, answers

def prepare_dataset(index, lm_name_initials, train_file, sample, query_type, template, omitted_props):
    dataset_path = "data/train_datasets/{}{}_{}.json".format(index, train_file, sample)
    if os.path.exists(dataset_path):
        print("read given dataset", dataset_path)
        with open(dataset_path, "r") as dataset_file:
            dictio_prop_triple = json.load(dataset_file)
    else:
        #prepare dataset
        print("create new dataset", dataset_path)
        dictio_prop_triple = {"subj_queries": {}, "obj_queries": {}}
        dataset_all_path = "data/train_datasets/{}_all.json".format(train_file)
        with open(dataset_all_path) as dataset_file:
            dataset = json.load(dataset_file)
            #take only a sample of the triples
            for queries_type in dataset:
                for prop in dataset[queries_type]:
                    #take only random triples when there are more than the max sample number
                    if len(dataset[queries_type][prop]) > int(sample):
                        dictio_prop_triple[queries_type][prop] = random.sample(dataset[queries_type][prop], int(sample))
                    else:
                        dictio_prop_triple[queries_type][prop] = dataset[queries_type][prop]
        with open(dataset_path, "w") as dataset_file_sample:
            json.dump(dictio_prop_triple, dataset_file_sample)

    all_queries = []
    all_answers = []
    if query_type == "subjobj":
        queries, answers = get_queries_answers(dictio_prop_triple, lm_name_initials, "subj_queries", template, omitted_props)
        all_queries = all_queries + queries
        all_answers = all_answers + answers
        queries, answers = get_queries_answers(dictio_prop_triple, lm_name_initials, "obj_queries", template, omitted_props)
        all_queries = all_queries + queries
        all_answers = all_answers + answers
    elif query_type == "subj":
        queries, answers = get_queries_answers(dictio_prop_triple, lm_name_initials, "subj_queries", template, omitted_props)
        all_queries = all_queries + queries
        all_answers = all_answers + answers
    elif query_type == "obj":
        queries, answers = get_queries_answers(dictio_prop_triple, lm_name_initials, "obj_queries", template, omitted_props)
        all_queries = all_queries + queries
        all_answers = all_answers + answers
    return all_queries, all_answers

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

def get_initials(lm_name):
    lm_name_initials = ""
    initials = lm_name.replace("/", "").split("-")
    for i, initial in enumerate(initials):
        if i == 0:
            lm_name_initials = initial
        else:
            lm_name_initials = lm_name_initials + initial.upper()[0]
    return lm_name_initials

def train(index, lm_name, train_file, sample, epoch, template, query_type, omitted_props):
    lm_name_initials = get_initials(lm_name)
    train_queries, train_answers = prepare_dataset(index, lm_name_initials, train_file, sample, query_type, template, omitted_props)
    print("check datapoint with {} template:".format(template), train_queries[0], train_answers[0])
    #use tokenizer to get encodings
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    train_question_encodings = tokenizer(train_queries, truncation=True, padding='max_length', max_length=256)
    train_answer_encodings = tokenizer(train_answers, truncation=True, padding='max_length', max_length=256)["input_ids"]
    #get final datasets for training
    train_dataset = MaskedDataset(train_question_encodings, train_answer_encodings)
    
    print("start training")
    if omitted_props:
        props_string = "_"
        omitted_props = sorted(omitted_props, key=lambda x: int("".join([i for i in x if i.isdigit()])))
        for prop in omitted_props: 
            props_string = props_string + prop
    else:
        props_string = ""
    
    model_path = "models/{}{}F_{}_{}_{}_{}_{}{}".format(index, lm_name_initials, train_file, sample, query_type, epoch, template, props_string)
    
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
    save_total_limit=0,
    save_strategy="no"
    )
    model = AutoModelForMaskedLM.from_pretrained("models/baselines/{}_{}".format(lm_name_initials, template))

    trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset          # training dataset
    )

    trainer.train()
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    result_path = "results/"+lm_name_initials+"_"+template
    return model_path, result_path

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    dictio_prop_template = json.load(open("/data/fichtel/BERTriple/templates.json", "r"))
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-lm_name', help="name of the model which should be fine-tuned (use the huggingface identifiers: https://huggingface.co/transformers/pretrained_models.html)")
    parser.add_argument('-train_file', help="training dataset name (AUTOPROMPT41)")
    parser.add_argument('-sample', help="set how many triple should be used of each property at maximum (e.g. 500 (=500 triples per prop for each query type) or all (= all given triples per prop for each query type))")
    parser.add_argument('-epoch', help="set how many epoches should be executed")
    parser.add_argument('-template', help="set which template should be used (LAMA or label or ID)")
    parser.add_argument('-query_type', help="set which queries should be used during training (subjobj= subject and object queries, subj= only subject queries, obj= only object queries)")
    parser.add_argument('-transfer_learning', action="store_true", default=False, help="enables one the fly training data creation for transferlearning")
    parser.add_argument('-lama_uhn', action="store_true", default=False, help="set this flag to evaluate also the filtered LAMA UHN dataset (not possible at transfer learning experiment)")

    args = parser.parse_args()
    print(args)
    lm_name = args.lm_name
    train_file = args.train_file
    epoch = int(args.epoch)
    template = args.template
    assert template in ["LAMA", "label", "ID"]
    sample = args.sample
    query_type = args.query_type
    assert query_type in ["subjobj", "subj", "obj"]
    transfer_learning = args.transfer_learning
    lama_uhn = args.lama_uhn

    if transfer_learning:
        #TODO adapt to hf
        #get results of baseline (bert-base-cased)
        result_baseline = dict((pd.read_csv('BERTriple/results/bert_base_{}.csv'.format(template), sep = ',', header = None)).values)
        
        #train all props if it was not already done with this setup
        lm_name_initials = get_initials(lm_name)
        props_string = ""
        model_path_all_trained = "models/{}F_{}_{}_{}_{}_{}{}".format(lm_name_initials, train_file, sample, query_type, epoch, template, props_string)
        if not os.path.exists(model_path_all_trained):
            model_path, model_dir_all_trained = train("", lm_name, train_file, sample, epoch, template, query_type, None)
            #evaluate with LAMA
            start_evaluation(model_dir_all_trained)
        else:
            model_dir_all_trained = model_path_all_trained.split("/")[-1]
        result_all_trained = dict((pd.read_csv('results/{}{}.csv'.format(model_dir_all_trained, lama_uhn), sep = ',', header = None)).values)

        #procotol to save the process of the transfer learning experiment
        protocol = {}
        all_props = ["P1001", "P106", "P1303", "P1376", "P1412", "P178", "P19", "P276", "P30", "P364", "P39", "P449", "P495", "P740", "P101", "P108", "P131", "P138", "P159", "P17", "P20", "P279", "P31", "P36", "P407", "P463", "P527", "P937", "P103", "P127", "P136", "P140", "P176", "P190", "P264", "P27", "P361", "P37", "P413", "P47", "P530"]
        
        #round0: get precision of baseline and after fine-tuning with all props
        round = "round0"
        protocol[round] = {}
        protocol[round]["tested_prop"] = {}
        threshold = 1.1
        for prop in all_props:
            protocol[round]["tested_prop"][prop] = {}
            protocol[round]["tested_prop"][prop]["baseline_prec@1"] = result_baseline[prop]
            protocol[round]["tested_prop"][prop]["trained_prec@1"] = result_all_trained[prop]

        #round1: omitt props seperately
        round = "round1"
        protocol[round] = []
        #remaining_props = protocol["round0"]["remaining_props"]
        for i, prop in enumerate(all_props):
            print("prop {} of {}".format(i, all_props))
            omitted_props = [prop]
            dictio = {}
            dictio["omitted_props"] = omitted_props
            #train
            model_path, model_dir_omitted = train(lm_name, train_file, sample, epoch, template, query_type, omitted_props, lama_uhn)
            #evaluate with LAMA
            result_file_name = start_evaluation(template, model_dir_omitted, omitted_props)
            print("remove dir of model")
            shutil.rmtree(model_path)
            result_omitted = dict((pd.read_csv("results/{}.csv".format(result_file_name), sep = ',', header = None)).values)
            dictio["tested_prop"] = {prop: {}}
            dictio["tested_prop"][prop]["omitted_prec@1"] = result_omitted[prop]
            protocol[round].append(dictio)
        
            #save protocol
            with open("results/transfer_learning_protocols/{}.json".format(result_file_name), "w+") as protocol_file:
                json.dump(protocol, protocol_file, indent=4)
    else:
        for index in ["", "2_", "3_"]:
            #train
            model_path, result_path = train(index, lm_name, train_file, sample, epoch, template, query_type, None)
            
            #evaluate with LAMA
            result_file_name = start_evaluation(template, model_path, result_path)
            if lama_uhn:
                result_file_name = start_evaluation(template, model_path, result_path, lama_uhn=True)
            if sample == "all":
                #because all triples are used during training, there is no need for a second and third run
                break
