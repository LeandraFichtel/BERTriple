import torch
from transformers import BertForMaskedLM, BertModel, Trainer, TrainingArguments
import json
from transformers import BertTokenizer
import os
import shutil
import argparse
from shutil import copyfile
import random
import pandas as pd
from itertools import chain, combinations
from evaluation import start_custom_model_eval
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


#functions
def get_queries_answers(dictio_prop_triple, query_type, omitted_props):
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
                    queries.append(query_template.replace("[X]", "[MASK]").replace("[Y]", obj_label))
                    answers.append(query_template.replace("[X]", subj_label).replace("[Y]", obj_label))
                elif query_type == "obj_queries":
                    queries.append(query_template.replace("[X]", subj_label).replace("[Y]", "[MASK]"))
                    answers.append(query_template.replace("[X]", subj_label).replace("[Y]", obj_label))
    return queries, answers

def prepare_dataset(dataset_path, dictio_prop_template, query_type, sample, template, omitted_props):
    if os.path.exists(dataset_path):
        print("read given dataset", dataset_path)
        with open(dataset_path, "r") as dataset_file:
            dictio_prop_triple = json.load(dataset_file)
    else:
        #prepare dataset
        print("create new dataset", dataset_path)
        dictio_prop_triple = {"subj_queries": {}, "obj_queries": {}}
        with open(dataset_path.replace(sample, "all")) as dataset_file:
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
        queries, answers = get_queries_answers(dictio_prop_triple, "subj_queries", omitted_props)
        all_queries = all_queries + queries
        all_answers = all_answers + answers
        queries, answers = get_queries_answers(dictio_prop_triple, "obj_queries", omitted_props)
        all_queries = all_queries + queries
        all_answers = all_answers + answers
    elif query_type == "subj":
        queries, answers = get_queries_answers(dictio_prop_triple, "subj_queries", omitted_props)
        all_queries = all_queries + queries
        all_answers = all_answers + answers
    elif query_type == "obj":
        queries, answers = get_queries_answers(dictio_prop_triple, "obj_queries", omitted_props)
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

def train(lm_name, train_file, sample, epoch, template, query_type, omitted_props):
    train_queries, train_answers = prepare_dataset("/data/fichtel/BERTriple/training_datasets/{}_{}.json".format(train_file, sample), dictio_prop_template, query_type, sample, template, omitted_props)
    print("check datapoints:", train_queries[0], train_answers[0])
    #use tokenizer to get encodings
    tokenizer = BertTokenizer.from_pretrained(lm_name)
    train_question_encodings = tokenizer(train_queries, truncation=True, padding='max_length', max_length=256)
    train_answer_encodings = tokenizer(train_answers, truncation=True, padding='max_length', max_length=256)["input_ids"]
    #get final datasets for training
    train_dataset = MaskedDataset(train_question_encodings, train_answer_encodings)
    
    print("start training")
    lm_name_short = lm_name.split("-")
    lm_name_capitals = lm_name_short[0].upper()[0] + lm_name_short[1].upper()[0] + lm_name_short[2].upper()[0]
    if omitted_props:
        props_string = "_"
        omitted_props = sorted(omitted_props, key=lambda x: int("".join([i for i in x if i.isdigit()])))
        for prop in omitted_props: 
            props_string = props_string + prop
    else:
        props_string = ""
    
    model_path = "/data/fichtel/BERTriple/models/{}F_{}_{}_{}_{}_{}{}".format(lm_name_capitals, train_file, sample, query_type, epoch, template, props_string)
    
    #if os.path.exists(model_path):
    #    print("remove dir of model")
    #    shutil.rmtree(model_path)
    #os.mkdir(model_path)
    
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
    return model_path.split("/")[-1]

if __name__ == "__main__":
    #os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    dictio_prop_template = json.load(open("/data/fichtel/BERTriple/templates.json", "r"))
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_file', help="training dataset name (LPAQAfiltered41/LPAQAfiltered25 or wikidata41/wikidata25)")
    parser.add_argument('-sample', help="set how many triple should be used of each property at maximum (e.g. 500 (=500 triples per prop for each query type) or all (= all given triples per prop for each query type))")
    parser.add_argument('-epoch', help="set how many epoches should be executed")
    parser.add_argument('-template', help="set which template should be used (LAMA or label)")
    parser.add_argument('-query_type', help="set which queries should be used during training (subjobj= subject and object queries, subj= only subject queries, obj= only object queries)")
    parser.add_argument('-transfer_learning', action="store_true", default=False, help="enables one the fly training data creation for transferlearning")

    args = parser.parse_args()
    print(args)
    train_file = args.train_file
    epoch = int(args.epoch)
    template = args.template
    sample = args.sample
    query_type = args.query_type
    assert(query_type in ["subjobj", "subj", "obj"])
    transfer_learning = args.transfer_learning

    #used LM
    lm_name = 'bert-base-cased'  
    
    if transfer_learning:
        #get results of baseline (bert-base-cased)
        result_baseline = dict((pd.read_csv('/home/fichtel/BERTriple/results/bert_base_{}.csv'.format(template), sep = ',', header = None)).values)
        
        #train all props if it was not already done with this setup
        lm_name_short = lm_name.split("-")
        lm_name_capitals = lm_name_short[0].upper()[0] + lm_name_short[1].upper()[0] + lm_name_short[2].upper()[0]
        props_string = ""
        model_path_all_trained = "/data/fichtel/BERTriple/models/{}F_{}_{}_{}_{}_{}{}".format(lm_name_capitals, train_file, sample, query_type, epoch, template, props_string)
        if not os.path.exists(model_path_all_trained):
            model_dir_all_trained = train(lm_name, train_file, sample, epoch, template, query_type, None)
            #evaluate with LAMA
            start_custom_model_eval(model_dir_all_trained)
        else:
            model_dir_all_trained = model_path_all_trained.split("/")[-1]
        result_all_trained = dict((pd.read_csv('/home/fichtel/BERTriple/results/{}.csv'.format(model_dir_all_trained), sep = ',', header = None)).values)

        #procotol to save the process of the transfer learning experiment
        protocol = {}
        all_props = ["P1001", "P106", "P1303", "P1376", "P1412", "P178", "P19", "P276", "P30", "P364", "P39", "P449", "P495", "P740", "P101", "P108", "P131", "P138", "P159", "P17", "P20", "P279", "P31", "P36", "P407", "P463", "P527", "P937", "P103", "P127", "P136", "P140", "P176", "P190", "P264", "P27", "P361", "P37", "P413", "P47", "P530"]
        
        #round0: remove props that have no prec@1 improvement after fine-tuning compared to baseline
        round = "round0"
        protocol[round] = {}
        protocol[round]["tested_prop"] = {}
        threshold = 1.1
        props = all_props.copy()
        remaining_props = []
        for prop in all_props:
            protocol[round]["tested_prop"][prop] = {}
            protocol[round]["tested_prop"][prop]["baseline_prec@1"] = result_baseline[prop]
            protocol[round]["tested_prop"][prop]["trained_prec@1"] = result_all_trained[prop]
            if protocol[round]["tested_prop"][prop]["trained_prec@1"] > threshold * protocol[round]["tested_prop"][prop]["baseline_prec@1"]:
                remaining_props.append(prop)
        protocol[round]["remaining_props"] = remaining_props
        
        #round1: omitt props only seperately
        round = "round1"
        protocol[round] = []
        for i, prop in enumerate(all_props):
            omitted_props = [prop]
            dictio = {}
            dictio["omitted_props"] = omitted_props
            ##ONLY FOR THIS TIME, HAS TO BE CHANGED TODO
            #if omitted_props:
            #    props_string = "_"
            #    omitted_props = sorted(omitted_props, key=lambda x: int("".join([i for i in x if i.isdigit()])))
            #    for prop in omitted_props: 
            #        props_string = props_string + prop
            #model_dir_omitted = "{}F_{}_{}_{}_{}_{}{}".format(lm_name_capitals, train_file, sample, query_type, epoch, template, props_string)
            #if os.path.exists("/home/fichtel/BERTriple/results/{}.csv".format(model_dir_omitted)):
            #    print("no training and eval for", prop)
            #else:
        
            #train
            model_dir_omitted = train(lm_name, train_file, sample, epoch, template, query_type, omitted_props)
            #evaluate with LAMA
            start_custom_model_eval(model_dir_omitted, omitted_props)
            result_omitted = dict((pd.read_csv("/home/fichtel/BERTriple/results/{}.csv".format(model_dir_omitted), sep = ',', header = None)).values)
            dictio["tested_prop"] = {prop: {}}
            dictio["tested_prop"][prop]["omitted_prec@1"] = result_omitted[prop]
            protocol[round].append(dictio)
        
        #save protocol
        with open("/home/fichtel/BERTriple/results/transfer_learning_protocols/{}F_{}_{}_{}_{}_{}.json".format(lm_name_capitals, train_file, sample, query_type, epoch, template), "w+") as protocol_file:
            json.dump(protocol, protocol_file, indent=4)

        #protocol = json.load(open("/home/fichtel/BERTriple/results/transfer_learning_protocols/BBCF_LPAQAfiltered41_100_obj_3_LAMA.json", "r"))
        #prepare round2 to remove the props for which trained_prec@1 != omitted_prec@1 and to define pairs of remaining props
        #threshold = 0.9
        #props = all_props.copy()
        #for experiment in protocol["round1"]:
        #    for prop in experiment["tested_prop"]:
        #        if experiment["tested_prop"][prop]["omitted_prec@1"] < threshold * experiment["tested_prop"][prop]["trained_prec@1"]:
        #            props.remove(prop)
        #print("remaining props:", props, len(props))
        #props_pairs = [(props[i],props[j]) for i in range(len(props)) for j in range(i+1, len(props))]
        props_pairs = [(all_props[i],all_props[j]) for i in range(len(all_props)) for j in range(i+1, len(all_props))]

        #round2: omitt pairs of props
        round = "round2"
        protocol[round] = []
        for i, pair in enumerate(props_pairs):
            print("considering pair {} of {} pairs".format(i, len(props_pairs)))
            omitted_props = list(pair)
            dictio = {}
            dictio["omitted_props"] = omitted_props
            ##ONLY FOR THIS TIME, HAS TO BE CHANGED TODO
            #if omitted_props:
            #    props_string = "_"
            #    omitted_props = sorted(omitted_props, key=lambda x: int("".join([i for i in x if i.isdigit()])))
            #    for prop in omitted_props: 
            #        props_string = props_string + prop
            #model_dir_omitted = "{}F_{}_{}_{}_{}_{}{}".format(lm_name_capitals, train_file, sample, query_type, epoch, template, props_string)
            #if os.path.exists("/home/fichtel/BERTriple/results/{}.csv".format(model_dir_omitted)):
            #    print("no training and eval for", prop)
            #else:

            #train
            model_dir_omitted = train(lm_name, train_file, sample, epoch, template, query_type, omitted_props)
            #evaluate with LAMA
            start_custom_model_eval(model_dir_omitted, omitted_props)
            result_omitted = dict((pd.read_csv("/home/fichtel/BERTriple/results/{}.csv".format(model_dir_omitted), sep = ',', header = None)).values)
            dictio["tested_prop"] = {}
            for prop in omitted_props:
                dictio["tested_prop"][prop] = {}
                dictio["tested_prop"][prop]["omitted_prec@1"] = result_omitted[prop]
            protocol[round].append(dictio)
        
        #save protocol
        with open("/home/fichtel/BERTriple/results/transfer_learning_protocols/{}F_{}_{}_{}_{}_{}.json".format(lm_name_capitals, train_file, sample, query_type, epoch, template), "w+") as protocol_file:
            json.dump(protocol, protocol_file, indent=4)

        #find groups
        dependent_props = {}
        threshold = 0.9
        for prop in all_props:
            dependent_props[prop] = []
            for experiment in protocol["round1"]:
                if prop in experiment["tested_prop"]:
                    round1_omitted_prec = experiment["tested_prop"][prop]["omitted_prec@1"]
                    break
            for experiment in protocol["round2"]:
                if prop in experiment["tested_prop"]:
                    round2_omitted_prec = experiment["tested_prop"][prop]["omitted_prec@1"]
                    if round2_omitted_prec < threshold * round1_omitted_prec:
                        omitted_props = experiment["omitted_props"].copy()
                        omitted_props.remove(prop)
                        dependent_props[prop].append(omitted_props[0])
        protocol["dependent_props"] = dependent_props
        #save protocol
        with open("/home/fichtel/BERTriple/results/transfer_learning_protocols/{}F_{}_{}_{}_{}_{}.json".format(lm_name_capitals, train_file, sample, query_type, epoch, template), "w+") as protocol_file:
            json.dump(protocol, protocol_file, indent=4)
        #building graph
        graph = {}
        for prop in all_props:
            graph[prop] = {}
            for experiment in protocol["round1"]:
                if prop in experiment["tested_prop"]:
                    round1_omitted_prec = experiment["tested_prop"][prop]["omitted_prec@1"]
                    break
            for experiment in protocol["round2"]:
                if prop in experiment["tested_prop"]:
                    round2_omitted_prec = experiment["tested_prop"][prop]["omitted_prec@1"]
                    omitted_props = experiment["omitted_props"].copy()
                    omitted_props.remove(prop)
                    graph[prop][omitted_props[0]] = round2_omitted_prec/round1_omitted_prec
        protocol["graph"] = graph
        #save protocol
        with open("/home/fichtel/BERTriple/results/transfer_learning_protocols/{}F_{}_{}_{}_{}_{}.json".format(lm_name_capitals, train_file, sample, query_type, epoch, template), "w+") as protocol_file:
            json.dump(protocol, protocol_file, indent=4)
    else:
        #train
        model_dir = train(lm_name, train_file, sample, epoch, template, query_type, None)
        #evaluate with LAMA
        start_custom_model_eval(model_dir)