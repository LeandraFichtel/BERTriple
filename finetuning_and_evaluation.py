import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline, Trainer, TrainingArguments
import json
import os
import shutil
import argparse
import random
import pandas as pd
from evaluation import start_evaluation
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def valid_label(obj_label, vocab_type, common_vocab, tokenizer):
    if vocab_type == "common":
        return obj_label in common_vocab
    elif vocab_type == "different":
        if 3 not in tokenizer(obj_label)["input_ids"]:
            return True
        else:
            return False

#functions
def get_queries_answers(dictio_prop_triple, vocab_type, lm_name_initials, query_type, template, omitted_props):
    model_path = "models/baselines_{}_vocab/{}_{}".format(vocab_type, lm_name_initials, template)
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

def prepare_dataset(index, vocab_type, lm_name, lm_name_initials, train_file, sample, query_type, template, omitted_props):
    if vocab_type == "common":
        common_vocab = set(open("data/common_vocab_cased.txt", "r").read().splitlines())
        tokenizer = None
        dataset_path = "data/train_datasets/{}{}_common_{}.json".format(index, train_file, sample)
        dataset_all_path = "data/train_datasets/{}_common_all.json".format(train_file)
    elif vocab_type == "different":
        common_vocab = None
        tokenizer = AutoTokenizer.from_pretrained(lm_name)
        dataset_path = "data/train_datasets/{}{}_{}_{}.json".format(index, train_file, lm_name_initials, sample)
        dataset_all_path = "data/train_datasets/{}_{}_all.json".format(train_file, lm_name_initials)
    if not os.path.exists(dataset_all_path):
        print("create new all dataset", dataset_all_path)
        dictio_prop_triple = {"subj_queries": {}, "obj_queries": {}}
        dataset_all_general_path = "data/train_datasets/{}_all.json".format(train_file)
        with open(dataset_all_general_path) as dataset_file:
            dataset = json.load(dataset_file)
            #take only the triples where the obj labels are valid, thus they are contained in the vocab
            for queries_type in dataset:
                for prop in dataset[queries_type]:
                    dictio_prop_triple[queries_type][prop] = []
                    for triple in dataset[queries_type][prop]:
                        if valid_label(triple["obj"], vocab_type, common_vocab, tokenizer):
                            dictio_prop_triple[queries_type][prop].append(triple)
        with open(dataset_all_path, "w") as dataset_file_sample:
            json.dump(dictio_prop_triple, dataset_file_sample)
    
    if os.path.exists(dataset_path):
        print("read given dataset", dataset_path)
        with open(dataset_path, "r") as dataset_file:
            dictio_prop_triple = json.load(dataset_file)
    else:
        #prepare dataset
        print("create new dataset", dataset_path)
        dictio_prop_triple = {"subj_queries": {}, "obj_queries": {}}
        if vocab_type == "common":
            dataset_all_path = "data/train_datasets/{}_common_all.json".format(train_file)
        elif vocab_type == "different":
            dataset_all_path = "data/train_datasets/{}_{}_all.json".format(train_file, lm_name_initials)
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
        queries, answers = get_queries_answers(dictio_prop_triple, vocab_type, lm_name_initials, "subj_queries", template, omitted_props)
        all_queries = all_queries + queries
        all_answers = all_answers + answers
        queries, answers = get_queries_answers(dictio_prop_triple, vocab_type, lm_name_initials, "obj_queries", template, omitted_props)
        all_queries = all_queries + queries
        all_answers = all_answers + answers
    elif query_type == "subj":
        queries, answers = get_queries_answers(dictio_prop_triple, vocab_type, lm_name_initials, "subj_queries", template, omitted_props)
        all_queries = all_queries + queries
        all_answers = all_answers + answers
    elif query_type == "obj":
        queries, answers = get_queries_answers(dictio_prop_triple, vocab_type, lm_name_initials, "obj_queries", template, omitted_props)
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

def train(index, vocab_type, lm_name, train_file, sample, epoch, template, query_type, omitted_props):
    lm_name_initials = get_initials(lm_name)
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    train_queries, train_answers = prepare_dataset(index, vocab_type, lm_name, lm_name_initials, train_file, sample, query_type, template, omitted_props)
    print("check datapoint with {} template:".format(template), train_queries[0], train_answers[0])
    #use tokenizer to get encodings
    train_question_encodings = tokenizer(train_queries, truncation=True, padding='max_length', max_length=256)
    train_answer_encodings = tokenizer(train_answers, truncation=True, padding='max_length', max_length=256)["input_ids"]
    #get final dataset for training
    train_dataset = MaskedDataset(train_question_encodings, train_answer_encodings)
    
    print("start training")
    if omitted_props:
        props_string = "_"
        omitted_props = sorted(omitted_props, key=lambda x: int("".join([i for i in x if i.isdigit()])))
        for prop in omitted_props: 
            props_string = props_string + prop
    else:
        props_string = ""
    
    model_dir = "{}{}F_{}_{}_{}_{}_{}_{}{}".format(index, lm_name_initials, train_file, vocab_type, sample, query_type, epoch, template, props_string)
    model_path = "models/"+model_dir
    if os.path.exists(model_path):
        print("remove dir of model")
        shutil.rmtree(model_path)
    os.mkdir(model_path)
    
    training_args = TrainingArguments(
    output_dir=model_path+'/results', # output directory
    num_train_epochs=epoch,           # total number of training epochs
    per_device_train_batch_size=12,   # batch size per device during training
    per_device_eval_batch_size=64,    # batch size for evaluation
    warmup_steps=500,                 # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                # strength of weight decay
    logging_dir=model_path+'/logs',   # directory for storing logs
    logging_steps=10,
    save_total_limit=0,
    save_strategy="no"
    )
    model = AutoModelForMaskedLM.from_pretrained("models/baselines_{}_vocab/{}_{}".format(vocab_type, lm_name_initials, template))

    trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset          # training dataset
    )

    trainer.train()
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    return model_path, model_dir

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    dictio_prop_template = json.load(open("/data/fichtel/BERTriple/templates.json", "r"))
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-lm_name', help="name of the model which should be fine-tuned (use the huggingface identifiers: https://huggingface.co/transformers/pretrained_models.html)")
    parser.add_argument('-vocab', help="set whether the common vocab or the corresponding vocab of the fine-tuned lm should be used")
    parser.add_argument('-train_file', help="training dataset name (AUTOPROMPT41)")
    parser.add_argument('-sample', help="set how many triple should be used of each property at maximum (e.g. 500 (=500 triples per prop for each query type) or all (= all given triples per prop for each query type))")
    parser.add_argument('-epoch', help="set how many epoches should be executed")
    parser.add_argument('-template', help="set which template should be used (LAMA or label or ID)")
    parser.add_argument('-query_type', help="set which queries should be used during training (subjobj= subject and object queries, subj= only subject queries, obj= only object queries)")
    parser.add_argument('-transfer_learning', action="store_true", default=False, help="enables on-the-fly training data creation for transfer learning experiment")
    parser.add_argument('-lama_uhn', action="store_true", default=False, help="set this flag to evaluate also the filtered LAMA UHN dataset (not possible at transfer learning experiment)")

    args = parser.parse_args()
    print(args)
    lm_name = args.lm_name
    vocab_type = args.vocab
    assert vocab_type in ["common", "different"]
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
        lm_name_initials = get_initials(lm_name)
        #get results of baseline
        results_baseline = dict((pd.read_csv('results/baselines_{}_vocab/{}_{}.csv'.format(vocab_type, lm_name_initials, template), sep = ',', header = None)).values)
        
        #train all props if it was not already done with this setup
        props_string = ""
        model_dir = "{}F_{}_{}_{}_{}_{}{}".format(lm_name_initials, train_file, sample, query_type, epoch, template, props_string)
        model_path_all_trained = "models/"+model_dir
        if not os.path.exists(model_path_all_trained):
            model_path_all_trained, results_file_name_all_trained = train("", vocab_type, lm_name, train_file, sample, epoch, template, query_type, None)
            #evaluate with huggingface
            start_evaluation(template, vocab_type, model_path_all_trained, results_file_name_all_trained)
        else:
            results_file_name_all_trained = model_dir
        results_all_trained = dict((pd.read_csv('results/{}.csv'.format(results_file_name_all_trained), sep = ',', header = None)).values)

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
            protocol[round]["tested_prop"][prop]["baseline_prec@1"] = results_baseline[prop]
            protocol[round]["tested_prop"][prop]["trained_prec@1"] = results_all_trained[prop]

        #round1: omitt props seperately
        round = "round1"
        protocol[round] = []
        for i, prop in enumerate(all_props):
            print("prop {} of {}".format(i, all_props))
            omitted_props = [prop]
            dictio = {}
            dictio["omitted_props"] = omitted_props
            #train
            model_path_omitted, results_file_name_omitted = train("", vocab_type, lm_name, train_file, sample, epoch, template, query_type, omitted_props)
            #evaluate with huggingface
            start_evaluation(template, vocab_type, model_path_omitted, results_file_name_omitted, omitted_props=omitted_props)
            print("remove dir of model")
            shutil.rmtree(model_path_omitted)
            results_omitted = dict((pd.read_csv("results/{}.csv".format(results_file_name_omitted), sep = ',', header = None)).values)
            dictio["tested_prop"] = {prop: {}}
            dictio["tested_prop"][prop]["omitted_prec@1"] = results_omitted[prop]
            protocol[round].append(dictio)
        
            #save protocol
            with open("results/transfer_learning_protocols/{}.json".format(results_file_name_all_trained), "w+") as protocol_file:
                json.dump(protocol, protocol_file, indent=4)
    else:
        for index in ["", "2_", "3_"]:
            #train
            model_path, results_file_name = train(index, vocab_type, lm_name, train_file, sample, epoch, template, query_type, None)
            
            #evaluate with huggingface
            start_evaluation(template, vocab_type, model_path, results_file_name)
            if lama_uhn:
                start_evaluation(template, vocab_type, model_path, results_file_name, lama_uhn=True)
            if sample == "all":
                #because all triples are used during training, there is no need for a second and third run
                break
