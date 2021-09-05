# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
from logging import config
from LAMA.scripts.batch_eval_KB_completion import main as run_evaluation
from LAMA.scripts.batch_eval_KB_completion import load_file
from LAMA.lama.modules import build_model_by_name
import pprint
import statistics
from os import listdir
import os
from os.path import isfile, join
from shutil import copyfile
from collections import defaultdict
import json
import shutil
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline, AutoConfig
from tqdm import tqdm

def run_experiments(
    relations,
    data_path_pre,
    data_path_post,
    input_param,
    use_negated_probes=False,
):
    model = None
    pp = pprint.PrettyPrinter(width=41, compact=True)

    all_Precision1 = []
    type_Precision1 = defaultdict(list)
    type_count = defaultdict(list)

    last_results_file = open("LAMA/last_results.csv", "w+")
    results_file = open("results/{}.csv".format(input_param["result_dir"]), "w+")

    print("evaluating {} props".format(len(relations)))
    for relation in relations:
        pp.pprint(relation)
        PARAMETERS = {
            "dataset_filename": "{}{}{}".format(
                data_path_pre, relation["relation"], data_path_post
            ),
            "common_vocab_filename": "LAMA/pre-trained_language_models/common_vocab_cased.txt",
            "template": "",
            "bert_vocab_name": "vocab.txt",
            "batch_size": 32,
            "logdir": "LAMA_output/output",
            "full_logdir": "LAMA_output/output/results/{}/{}".format(
                input_param["result_dir"], relation["relation"]
            ),
            "lowercase": False,
            "max_sentence_length": 100,
            "threads": -1,
            "interactive": False,
            "use_negated_probes": use_negated_probes,
        }
        #choose between two template types: LAMA templates or label templates
        if "LAMA" in input_param["label"]:
            print("using LAMA templates")
            if "template" in relation:
                PARAMETERS["template"] = relation["template"]
            print(PARAMETERS["template"])
        elif "label" in input_param["label"]:
            print("using label templates")
            with open("/data/fichtel/BERTriple/templates.json", "r") as template__file:
                templates = json.load(template__file)
                PARAMETERS["template"] = templates[relation["relation"]]["label"]
                print(PARAMETERS["template"])
        elif "ID" in input_param["label"]:
            print("using ID templates")
            with open("/data/fichtel/BERTriple/templates.json", "r") as template__file:
                templates = json.load(template__file)
                PARAMETERS["template"] = templates[relation["relation"]]["ID"]
                print(PARAMETERS["template"])
        else:
            exit("Choose only between LAMA templates or label templates or ID templates!")

        PARAMETERS.update(input_param)
        #print(PARAMETERS)

        args = argparse.Namespace(**PARAMETERS)

        # see if file exists
        try:
            data = load_file(args.dataset_filename)
        except Exception as e:
            print("Relation {} excluded.".format(relation["relation"]))
            print("Exception: {}".format(e))
            continue

        if model is None:
            [model_type_name] = args.models_names
            model = build_model_by_name(model_type_name, args)

        Precision1 = run_evaluation(args, shuffle_data=False, model=model)
        print("P@1 : {}".format(Precision1), flush=True)
        all_Precision1.append(Precision1)
        
        last_results_file.write(
            "{},{}\n".format(relation["relation"], round(Precision1 * 100, 2))
        )
        last_results_file.flush()

        results_file.write(
            "{},{}\n".format(relation["relation"], round(Precision1 * 100, 2))
        )
        results_file.flush()

        if "type" in relation:
            type_Precision1[relation["type"]].append(Precision1)
            data = load_file(PARAMETERS["dataset_filename"])
            type_count[relation["type"]].append(len(data))

    mean_p1 = statistics.mean(all_Precision1)
    print("@@@ {} - mean P@1: {}".format(input_param["label"], mean_p1))
    last_results_file.close()
    results_file.write(
        "avg,{}\n".format(mean_p1)
    )
    results_file.close()

    for t, l in type_Precision1.items():

        print(
            "@@@ ",
            input_param["label"],
            t,
            statistics.mean(l),
            sum(type_count[t]),
            len(type_count[t]),
            flush=True,
        )

    return mean_p1, all_Precision1


def get_TREx_parameters(omitted_props, data_path_pre="LAMA/data/"):
    relations = load_file("{}relations.jsonl".format(data_path_pre))
    #only evaluate omitted props to save computational time
    if omitted_props:
        relations_copy = relations.copy()
        for relation in relations:
            if relation["relation"] not in omitted_props:
                relations_copy.remove(relation)
        relations = relations_copy
    data_path_pre += "TREx/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post

def get_TREx_uhn_parameters(omitted_props, data_path_pre="LAMA/data/"):
    relations = load_file("{}relations.jsonl".format(data_path_pre))
    #only evaluate omitted props to save computational time
    if omitted_props:
        relations_copy = relations.copy()
        for relation in relations:
            if relation["relation"] not in omitted_props:
                relations_copy.remove(relation)
        relations = relations_copy
    data_path_pre += "TREx_UHN/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def run_all_LMs(parameters, LMs):
    for ip in LMs:
        print(ip["label"])
        run_experiments(*parameters, input_param=ip, use_negated_probes=False)

def start_custom_model_eval(model_dir, omitted_props=None, lama_uhn=False):
    if lama_uhn:
        print("T-REx UHN evaluation: our models")
        parameters = get_TREx_uhn_parameters(omitted_props)
        result_dir = model_dir + "_uhn"
    else:
        print("T-REx evaluation: our models")
        parameters = get_TREx_parameters(omitted_props)
        result_dir = model_dir
    LMs = [
        {
            "lm": "bert",
            "label": model_dir,
            "result_dir": result_dir,
            "models_names": ["bert"],
            "bert_model_name": "bert-base-cased-finetuned",
            "bert_model_dir": "models/{}".format(model_dir),
        },
    ]
    run_all_LMs(parameters, LMs)
    shutil.rmtree("LAMA_output/output/results/{}".format(result_dir))
    return result_dir        

def start_eval(lm_name, template_type, model_path, result_path, omitted_props=None, lama_uhn=False):
    all_props = ["P1001", "P106", "P1303", "P1376", "P1412", "P178", "P19", "P276", "P30", "P364", "P39", "P449", "P495", "P740", "P101", "P108", "P131", "P138", "P159", "P17", "P20", "P279", "P31", "P36", "P407", "P463", "P527", "P937", "P103", "P127", "P136", "P140", "P176", "P190", "P264", "P27", "P361", "P37", "P413", "P47", "P530"]
    print("considering {} properties".format(len(all_props)))
    fill_mask = pipeline("fill-mask", model=model_path, device=1)
    #choose the dataset und logging path
    if lama_uhn:
        test_data_path = "data/test_datasets/TREx_UHN/"
        result_path = result_path+"_uhn"
        logging_path = model_path+"/logging_lama_uhn"
        print("\nT-REx UHN evaluation: {} saved at {}".format(lm_name, model_path))
    else:
        test_data_path = "data/test_datasets/TREx/"
        logging_path = model_path+"/logging_lama"
        print("\nT-REx evaluation: {} saved at {}".format(lm_name, model_path))
    
    common_vocab = set(open("data/common_vocab_cased.txt", "r").read().splitlines())
    if os.path.exists(logging_path):
        print("remove logging dir of model")
        shutil.rmtree(logging_path)
    os.mkdir(logging_path)
    #iterate through all test_data for all properties
    avg_prec_at_1 = 0
    results_file = open(result_path+".csv", "w+")
    for file_name in os.listdir(test_data_path):
        prop = file_name.split(".")[0]
        if prop in all_props: 
            logging_file = open(logging_path+"/"+file_name, "w+")
            prec_at_1 = 0
            #a triple is valid if the object label of the triple is contained in the common_vocab
            valid_triples = []
            with open(test_data_path+file_name, "r") as file:
                for line in file:
                    dictio_triple = json.loads(line)
                    if dictio_triple["obj_label"] in common_vocab:
                        subj_label = dictio_triple["sub_label"]
                        obj_label = dictio_triple["obj_label"]
                        subj_qid = dictio_triple["sub_uri"]
                        obj_qid = dictio_triple["obj_uri"]
                        valid_triples.append({"subj_label": subj_label, "subj_qid": subj_qid, "obj_label": obj_label, "obj_qid": obj_qid})            
                    else:
                        print(dictio_triple)
            
            print("property {}: {} test queries ".format(prop, len(valid_triples)))
            #get the right template for the prop
            dictio_prop_template = json.load(open("data/templates.json", "r"))
            template = dictio_prop_template[prop][template_type]
            print("using the {} template: {}".format(template_type, template))
            #evaluation (precision@1 per prop and overall avg precison@1)
            for dictio_triple in tqdm(valid_triples):
                #TODO should be the beginning of a sentence with a capital letter?
                mask_query = template.replace("[X]", dictio_triple["subj_label"]).replace("[Y]", fill_mask.tokenizer.mask_token)
                dictio_result = fill_mask(mask_query, top_k=1)[0]
                #create dictio for logging
                dictio_logging = {}
                dictio_triple["masked_sentences"] = [mask_query]
                dictio_logging["query"] = dictio_triple
                dictio_logging["result"] = dictio_result 
                #check whether the predicted token is correct
                if dictio_result["token_str"] == dictio_triple["obj_label"]:
                    prec_at_1 = prec_at_1 + 1
                    dictio_logging["prec@1"] = 1
                else:
                    dictio_logging["prec@1"] = 0
                json.dump(dictio_logging, logging_file, indent=4)
            #calculate precision@1 of each prop averaged over all test queries
            prec_at_1 = prec_at_1/len(valid_triples)
            results_file.write("{},{}\n".format(prop, round(prec_at_1 * 100, 2)))
            avg_prec_at_1 = avg_prec_at_1 + prec_at_1
    #calculate overall precision@1 averaged over all prop
    avg_prec_at_1 = avg_prec_at_1/len(all_props)
    results_file.write("avg,{}\n".format(avg_prec_at_1))

def get_initials(lm_name):
    lm_name_initials = ""
    initials = lm_name.replace("/", "").split("-")
    for i, initial in enumerate(initials):
        if i == 0:
            lm_name_initials = initial
        else:
            lm_name_initials = lm_name_initials + initial.upper()[0]
    return lm_name_initials

if __name__ == "__main__":
    #use the huggingface identifiers: https://huggingface.co/transformers/pretrained_models.html
    examined_lms = ["bert-base-cased", "distilbert-base-cased"]
    #TODO check that there is no "uncased" in the lm name
    
    if not os.path.exists("data/"):
        exit("Please download the data dir from this url: TODO")

    #create dir of each baseline model
    if os.path.exists("models/baselines/"):
        print("remove dir of baseline models")
        shutil.rmtree("models/baselines/")
    os.mkdir("models/baselines/")
    
    #use the common vocab of LAMA as basis and find intersection of all vocabs of examined language models
    common_vocab = set(open("data/LAMA_common_vocab_cased.txt", "r").read().splitlines())
    for lm_name in examined_lms:
        tokenizer = AutoTokenizer.from_pretrained(lm_name)
        vocab = tokenizer.get_vocab().keys()
        common_vocab = common_vocab.intersection(vocab)
    
    assert len(common_vocab) > 0
    print("save new common_vocab_cased.txt")
    with open("data/common_vocab_cased.txt", "w+") as common_vocab_file:
        for token in common_vocab:
            common_vocab_file.write(token+"\n")

    for lm_name in examined_lms:
        lm_name_initials = get_initials(lm_name)
        dictio_prop_template = json.load(open("data/templates.json", "r"))
        for prop in dictio_prop_template:
            all_templates = dictio_prop_template[prop].keys()
            break
        for template_type in all_templates:
            model_path = "models/baselines/"+lm_name_initials+"_"+template_type
            result_path = "results/baselines/"+lm_name_initials+"_"+template_type
            model = AutoModelForMaskedLM.from_pretrained(lm_name)
            model.save_pretrained(model_path, config=True)
            tokenizer = AutoTokenizer.from_pretrained(lm_name)
            tokenizer.save_pretrained(model_path)
            start_eval(lm_name, template_type, model_path, result_path)
            start_eval(lm_name, template_type, model_path, result_path, lama_uhn=True)


