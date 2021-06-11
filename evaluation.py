# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
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


def run_experiments(
    relations,
    data_path_pre,
    data_path_post,
    input_param={
        "lm": "bert",
        "label": "bert_large",
        "models_names": ["bert"],
        "bert_model_name": "bert-large-cased",
        "bert_model_dir": "LAMA/pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    },
    use_negated_probes=False,
):
    model = None
    pp = pprint.PrettyPrinter(width=41, compact=True)

    all_Precision1 = []
    type_Precision1 = defaultdict(list)
    type_count = defaultdict(list)

    last_results_file = open("LAMA/last_results.csv", "w+")
    results_file = open("/home/fichtel/BERTriple/results/{}.csv".format(input_param["label"]), "w+")

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
            "logdir": "/data/fichtel/BERTriple/LAMA_output/output",
            "full_logdir": "/data/fichtel/BERTriple/LAMA_output/output/results/{}/{}".format(
                input_param["label"], relation["relation"]
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
        else:
            exit("Choose only between LAMA templates or label templates!")

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


def run_all_LMs(parameters, LMs):
    for ip in LMs:
        print(ip["label"])
        run_experiments(*parameters, input_param=ip, use_negated_probes=False)

def start_custom_model_eval(model_dir, omitted_props):
    print("T-REx evaluation: our models")
    parameters = get_TREx_parameters(omitted_props)
    LMs = [
        {
            "lm": "bert",
            "label": model_dir,
            "models_names": ["bert"],
            "bert_model_name": "bert-base-cased-finetuned",
            "bert_model_dir": "/home/fichtel/BERTriple/models/{}".format(model_dir),
        },
    ]
    shutil.rmtree("/data/fichtel/BERTriple/LAMA_output/output/results/{}".format(model_dir))
    run_all_LMs(parameters, LMs)


if __name__ == "__main__":
    print("T-REx evaluation: baseline")
    parameters = get_TREx_parameters(None)
    LMs = [
        {
            "lm": "bert",
            "label": "bert_base_label",
            "models_names": ["bert"],
            "bert_model_name": "bert-base-cased",
            "bert_model_dir": "LAMA/pre-trained_language_models/bert/cased_L-12_H-768_A-12",
        },
        {
            "lm": "bert",
            "label": "bert_base_LAMA",
            "models_names": ["bert"],
            "bert_model_name": "bert-base-cased",
            "bert_model_dir": "LAMA/pre-trained_language_models/bert/cased_L-12_H-768_A-12",
        },
    ]
    run_all_LMs(parameters, LMs)


