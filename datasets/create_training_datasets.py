import json
import random
import os
import argparse

def create_training_dataset_file(string_token, template, bool_all_alone, min_sample, sample):
    if bool_all_alone == "all":
        bool_all_alone = ""
    if template == "auto":
        if os.path.exists("/data/fichtel/projektarbeit/auto_templates/tuples_for_ranking_{}_{}_{}_{}.json".format(string_token, template, min_sample, sample)):
            print("removed tuples_for_ranking_{}_{}_{}_{}.json".format(string_token, template, min_sample, sample))
            os.remove("/data/fichtel/projektarbeit/auto_templates/tuples_for_ranking_{}_{}_{}_{}.json".format(string_token, template, min_sample, sample))
        training_file = open("/data/fichtel/projektarbeit/auto_templates/tuples_for_ranking_{}_{}_{}_{}.json".format(string_token, template, min_sample, sample), "w")
    else:
        if os.path.exists("/data/fichtel/projektarbeit/training_datasets/training_dataset_100_{}_{}{}_{}_{}.json".format(string_token, template, bool_all_alone, min_sample, sample)):
            print("removed training_dataset100__{}_{}{}_{}_{}.json".format(string_token, template, bool_all_alone, min_sample, sample))
            os.remove("/data/fichtel/projektarbeit/training_datasets/training_dataset_100_{}_{}{}_{}_{}.json".format(string_token, template, bool_all_alone, min_sample, sample))
        training_file = open("/data/fichtel/projektarbeit/training_datasets/training_dataset_100_{}_{}{}_{}_{}.json".format(string_token, template, bool_all_alone, min_sample, sample), "w")
    return training_file

def create_training_dataset_files_random_props(string_token, template, bool_all_alone, min_sample, sample, percentages):
    if bool_all_alone == "all":
        bool_all_alone = ""
    training_files = {}
    for percent in percentages:
        training_files[percent] = []
        for i in range(len(dictio_random_props[percent])):
            if os.path.exists("/data/fichtel/projektarbeit/training_datasets/training_dataset_{}-{}_{}_{}{}_{}_{}.json".format(percent, i, string_token, template, bool_all_alone, min_sample, sample)):
                print("removed training_dataset_{}-{}_{}_{}{}_{}_{}.json".format(percent, i, string_token, template, bool_all_alone, min_sample, sample))
                os.remove("/data/fichtel/projektarbeit/training_datasets/training_dataset_{}-{}_{}_{}{}_{}_{}.json".format(percent, i, string_token, template, bool_all_alone, min_sample, sample))
            training_files[percent].append(open("/data/fichtel/projektarbeit/training_datasets/training_dataset_{}-{}_{}_{}{}_{}_{}.json".format(percent, i, string_token, template, bool_all_alone, min_sample, sample), "w"))
    return training_files

if __name__ == "__main__":
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-min_sample', help="set how many triplets at least should exist of each property in wikidata_onetoken_missing")
    parser.add_argument('-sample', help="set how many triplets should be used of each property (e.g. 500 for 500 subject and 500 object triplets)")
    parser.add_argument('-template', help="set which template should be used (LAMA or label or all)")
    parser.add_argument('-string_token', help="set if obj and subj labels should consist of only one word (oneword) and are also in vocab file (onetoken)")
    parser.add_argument('-random_props', action="store_true", default=False, help="set flag if training file for random props should also be created, only for sample=100 and sample=500 and temple=label")
    
    args = parser.parse_args()
    print(args)
    template = args.template
    sample = args.sample
    min_sample = args.min_sample
    if int(sample) > int(min_sample):
        raise("ERROR: the sample size cannot be bigger than the min_sample size")
    string_token = args.string_token
    assert(string_token in ["onetoken", "oneword", ""])
    random_props = args.random_props

    dataset_file = open("/data/fichtel/projektarbeit/dataset_{}_{}.json".format(string_token, min_sample), "r")
    dictio_prop_train_query_answer = json.load(dataset_file)
    number_all_queries = 0
    number_training_queries = 0

    #get all query-answer-tuplets of each property
    dictio_random_props = dictio_prop_train_query_answer["random_props"]
    del dictio_prop_train_query_answer["random_props"]
    print("#props", len(dictio_prop_train_query_answer))
    
    if template == "all":
        #same random datapoints for all templates
        LAMA_training_file = create_training_dataset_file(string_token, "LAMA", "all", min_sample, sample)
        label_training_file = create_training_dataset_file(string_token, "label", "all", min_sample, sample)
        auto_tuples_file = create_training_dataset_file(string_token, "auto", "all", min_sample, sample)
        if random_props:
            if int(sample)==500 or int(sample)==100:
                random_props_training_files = create_training_dataset_files_random_props(string_token, "label", "all", min_sample, sample, dictio_random_props.keys())
            else:
                random_props = False
                print("WARNING no trainings files with random props are created, because it is only for sample=100 and sample=500")
    elif template == "LAMA":
        #random datapoints only for LAMA templtes
        LAMA_training_file = create_training_dataset_file(string_token, "LAMA", "alone", min_sample, sample)
        label_training_file = None
        auto_tuples_file = None
        if random_props:
            random_props = False
            print("WARNING no trainings files with random props are created, because it is only for label templates")
    elif template == "label":
        #random datapoints only for label templtes
        LAMA_training_file = None
        label_training_file = create_training_dataset_file(string_token, "label", "alone", min_sample, sample)
        auto_tuples_file = None
        if random_props:
            if int(sample)==500 or int(sample)==100:
                random_props_training_files = create_training_dataset_files_random_props(string_token, "label", "alone", min_sample, sample, dictio_random_props.keys())
            else:
                random_props = False
                print("WARNING no trainings files with random props are created, because it is only for sample=100 and sample=500")
    else:
        raise("only /all or /label or /LAMA is supported yet")

    dictio_auto_templates_tuples = {}
    for prop in dictio_prop_train_query_answer:
        dictio_auto_templates_tuples[prop] = {}
        for query_type in dictio_prop_train_query_answer[prop]:
            dictio_auto_templates_tuples[prop][query_type] = []
            array_tuple_query_answer = set()
            array_query_answer = dictio_prop_train_query_answer[prop][query_type]
            for query_answer in array_query_answer:
                number_all_queries = number_all_queries + 1
            #get sample of training data
            if sample == "all":
                exit("do not supported anymore, must be a number")
            sample = int(sample)
            if len(array_query_answer) >= sample:
                array_random_query_answer = random.sample(array_query_answer, sample)
            else:
                exit("ERROR min_sample not okay")
            for query_answer in array_random_query_answer:
                number_training_queries = number_training_queries + 1
            
            for dictio in array_random_query_answer:
                if LAMA_training_file:
                    query = dictio["LAMA"]
                    answer = dictio["answer"]
                    json.dump({"prop": prop, "query": query, "answer": answer}, LAMA_training_file)
                    LAMA_training_file.write("\n")
                if label_training_file:
                    query = dictio["label"]
                    answer = dictio["answer"]
                    json.dump({"prop": prop, "query": query, "answer": answer}, label_training_file)
                    label_training_file.write("\n")
                if auto_tuples_file:
                    (subj_label, obj_label) = dictio["tuple"]
                    dictio_auto_templates_tuples[prop][query_type].append((subj_label, obj_label))
                #add triplets to trainig_file with random_props
                if random_props:
                    query = dictio["label"]
                    for percent in dictio_random_props.keys():
                        for i in range(len(dictio_random_props[percent])):
                            if prop in dictio_random_props[percent][i]:
                                json.dump({"prop": prop, "query": query, "answer": answer}, random_props_training_files[percent][i])
                                random_props_training_files[percent][i].write("\n")


    if auto_tuples_file:
        json.dump(dictio_auto_templates_tuples, auto_tuples_file) 
    #final
    print("#all", number_all_queries)
    print("#training", number_training_queries)