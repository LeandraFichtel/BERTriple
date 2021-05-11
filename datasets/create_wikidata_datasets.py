import json
import random
import os
from os import path
from transformers import BertTokenizer
import shutil

def check_tokenizer(subj_label, obj_label, tokenizer, string_token):
    tokenized_subj = tokenizer.tokenize(subj_label)
    tokenized_obj = tokenizer.tokenize(obj_label)
    #generel check, that the labels can be representend with the tokenizer
    if "[UNK]" not in tokenized_subj and  "[UNK]" not in tokenized_obj:
        if string_token == "":
            #all labels are accepted --> triple can be used as subject and object query
            return "subjobj"
        elif string_token == "oneword":
            #only labels which consists of only one word are accepted
            if len(subj_label.split(" ")) == 1 and len(obj_label.split(" ")) == 1:
                #triple can be used as subject and object query
                return "subjobj"
            elif len(subj_label.split(" ")) == 1 and len(obj_label.split(" ")) > 1:
                #triple can be used only as subject query
                return "subj"
            elif len(subj_label.split(" ")) > 1 and len(obj_label.split(" ")) == 1:
                #triple can be used only as object query
                return "obj"
        elif string_token == "onetoken":
            #only labels which are in vocab file (=one token) and consist of only one word are accepted
            if len(tokenized_subj) == 1 and len(tokenized_obj) == 1:
                #triple can be used as subject and object query
                return "subjobj"
            elif len(tokenized_subj) == 1 and len(tokenized_obj) > 1:
                #triple can be used only as subject query
                return "subj"
            elif len(tokenized_subj) > 1 and len(tokenized_obj) == 1:
                #triple can be used only as object query
                return "obj"    
    #print(subj_label, tokenized_subj, obj_label, tokenized_obj)
    return None


def get_dictio_all_triple(wikidata_all_triplets, dictio_prop_template, dictio_id_label, tokenizer,string_token):
    dictio_prop_triple = {}
    for line in wikidata_all_triplets:
        triple = ((line.replace("\n", "")).replace(".", "")).split(" ")
        prop = str(triple[1]).split('/')[-1].replace('>', "")
        if prop in dictio_prop_template:
            subj = str(triple[0]).split('/')[-1].replace('>', "")
            obj = str(triple[2]).split('/')[-1].replace('>', "")
            if subj in dictio_id_label and obj in dictio_id_label:
                subj_label = dictio_id_label[subj][0]
                obj_label = dictio_id_label[obj][0]
                query_type = check_tokenizer(subj_label, obj_label, tokenizer, string_token)
                if query_type:
                    if prop not in dictio_prop_triple:
                        dictio_prop_triple[prop] = {}
                        dictio_prop_triple[prop]["subjobj"] = []
                        dictio_prop_triple[prop]["subj"] = []
                        dictio_prop_triple[prop]["obj"] = []
                    dictio_prop_triple[prop][query_type].append((subj, obj))
            else:
                print("Obj or Subj Id not found", subj, obj)
    return dictio_prop_triple 
                    
def get_random_queries(dictio_prop_triple):
    dictio_prop_query_answer = {}
    dictio_prop_triple_missing = {}
    for prop in dictio_prop_triple:
        print(prop)
        dictio_prop_query_answer[prop] = {}
        #get 50 subject queries
        triplets_subject = set(dictio_prop_triple[prop]["subjobj"].copy())
        triplets_subject = triplets_subject.union(dictio_prop_triple[prop]["subj"].copy())
        triplets_subject_all = set(dictio_prop_triple[prop]["subjobj"].copy())
        triplets_subject_all = triplets_subject_all.union(dictio_prop_triple[prop]["subj"].copy())
        count = 0
        while triplets_subject != set() and count < 50:
            #get random triple
            (subj, obj) = random.sample(triplets_subject, 1)[0]
            query = "?_{}_{}".format(prop, obj)
            if query not in dictio_prop_query_answer[prop]:
                count = count + 1
                triplets_subject.remove((subj, obj))
                triplets_subject_all.remove((subj, obj))
                dictio_prop_query_answer[prop][query] = []
                dictio_prop_query_answer[prop][query].append(subj)
                triplets_copy = triplets_subject.copy()
                #find all answers to the query
                for (actu_subj, actu_obj) in triplets_copy:
                    if actu_obj == obj:
                        dictio_prop_query_answer[prop][query].append(actu_subj)
                        triplets_subject.remove((actu_subj, actu_obj))
                        triplets_subject_all.remove((actu_subj, actu_obj))
                #check if there are more than 100 answers --> randomly choose only 100
                answers = dictio_prop_query_answer[prop][query]
                if len(answers) > 100:
                    sample_answers = random.sample(answers, 100)
                    dictio_prop_query_answer[prop][query] = sample_answers
                    for subj in (set(answers) - set(sample_answers)):
                        triplets_subject_all.add((subj, obj))
                assert(len(triplets_subject_all) >= len(triplets_subject))

        #get 50 object queries
        triplets_object = set(dictio_prop_triple[prop]["subjobj"].copy())
        triplets_object = triplets_object.union(dictio_prop_triple[prop]["obj"].copy())
        triplets_object_all = set(dictio_prop_triple[prop]["subjobj"].copy())
        triplets_object_all = triplets_object_all.union(dictio_prop_triple[prop]["obj"].copy())
        count = 0
        while triplets_object != set() and count < 50:
            #get random triple
            (subj, obj) = random.sample(triplets_object, 1)[0]
            query = "{}_{}_?".format(subj, prop)
            if query not in dictio_prop_query_answer[prop]:
                count = count + 1
                triplets_object.remove((subj, obj))
                triplets_object_all.remove((subj, obj))
                dictio_prop_query_answer[prop][query] = []
                dictio_prop_query_answer[prop][query].append(obj)
                triplets_copy = triplets_object.copy()
                #find all answers to the query
                for (actu_subj, actu_obj) in triplets_copy:
                    if actu_subj == subj:
                        dictio_prop_query_answer[prop][query].append(actu_obj)
                        triplets_object.remove((actu_subj, actu_obj))
                        triplets_object_all.remove((actu_subj, actu_obj))
                #check if there are more than 100 answers --> randomly choose only 100
                answers = dictio_prop_query_answer[prop][query]
                if len(answers) > 100:
                    sample_answers = random.sample(answers, 100)
                    dictio_prop_query_answer[prop][query] = sample_answers
                    for obj in (set(answers) - set(sample_answers)):
                        triplets_object_all.add((subj, obj))
                assert(len(triplets_object_all) >= len(triplets_object))
        
        #get training data == not_used_triple
        subjobj_not_used_triple = triplets_subject_all.intersection(triplets_object_all)
        subj_not_used_triple = triplets_subject_all.intersection(set(dictio_prop_triple[prop]["subj"]))
        obj_not_used_triple = triplets_object_all.intersection(set(dictio_prop_triple[prop]["obj"]))
        
        #check that there are no query triplets in training data(== not_used_triple)
        for query in dictio_prop_query_answer[prop]:
            answers = dictio_prop_query_answer[prop][query]
            for answer in answers:
                triple = query.replace("?", answer).split("_")
                subj = triple[0]
                obj = triple[2]
                if (subj, obj) in subjobj_not_used_triple or (subj, obj) in subj_not_used_triple or (subj, obj) in obj_not_used_triple:
                    raise("ERROR query triple in training data")
        dictio_prop_triple_missing[prop] = {}
        dictio_prop_triple_missing[prop]["subjobj"] = list(subjobj_not_used_triple)
        dictio_prop_triple_missing[prop]["subj"] = list(subj_not_used_triple)
        dictio_prop_triple_missing[prop]["obj"] = list(obj_not_used_triple)
    return dictio_prop_query_answer, dictio_prop_triple_missing

def get_training_dataset(dictio_prop_triple_missing):
    dictio_prop_template = json.load(open("/data/fichtel/projektarbeit/templates.json", "r"))
    dictio_id_label = json.load(open("/data/fichtel/projektarbeit/entity2label_onlyrdflabel.json", "r"))
    dictio_prop_train_query_answer = {}
    for prop in dictio_prop_triple_missing:
        if prop not in dictio_prop_train_query_answer:
            dictio_prop_train_query_answer[prop] = {"subj": [], "obj": []}
        for query_type in dictio_prop_triple_missing[prop]:
            for (subj, obj) in dictio_prop_triple_missing[prop][query_type]:
                if subj in dictio_id_label and obj in dictio_id_label:
                    subj_label = dictio_id_label[subj][0]
                    obj_label = dictio_id_label[obj][0]
                    if query_type == "subjobj":
                        #each triple can be used as two datapoints in training --> subject and object query
                        LAMA_subject_query = (dictio_prop_template[prop]["LAMA"].replace("[S]", "[MASK]")).replace("[O]", obj_label)
                        label_subject_query = (dictio_prop_template[prop]["label"].replace("[S]", "[MASK]")).replace("[O]", obj_label)
                        dictio_prop_train_query_answer[prop]["subj"].append({"LAMA": LAMA_subject_query, "label": label_subject_query, "answer": subj_label.capitalize(), "tuple": (subj_label, obj_label)})

                        LAMA_object_query = (dictio_prop_template[prop]["LAMA"].replace("[S]", subj_label.capitalize())).replace("[O]", "[MASK]")
                        label_object_query = (dictio_prop_template[prop]["label"].replace("[S]", subj_label.capitalize())).replace("[O]", "[MASK]")
                        dictio_prop_train_query_answer[prop]["obj"].append({"LAMA": LAMA_object_query, "label": label_object_query, "answer": obj_label, "tuple": (subj_label, obj_label)})
                    elif query_type == "subj":
                        #each triple can only be used as one datapoints in training --> subject query
                        LAMA_subject_query = (dictio_prop_template[prop]["LAMA"].replace("[S]", "[MASK]")).replace("[O]", obj_label)
                        label_subject_query = (dictio_prop_template[prop]["label"].replace("[S]", "[MASK]")).replace("[O]", obj_label)
                        dictio_prop_train_query_answer[prop]["subj"].append({"LAMA": LAMA_subject_query, "label": label_subject_query, "answer": subj_label.capitalize(), "tuple": (subj_label, obj_label)})
                    elif query_type == "obj":
                        #each triple can only be used as one datapoints in training --> object query
                        LAMA_object_query = (dictio_prop_template[prop]["LAMA"].replace("[S]", subj_label.capitalize())).replace("[O]", "[MASK]")
                        label_object_query = (dictio_prop_template[prop]["label"].replace("[S]", subj_label.capitalize())).replace("[O]", "[MASK]")
                        dictio_prop_train_query_answer[prop]["obj"].append({"LAMA": LAMA_object_query, "label": label_object_query, "answer": obj_label, "tuple": (subj_label, obj_label)})
                    else:
                        print("query_type not okay", query_type)
                else:
                    print("Obj or Subj Id not found", subj, obj)
    
    #sometimes two IDs are mapped to the same label, but there should not be the same datapoint in training multiple times
    for prop in dictio_prop_train_query_answer:
        dictio_prop_train_query_answer[prop]["subj"] = list({frozenset(item.items()) : item for item in dictio_prop_train_query_answer[prop]["subj"]}.values())
        dictio_prop_train_query_answer[prop]["obj"] = list({frozenset(item.items()) : item for item in dictio_prop_train_query_answer[prop]["obj"]}.values())
    return dictio_prop_train_query_answer



if __name__ == "__main__":
    #set how many triplets at least should exist of each property in wikidata_onetoken_missing (e.g. min_sample=3000 --> 3000 triplets for subj queries and 3000 triplets for object queries)
    min_sample = "500"
    print("min sample", min_sample)
    #set string_token to "onetoken", if only triplets should be accepted where all subj-labels and obj-labels are in vocab file (=one token) and consist of only one word (=Berlin and !=Los Angeles)
    #set string_token to "oneword", if only triplets should be accepted where all subj-labels and obj-labels consist of only one word (=Berlin and !=Los Angeles)
    ##set string_token to "", if triplets can be multi token in both ways
    string_token = "onetoken"
    assert(string_token in ["onetoken", "oneword", ""])
    print("string_token", string_token)
    #files for all triple which are onetoken
    if not path.exists("/data/fichtel/projektarbeit/wikidata_{}_all.json".format(string_token)):
        print("create files")
        wikidata_all_triplets = open("/data/fichtel/projektarbeit/gold_dataset.nt", "r")
        dictio_prop_template = json.load(open("/data/fichtel/projektarbeit/templates.json", "r"))
        dictio_id_label = json.load(open("/data/fichtel/projektarbeit/entity2label_onlyrdflabel.json", "r"))
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        
        dictio_prop_triple = get_dictio_all_triple(wikidata_all_triplets, dictio_prop_template, dictio_id_label, tokenizer, string_token)
        wikidata_onetoken_all = open("/data/fichtel/projektarbeit/wikidata_{}_all.json".format(string_token), "w")
        json.dump(dictio_prop_triple, wikidata_onetoken_all)
    
        dictio_prop_query_answer, dictio_prop_triple_missing = get_random_queries(dictio_prop_triple)
        if path.exists("/data/fichtel/projektarbeit/queries_{}_all.json".format(string_token)):
            print("removed queries_{}_all.json".format(string_token))
            os.remove("/data/fichtel/projektarbeit/queries_{}_all.json".format(string_token))
        queries_all = open("/data/fichtel/projektarbeit/queries_{}_all.json".format(string_token), "w")
        json.dump(dictio_prop_query_answer, queries_all)

        #file for all triple, which are not used in the queries and which are later used for the training dataset
        if path.exists("/data/fichtel/projektarbeit/wikidata_{}_missing_all.json".format(string_token)):
            print("removed wikidata_{}_missing_all.json".format(string_token))
            os.remove("/data/fichtel/projektarbeit/wikidata_{}_missing_all.json".format(string_token))
        wikidata_onetoken_missing_all = open("/data/fichtel/projektarbeit/wikidata_{}_missing_all.json".format(string_token), "w")
        json.dump(dictio_prop_triple_missing, wikidata_onetoken_missing_all)
    else:
        print("load existing files")
        #dictio_prop_triple = json.load(open("/data/fichtel/projektarbeit/wikidata_{}_all.json".format(string_token))
        dictio_prop_query_answer = json.load(open("/data/fichtel/projektarbeit/queries_{}_all.json".format(string_token), "r"))
        dictio_prop_triple_missing = json.load(open("/data/fichtel/projektarbeit/wikidata_{}_missing_all.json".format(string_token), "r"))
    
    #file for the queries based on the wikidata_onetoken_all.nt file
    if path.exists("/data/fichtel/projektarbeit/queries_{}_{}.json".format(string_token, min_sample)):
        print("removed queries_{}_{}.json".format(string_token, min_sample))
        os.remove("/data/fichtel/projektarbeit/queries_{}_{}.json".format(string_token, min_sample))
    queries = open("/data/fichtel/projektarbeit/queries_{}_{}.json".format(string_token, min_sample), "w")
    if os.path.exists("/data/fichtel/projektarbeit/dataset_{}_{}.json".format(string_token, min_sample)):
        print("removed dataset_{}_{}.json".format(string_token, min_sample))
        os.remove("/data/fichtel/projektarbeit/dataset_{}_{}.json".format(string_token, min_sample))
    dataset_file = open("/data/fichtel/projektarbeit/dataset_{}_{}.json".format(string_token, min_sample), "w")
    #create training_dataset directory and delete the files in it, because a new dataset base (e.g. dataset_onetoken_6000.json) with new queries was created
    if not os.path.exists("/data/fichtel/projektarbeit/training_datasets/"):
        os.mkdir("/data/fichtel/projektarbeit/training_datasets/")
    else:
        for file in os.listdir("/data/fichtel/projektarbeit/training_datasets/"):
            if "{}_".format(min_sample) in file:
                os.remove(os.path.join("/data/fichtel/projektarbeit/training_datasets/", file))
                print("removed {}".format(file))
    #delete the model directories, because a new dataset base (e.g. dataset_onetoken_6000.json) with new queries was created
    for file in os.listdir("/data/fichtel/projektarbeit/"):
        if "bert_base_cased_finetuned" in file and "{}_".format(min_sample) in file:
            shutil.rmtree(os.path.join("/data/fichtel/projektarbeit/", file))
            print("removed {}".format(file))

    #get the training dataset with template and labels and not IDs anymore
    dictio_prop_train_query_answer = get_training_dataset(dictio_prop_triple_missing)
    #get all props which fullfill the min_sample condition (triplets can be used as max two datapoints in training --> object und subject query)
    valid_props = {}
    for prop in dictio_prop_train_query_answer:
        if len(dictio_prop_train_query_answer[prop]["subj"]) >= int(min_sample) and len(dictio_prop_train_query_answer[prop]["obj"]) >= int(min_sample):
            valid_props[prop] = [len(dictio_prop_train_query_answer[prop]["subj"]), len(dictio_prop_train_query_answer[prop]["obj"])]
    print("#valid props with min_sample={}: {}/{}".format(min_sample, len(valid_props), len(dictio_prop_triple_missing)))

    #get random props which should be trained to check whether it has an impact on props for which no triplets have been seen during training
    dictio_random_props = {}
    count_valid_props = len(valid_props)
    for percent in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        string_percent = str(int(percent*100))
        dictio_random_props[string_percent] = []
        for _ in range(5):
            random_props = random.sample(valid_props.keys(), int(count_valid_props*percent))
            dictio_random_props[string_percent].append(random_props)

    dictio_query_answer = {}
    print("prop", "#queries", "#training triplets (subj, obj)")
    dictio_prop_train_query_answer_final = {}
    dictio_prop_train_query_answer_final["random_props"] = dictio_random_props
    for prop in valid_props:
        #create queries dictio
        print(prop, len(dictio_prop_query_answer[prop]), valid_props[prop])
        for query in dictio_prop_query_answer[prop]:
            dictio_query_answer[query] = dictio_prop_query_answer[prop][query]
        #save training data only for valid props
        dictio_prop_train_query_answer_final[prop] = dictio_prop_train_query_answer[prop]
    #write query/answer dict into queries file
    json.dump(dictio_query_answer, queries)
    #write train query/answer dict into dataset file
    json.dump(dictio_prop_train_query_answer_final, dataset_file)