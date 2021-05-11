from transformers import pipeline, BertForMaskedLM
import json
import os
import argparse

def read_dataset(dataset_path):
    dataset = []
    dataset_file = open(dataset_path, "r")
    for line in dataset_file:
        dictio_query_answer = json.loads(line)
        dataset.append(dictio_query_answer)
    return dataset

#for queries
def write_to_dictio_results_queries(lm_name, count_results, unmasker, dictio_query_answer, dictio_prop_template, dictio_id_label, dictio_results, template, pers_pronoun, sample, min_sample):
    props_with_no_templates = set()
    for query in dictio_query_answer:
        triple = query.split("_")
        prop = triple[1]
        if template not in dictio_prop_template[prop]:
            props_with_no_templates.add(prop)
    if len(props_with_no_templates) > 0:
        exit("Some props has no {} templates: {}".format(template, props_with_no_templates))
    else:
        for i, query in enumerate(dictio_query_answer):
            triple = query.split("_")
            subj = triple[0]
            prop = triple[1]
            obj = triple[2]
            if prop not in dictio_results:
                dictio_results[prop] = {}
            if query not in dictio_results[prop]:
                dictio_results[prop][query] = {}
            if subj == "?":
                obj_label = dictio_id_label[obj][0]
                if template == "auto":
                    subject_query = (dictio_prop_template[prop][template][min_sample][sample].replace("[S]", "[MASK]")).replace("[O]", obj_label)
                else:
                    subject_query = (dictio_prop_template[prop][template].replace("[S]", "[MASK]")).replace("[O]", obj_label)
                lm_answers = unmasker(subject_query)
                if i == 0:
                    print("test query for {} templates: {}, triple: {}".format(template, subject_query, triple))
            elif obj == "?":
                subj_label = dictio_id_label[subj][0]
                if template == "auto":
                    object_query = (dictio_prop_template[prop][template][min_sample][sample].replace("[S]", subj_label.capitalize())).replace("[O]", "[MASK]")
                else:
                    object_query = (dictio_prop_template[prop][template].replace("[S]", subj_label.capitalize())).replace("[O]", "[MASK]")
                lm_answers = unmasker(object_query)
                if i == 0:
                    print("test query for {} templates: {}, triple: {}".format(template, object_query, triple))
            else:
                exit("something wrong with the query {}".format(query))
            
            #filter the lm answers to exclude personal pronouns if pers_pronoun == false
            if not pers_pronoun:
                personal_pronouns = ["I", "you", "she", "he", "it", "we", "they", "me", "you", "her", "him", "it", "us", "them"]
                lm_answers_copy = lm_answers.copy()
                for data in lm_answers_copy:
                    lm_answer = data["token_str"]
                    if lm_answer.lower() in personal_pronouns:
                        lm_answers.remove(data)
                
            #get correct answers
            correct_answers = dictio_query_answer[query]
            correct_answers_label = set()
            for id in correct_answers:
                label = dictio_id_label[id][0]
                if subj == "?":
                    correct_answers_label.add(label.capitalize())
                else:
                    correct_answers_label.add(label)
            if "correct" not in dictio_results[prop][query]:
                dictio_results[prop][query]["correct"] = list(correct_answers_label)
            
            #dictio to store results of current lm to the current query 
            dictio_results[prop][query][lm_name] = {}

            if pers_pronoun and "finetuned" in lm_name:
                #find index and score of finetuned answer of current query where first normal answer is a personal pronoun
                normal_lm_name = lm_name.replace("finetuned", "normal")
                first_answer_normal = dictio_results[prop][query][normal_lm_name]["all_answers"][0][0]
                personal_pronouns = ["I", "you", "she", "he", "it", "we", "they", "me", "you", "her", "him", "it", "us", "them"]
                if first_answer_normal.lower() in personal_pronouns:
                    for index, data in enumerate(lm_answers):
                        lm_answer = data["token_str"]
                        score = data["score"]
                        if lm_answer.lower() == first_answer_normal.lower():
                            dictio_results[prop][query][lm_name]["personal_pronoun_index"] = [index, score]
                            break
            
            dictio_results[prop][query][lm_name]["all_answers"] = []
            dictio_results[prop][query][lm_name]["correct_answers_indices"] = []
            for index, data in enumerate(lm_answers[:count_results]):
                lm_answer = data["token_str"]
                score = data["score"]
                if lm_answer in correct_answers_label:
                    dictio_results[prop][query][lm_name]["correct_answers_indices"].append(index)
                #save all results of the lm for the currect query
                dictio_results[prop][query][lm_name]["all_answers"].append([lm_answer, score])
    del unmasker
    return dictio_results

#for training data
def write_to_dictio_results_train_data(lm_name, count_results, unmasker, training_dataset, dictio_prop_template, dictio_id_label, dictio_results):
    for dictio_query_answer in training_dataset:
        prop = dictio_query_answer["prop"]
        if prop not in dictio_results:
            dictio_results[prop] = {}
        if lm_name not in dictio_results[prop]:
            dictio_results[prop][lm_name] = []

        current_result = {}
        query = dictio_query_answer["query"]
        current_result["query"] = query
        
        lm_answers = unmasker(query)
        current_result["correct"] = dictio_query_answer["answer"]
        
        current_result["all_answers"] = []
        current_result["correct_answers_indices"] = []
        for i, data in enumerate(lm_answers[:count_results]):
            lm_answer = data["token_str"]
            score = data["score"]
            if lm_answer == current_result["correct"]:
                current_result["correct_answers_indices"].append(i)
            #save all results of the lm for the currect query
            current_result["all_answers"].append([lm_answer, score])
        dictio_results[prop][lm_name].append(current_result)
    del unmasker
    return dictio_results

def get_results_of_queries(lm_name, epoch, template, alone, min_sample, sample, pers_pronoun, string_token, perc_prop):
    #results of the queries
    dictio_query_answer = json.load(open("/data/fichtel/projektarbeit/queries_{}_{}.json".format(string_token, min_sample), "r"))
    dictio_prop_template = json.load(open("/data/fichtel/projektarbeit/templates.json", "r"))
    dictio_id_label = json.load(open("/data/fichtel/projektarbeit/entity2label_onlyrdflabel.json", "r"))
    print("#queries: {}".format(len(dictio_query_answer)))
    #how many results of the lm should be ovserved
    count_results = 5
    #final dictio with the results of each lm for each query    
    dictio_results = {}
    #lm which should be evaluated
    unmasker_normal = pipeline('fill-mask', model=lm_name, device=0, top_k=1000)
    dictio_results = write_to_dictio_results_queries(lm_name+"-normal", count_results,  unmasker_normal, dictio_query_answer, dictio_prop_template, dictio_id_label, dictio_results, template, pers_pronoun, sample, min_sample)
    if template != "auto":
        finetuned_lm_path = "/data/fichtel/projektarbeit/bert_base_cased_finetuned_{}_{}_{}_{}{}_{}_{}".format(perc_prop, string_token, epoch, template, alone, min_sample, sample)
        unmasker_finetuned = pipeline('fill-mask', tokenizer= lm_name, model = BertForMaskedLM.from_pretrained(finetuned_lm_path), device=0, top_k=1000)
        dictio_results = write_to_dictio_results_queries(lm_name+"-finetuned", count_results, unmasker_finetuned, dictio_query_answer, dictio_prop_template, dictio_id_label, dictio_results, template, pers_pronoun, sample, min_sample) 
    return dictio_results

def get_results_of_training_data(lm_name, epoch, template, alone, min_sample, sample, string_token, perc_prop):
    #results of the training data
    #read datasets from path
    training_dataset = read_dataset("/data/fichtel/projektarbeit/training_datasets/training_dataset_{}_{}_{}{}_{}_{}.json".format(perc_prop, string_token, template, alone, min_sample, sample))
    dictio_prop_template = json.load(open("/data/fichtel/projektarbeit/templates.json", "r"))
    dictio_id_label = json.load(open("/data/fichtel/projektarbeit/entity2label_onlyrdflabel.json", "r"))
    print("#train queries: {}".format(len(training_dataset)))
    #how many results of the lm should be ovserved
    count_results = 5
    #final dictio with the results of each lm for each query    
    dictio_results = {}
    #lm which should be evaluated
    unmasker_normal = pipeline('fill-mask', model=lm_name, device=0)
    dictio_results = write_to_dictio_results_train_data(lm_name+"-normal", count_results,  unmasker_normal, training_dataset, dictio_prop_template, dictio_id_label, dictio_results)
    
    finetuned_lm_path = "/data/fichtel/projektarbeit/bert_base_cased_finetuned_{}_{}_{}_{}{}_{}_{}".format(perc_prop, string_token, epoch, template, alone, min_sample, sample)
    unmasker_finetuned = pipeline('fill-mask', tokenizer= lm_name, model = BertForMaskedLM.from_pretrained(finetuned_lm_path), device=0)
    dictio_results = write_to_dictio_results_train_data(lm_name+"-finetuned", count_results, unmasker_finetuned, training_dataset, dictio_prop_template, dictio_id_label, dictio_results) 
    return dictio_results

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    
    #used LM
    lm_name = 'bert-base-cased'

    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-queries',action="store_true", default=False, help="set flag if queries should be evaluated")
    parser.add_argument('-train', action="store_true", default=False, help="set flag if training data should be evaluated")
    parser.add_argument('-min_sample', help="set how many triple at least should exist of each property in wikidata_onetoken_missing")
    parser.add_argument('-sample', help="set how many triple should be used of each property (e.g. 10000)")
    parser.add_argument('-epoch', help="set how many epoches should be executed")
    parser.add_argument('-template', help="set which template should be used (LAMA or label or auto)")
    parser.add_argument('-alone',action="store_true", default=False, help="set flag if training data should be used which is only for one template")
    parser.add_argument('-pers_pronoun',action="store_true", default=False, help="set flag if personal pronouns should be analyzed")
    parser.add_argument('-string_token', help="set if obj and subj labels should consist of only one word (oneword) and are also in vocab file (onetoken)")
    parser.add_argument('-perc_prop', help="set how many props should be used for training (e.g. 100 for all props or 90-0 for first random_prop selection with 90% of the props)")
    
    args = parser.parse_args()
    epoch = args.epoch
    template = args.template
    alone = args.alone
    if alone:
        alone = "alone"
    else:
        alone = ""
    pers_pronoun = args.pers_pronoun
    if pers_pronoun:
        pers_pronoun_string = "_p"
    else:
        pers_pronoun_string = ""
    sample = args.sample
    min_sample = args.min_sample
    if int(sample) > int(min_sample):
        exit("ERROR: the sample size cannot be bigger than the min_sample size")
    string_token = args.string_token
    assert(string_token in ["onetoken", "oneword", ""])
    perc_prop = args.perc_prop
    if perc_prop != "100" and not (int(sample)==500 or int(sample)==100 or template=="LAMA"):
        exit("ERROR training on less than 100% of the props is only possible with sample=500 or sample=100 and template=label")
    
    print(args)
    if args.queries:
        dictio_results = get_results_of_queries(lm_name, epoch, template, alone, min_sample, sample, pers_pronoun, string_token, perc_prop)
        if os.path.exists("results/queries/results_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, sample, pers_pronoun_string )):
            os.remove("results/queries/results_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, sample, pers_pronoun_string))
            print("removed results/queries/results_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, sample, pers_pronoun_string))
        result_file = open("results/queries/results_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, sample, pers_pronoun_string), "w")
        json.dump(dictio_results, result_file)
        result_file.close()
        print("queries ready")
    if args.train:
        if template == "auto":
            exit("BERT has not be trained with auto templates, so the training data memorization cannot be examined")
        dictio_results = get_results_of_training_data(lm_name, epoch, template, alone, min_sample, sample, string_token, perc_prop)
        if os.path.exists("results/train_data/results_train_queries_{}_{}_{}_{}{}_{}_{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, sample)):
            os.remove("results/train_data/results_train_queries_{}_{}_{}_{}{}_{}_{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, sample))
            print("removed results/train_data/results_train_queries_{}_{}_{}_{}{}_{}_{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, sample))
        result_file = open("results/train_data/results_train_queries_{}_{}_{}_{}{}_{}_{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, sample), "w")
        json.dump(dictio_results, result_file)
        result_file.close()
        print("train queries ready")
