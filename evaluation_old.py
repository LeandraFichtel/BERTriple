import json
import argparse
import os

def read_dataset(dataset_path):
    dataset = []
    dataset_file = open(dataset_path, "r")
    for line in dataset_file:
        dictio_query_answer = json.loads(line)
        dataset.append(dictio_query_answer)
    return dataset

#function to get only the answers of the indices (=correct answers) or to get the answers which are not in the indices (=incorrect answers)
def get_answers(string_correct_incorrect, all_answers, indices):
    if string_correct_incorrect == "correct":
        answers = []
        for index in indices:
            if index < len(all_answers):
                answers.append(all_answers[index])
        return answers
    else:
        all_answers_copy = all_answers.copy()
        for index in indices:
            if index < len(all_answers):
                all_answers_copy.remove(all_answers[index])
        return all_answers_copy


#function to get a dictio_results with all answers and evaluations of the two LMs
def get_results(dictio_all_results, training_dataset, count_results, string_token, perc_prop, min_sample):
    dictio_results = {"correct": {}, "incorrect": {}, "subjobj": {}, "subject": {}, "object": {}, "props": []}
    dictio_results["correct"]["subjobj"] = {"normal": {}, "finetuned": {}}
    dictio_results["correct"]["subject"] = {"normal": {}, "finetuned": {}}
    dictio_results["correct"]["object"] = {"normal": {}, "finetuned": {}}
    dictio_results["incorrect"]["subjobj"] = {"normal": {}, "finetuned": {}}
    dictio_results["incorrect"]["object"] = {"normal": {}, "finetuned": {}}
    dictio_results["incorrect"]["subject"] = {"normal": {}, "finetuned": {}}
    dictio_results["subjobj"] = {"normal": {}, "finetuned": {}}
    dictio_results["subject"] = {"normal": {}, "finetuned": {}}
    dictio_results["object"] = {"normal": {}, "finetuned": {}}

    for prop in dictio_all_results:
        dictio_results["props"].append(prop)
        dictio_results["subjobj"]["normal"][prop] = {}
        dictio_results["subject"]["normal"][prop] = {}
        dictio_results["object"]["normal"][prop] = {}
        dictio_results["subjobj"]["finetuned"][prop] = {}
        dictio_results["subject"]["finetuned"][prop] = {}
        dictio_results["object"]["finetuned"][prop] = {}

        for query in dictio_all_results[prop]:
            if query[0] == "?":
                query_type = "subject"
            else:
                query_type = "object"
            #all results
            normal_results = dictio_all_results[prop][query]["bert-base-cased-normal"]
            finetuned_results = dictio_all_results[prop][query]["bert-base-cased-finetuned"]
            normal_all_answers = normal_results["all_answers"][:count_results]
            finetuned_all_answers = finetuned_results["all_answers"][:count_results]
            #correct results
            normal_correct_answers = {a[0] for a in get_answers("correct", normal_all_answers, normal_results["correct_answers_indices"])}
            finetuned_correct_answers = {a[0] for a in get_answers("correct", finetuned_all_answers, finetuned_results["correct_answers_indices"])}
            
            #group in subject-object queries, subject queries and object queries
            
            dictio_results["subjobj"]["normal"][prop][query] = normal_results
            dictio_results[query_type]["normal"][prop][query] = normal_results
            dictio_results["subjobj"]["finetuned"][prop][query] = finetuned_results
            dictio_results[query_type]["finetuned"][prop][query] = finetuned_results
            #group in correct and incorrect queries
            if len(normal_correct_answers) > 0:
                dictio_results["correct"]["subjobj"]["normal"][query] = get_answers("correct", normal_all_answers, normal_results["correct_answers_indices"])
                dictio_results["correct"][query_type]["normal"][query] = get_answers("correct", normal_all_answers, normal_results["correct_answers_indices"])
            else:
                dictio_results["incorrect"]["subjobj"]["normal"][query] = get_answers("incorrect", normal_all_answers, normal_results["correct_answers_indices"])
                dictio_results["incorrect"][query_type]["normal"][query] = get_answers("incorrect", normal_all_answers, normal_results["correct_answers_indices"])
            if len(finetuned_correct_answers) > 0:
                dictio_results["correct"]["subjobj"]["finetuned"][query] = get_answers("correct", finetuned_all_answers, finetuned_results["correct_answers_indices"])
                dictio_results["correct"][query_type]["finetuned"][query] = get_answers("correct", finetuned_all_answers, finetuned_results["correct_answers_indices"])
            else:
                dictio_results["incorrect"]["subjobj"]["finetuned"][query] = get_answers("incorrect", finetuned_all_answers, finetuned_results["correct_answers_indices"])
                dictio_results["incorrect"][query_type]["finetuned"][query] = get_answers("incorrect", finetuned_all_answers, finetuned_results["correct_answers_indices"])
    
    #calculate average probability
    for query_type in ["subjobj", "subject", "object"]:
        normal_probabilities_correct, finetuned_probabilities_correct = get_probability(dictio_results["correct"][query_type], count_results)
        normal_probabilities_incorrect, finetuned_probabilities_incorrect = get_probability(dictio_results["incorrect"][query_type], count_results)
        dictio_results["correct"][query_type]["probability@{}".format(count_results)] = {"normal": normal_probabilities_correct, "finetuned": finetuned_probabilities_correct}
        dictio_results["incorrect"][query_type]["probability@{}".format(count_results)] = {"normal": normal_probabilities_incorrect, "finetuned": finetuned_probabilities_incorrect}
    
    #calculate estimated calibration error (ECE)
    for query_type in ["subjobj", "subject", "object"]:
        ece_normal = get_ece(dictio_results[query_type]["normal"], string_token, min_sample)
        ece_finetuned = get_ece(dictio_results[query_type]["finetuned"], string_token, min_sample)
        dictio_results[query_type]["ece"] = {"normal": ece_normal, "finetuned": ece_finetuned}

    #caculate mean precision at 1
    for query_type in ["subjobj", "subject", "object"]:
        mean_precision_at_1_normal, dictio_prop_mean_precision_normal = get_mean_prec_at_1(dictio_results[query_type]["normal"])
        mean_precision_at_1_finetuned, dictio_prop_mean_precision_finetuned = get_mean_prec_at_1(dictio_results[query_type]["finetuned"])
        dictio_results[query_type]["precision@{}".format(count_results)] = {}
        dictio_results[query_type]["precision@{}".format(count_results)]["overall"] = {"normal": mean_precision_at_1_normal, "finetuned": mean_precision_at_1_finetuned}
        dictio_results[query_type]["precision@{}".format(count_results)]["per_prop"] = {"normal": dictio_prop_mean_precision_normal, "finetuned": dictio_prop_mean_precision_finetuned}

    for query_type in ["subject", "object"]:
        mean_prec_bups_normal, dictio_prop_mean_precision_normal = get_mean_precision_bups(dictio_results[query_type]["normal"], query_type, string_token, min_sample)
        mean_prec_bups_finetuned, dictio_prop_mean_precision_finetuned = get_mean_precision_bups(dictio_results[query_type]["finetuned"], query_type, string_token, min_sample)
        dictio_results[query_type]["precision_bups"] = {}
        dictio_results[query_type]["precision_bups"]["overall"] = {"normal": mean_prec_bups_normal, "finetuned": mean_prec_bups_finetuned}
        dictio_results[query_type]["precision_bups"]["per_prop"] = {"normal": dictio_prop_mean_precision_normal, "finetuned": dictio_prop_mean_precision_finetuned}
    mean_prec_bups_normal = (dictio_results["subject"]["precision_bups"]["overall"]["normal"] + dictio_results["object"]["precision_bups"]["overall"]["normal"]) / 2
    mean_prec_bups_finetuned = (dictio_results["subject"]["precision_bups"]["overall"]["finetuned"] + dictio_results["object"]["precision_bups"]["overall"]["finetuned"]) / 2
    dictio_results["subjobj"]["precision_bups"] = {}
    dictio_results["subjobj"]["precision_bups"]["overall"] =  {"normal": mean_prec_bups_normal, "finetuned": mean_prec_bups_finetuned}
    
    #count queries which are answered with answers which were already seen during training
    count_queries, count_all_queries, new_answers = get_count_queries_already_seen_in_training(dictio_all_results, training_dataset)
    dictio_results["seen_in_training"] = {"count_queries": count_queries, "count_all_queries": count_all_queries, "new_answers": new_answers}
    
    #count queries which are answered by the finetuned lm correctly with LAMA *and* label template
    count_queries, count_correct_queries_LAMA, count_correct_queries_label = get_count_same_query_correct_answered(epoch, alone, min_sample, sample, string_token, perc_prop)
    dictio_results["correct_both_templates"] = {"count_queries": count_queries, "count_correct_queries_LAMA": count_correct_queries_LAMA, "count_correct_queries_label": count_correct_queries_label}
    return dictio_results

def get_results_auto_template(dictio_all_results, count_results, string_token, perc_prop):
    dictio_results = {"correct": {}, "incorrect": {}, "subjobj": {}, "subject": {}, "object": {}, "props": []}
    dictio_results["correct"]["subjobj"] = {"normal": {}}
    dictio_results["correct"]["subject"] = {"normal": {}}
    dictio_results["correct"]["object"] = {"normal": {}}
    dictio_results["incorrect"]["subjobj"] = {"normal": {}}
    dictio_results["incorrect"]["object"] = {"normal": {}}
    dictio_results["incorrect"]["subject"] = {"normal": {}}
    dictio_results["subjobj"] = {"normal": {}}
    dictio_results["subject"] = {"normal": {}}
    dictio_results["object"] = {"normal": {}}
    for prop in dictio_all_results:
        dictio_results["props"].append(prop)
        dictio_results["subjobj"]["normal"][prop] = {}
        dictio_results["subject"]["normal"][prop] = {}
        dictio_results["object"]["normal"][prop] = {}
               
        for query in dictio_all_results[prop]:
            if query[0] == "?":
                query_type = "subject"
            else:
                query_type = "object"
            #all results
            normal_results = dictio_all_results[prop][query]["bert-base-cased-normal"]
            normal_all_answers = normal_results["all_answers"][:count_results]
            #correct results
            normal_correct_answers = {a[0] for a in get_answers("correct", normal_all_answers, normal_results["correct_answers_indices"])}
            
            #group in subject-object queries, subject queries and object queries
            dictio_results["subjobj"]["normal"][prop][query] = normal_results
            dictio_results[query_type]["normal"][prop][query] = normal_results
            #group in correct and incorrect queries
            if len(normal_correct_answers) > 0:
                dictio_results["correct"]["subjobj"]["normal"][query] = get_answers("correct", normal_all_answers, normal_results["correct_answers_indices"])
                dictio_results["correct"][query_type]["normal"][query] = get_answers("correct", normal_all_answers, normal_results["correct_answers_indices"])
            else:
                dictio_results["incorrect"]["subjobj"]["normal"][query] = get_answers("incorrect", normal_all_answers, normal_results["correct_answers_indices"])
                dictio_results["incorrect"][query_type]["normal"][query] = get_answers("incorrect", normal_all_answers, normal_results["correct_answers_indices"])

    #caculate mean precision at 1
    for query_type in ["subjobj", "subject", "object"]:
        mean_precision_at_1_normal, dictio_prop_mean_precision_normal = get_mean_prec_at_1(dictio_results[query_type]["normal"])
        dictio_results[query_type]["precision@{}".format(count_results)] = {}
        dictio_results[query_type]["precision@{}".format(count_results)]["overall"] = {"normal": mean_precision_at_1_normal}
        dictio_results[query_type]["precision@{}".format(count_results)]["per_prop"] = {"normal": dictio_prop_mean_precision_normal}

    for query_type in ["subject", "object"]:
        mean_prec_bups_normal, dictio_prop_mean_precision_normal = get_mean_precision_bups(dictio_results[query_type]["normal"], query_type, string_token, min_sample)
        dictio_results[query_type]["precision_bups"] = {}
        dictio_results[query_type]["precision_bups"]["overall"] = {"normal": mean_prec_bups_normal}
        dictio_results[query_type]["precision_bups"]["per_prop"] = {"normal": dictio_prop_mean_precision_normal}
    mean_prec_bups_normal = (dictio_results["subject"]["precision_bups"]["overall"]["normal"] + dictio_results["object"]["precision_bups"]["overall"]["normal"]) / 2
    dictio_results["subjobj"]["precision_bups"] = {}
    dictio_results["subjobj"]["precision_bups"]["overall"] =  {"normal": mean_prec_bups_normal}
    return dictio_results

def get_probability(dictio_results, count_results):
    normal_probabilities = []
    finetuned_probabilities = []
    for query in dictio_results["normal"]:
        results = dictio_results["normal"][query]
        score_normal = 0
        for result in results[:count_results]:
            score_normal = score_normal + result[1]
        score_normal = score_normal / len(results[:count_results])
        normal_probabilities.append(score_normal)
    for query in dictio_results["finetuned"]:
        results = dictio_results["finetuned"][query]
        score_finetuned = 0
        for result in results[:count_results]:
            score_finetuned = score_finetuned + result[1]
        score_finetuned = score_finetuned / len(results[:count_results])
        finetuned_probabilities.append(score_finetuned)
    return normal_probabilities, finetuned_probabilities

def get_mean_prec_at_1(dictio_results):
    #mean precision over all props
    mean_prec_at_1 = 0
    dictio_prop_mean_precision = {}
    for prop in dictio_results:
        #precision of each prop over all queries
        precision_at_1 = 0
        for query in dictio_results[prop]:
            #correct_answers_indices to current query and prop
            indices = dictio_results[prop][query]["correct_answers_indices"]
            if len(indices) > 0 and 0 in indices:
                precision_at_1 = precision_at_1 + 1
        count_queries = len(dictio_results[prop])
        precision_at_1 = precision_at_1/count_queries 
        dictio_prop_mean_precision[prop] = precision_at_1
        mean_prec_at_1 = mean_prec_at_1 + precision_at_1
    count_props = len(dictio_results)
    mean_prec_at_1 = mean_prec_at_1/count_props
    return mean_prec_at_1, dictio_prop_mean_precision

def get_mean_precision_bups(dictio_results, query_type, string_token, min_sample):
    if query_type == "subject":
        query_type = "subj_label"
    elif query_type == "object":
        query_type = "obj_label"
    dictio_prop_distribution = json.load(open("/data/fichtel/projektarbeit/distribution_queries_{}_{}.json".format(string_token, min_sample), "r"))
    #bups: BERT's uniquness perfomance score
    #mean precision over all props
    mean_prec_bups = 0
    dictio_prop_mean_precision = {}
    for prop in dictio_results:
        #precision of each prop over all queries
        precision = 0
        count_queries = len(dictio_results[prop])
        bups_perfect = 1/count_queries
        #print("bups perf", bups_perfect)
        
        for query in dictio_results[prop]:
            #correct_answers_indices to current query and prop
            indices = dictio_results[prop][query]["correct_answers_indices"]
            if len(indices) > 0 and 0 in indices:
                correct_answer = dictio_results[prop][query]["all_answers"][0][0]
                if dictio_prop_distribution[prop][query_type][correct_answer] > count_queries:
                    exit("dictio_prop_distribution[{}][{}][{}] > count_queries".format(prop, query_type, correct_answer))
                bups = dictio_prop_distribution[prop][query_type][correct_answer]/count_queries
                precision = precision + (1 - (bups - bups_perfect))  
        
        precision = precision/count_queries 
        dictio_prop_mean_precision[prop] = precision
        mean_prec_bups = mean_prec_bups + precision
    count_props = len(dictio_results)
    mean_prec_bups = mean_prec_bups/count_props
    return mean_prec_bups, dictio_prop_mean_precision

def get_interval_index(confidence, intervals):
    for i, interval in enumerate(intervals):
        if confidence in interval:
            return i
    

def get_ece(dictio_results, string_token, min_sample, M=10):
    import intervals as I
    #expected calibration error (ECE) over all props
    intervals = []
    for m in range(1, M+1):
        if m == 1:
            intervals.append(I.closed((m-1)/M, m/M))
        else:
            intervals.append(I.openclosed((m-1)/M, m/M))
    #print(intervals)
    bins = [[] for _ in range(M)]
    for prop in dictio_results:
        for query in dictio_results[prop]:
            #probability = confidence
            best_answer = dictio_results[prop][query]["all_answers"][0]
            probability = best_answer[1]
            i = get_interval_index(probability, intervals)
            bins[i].append(query)
    #print(bins)
    precisions = [None] * len(bins)
    confidences = [None] * len(bins)
    count_all_queries = 0
    for i, queries in enumerate(bins):
        count_all_queries = count_all_queries + len(queries)
        precision_at_1 = 0
        probability_at_1 = 0
        for query in queries:
            triple = query.split("_")
            prop = triple[1]
            #calculate precision@1
            indices = dictio_results[prop][query]["correct_answers_indices"]
            if len(indices) > 0 and 0 in indices:
                precision_at_1 = precision_at_1 + 1
            #calculate confidence = probability@1
            answer_probability = dictio_results[prop][query]["all_answers"][0][1]
            probability_at_1 = probability_at_1 + answer_probability
        
        precisions[i] = precision_at_1/len(queries) if len(queries) != 0 else 0
        confidences[i] = probability_at_1/len(queries) if len(queries) != 0 else 0
    ece = 0
    for i, queries in enumerate(bins):
        ece = ece + (len(queries)/count_all_queries) * abs(precisions[i] - confidences[i])
    #print(ece)
    return ece

def get_count_memorized_correctly(dictio_all_results):
    count_memorized_correctly_normal = 0
    count_memorized_correctly_finetuned = 0
    count_all_queries = 0
    for prop in dictio_all_results:
        for result in dictio_all_results[prop]["bert-base-cased-normal"]:
            count_all_queries = count_all_queries + 1
            indices_normal = result["correct_answers_indices"]
            if len(indices_normal) > 0:
                count_memorized_correctly_normal = count_memorized_correctly_normal + 1
        for result in dictio_all_results[prop]["bert-base-cased-finetuned"]:
            indices_finetuned = result["correct_answers_indices"]
            if len(indices_finetuned) > 0:
                count_memorized_correctly_finetuned = count_memorized_correctly_finetuned + 1
    return count_memorized_correctly_normal, count_memorized_correctly_finetuned, count_all_queries

def get_count_queries_already_seen_in_training(dictio_all_results, training_dataset):
    #all answers of the queries of each prop in the training dataset
    dictio_prop_answers = {}
    for dictio_query_answer in training_dataset:
        prop = dictio_query_answer["prop"]
        answer = dictio_query_answer["answer"]
        if prop not in dictio_prop_answers:
            dictio_prop_answers[prop] = set()
        dictio_prop_answers[prop].add(answer)
    #count for the queries, that are answered by the lm with an answer already seen in the training data
    count_queries = 0
    #count of all queries
    count_all_queries = 0
    #new answers not seen in training
    new_answers = set()
    for prop in dictio_all_results:
        for query in dictio_all_results[prop]:
            count_all_queries = count_all_queries + 1
            #answer of current query of lm with best probability
            indices = dictio_all_results[prop][query]["bert-base-cased-finetuned"]["correct_answers_indices"]
            if 0 not in indices:
                #only look at inccorect first answer
                first_answer_incorrect = dictio_all_results[prop][query]["bert-base-cased-finetuned"]["all_answers"][0][0]
                if prop in dictio_prop_answers and first_answer_incorrect in dictio_prop_answers[prop]:
                    #first_answer_incorrect already seen in training
                    count_queries = count_queries + 1
                else:
                    new_answers.add(first_answer_incorrect)

    return count_queries, count_all_queries, list(new_answers)

def get_count_same_query_correct_answered(epoch, alone, min_sample, sample, string_token, perc_prop):
    count_queries = 0
    count_correct_queries_LAMA = 0
    count_correct_queries_label = 0
    if not alone and perc_prop=="100":
        dictio_all_results_LAMA = json.load(open("results/queries/results_queries_{}_{}_{}_{}{}_{}_{}.json".format(perc_prop, string_token, epoch, "LAMA", alone, min_sample, sample), "r"))
        dictio_all_results_label = json.load(open("results/queries/results_queries_{}_{}_{}_{}{}_{}_{}.json".format(perc_prop, string_token, epoch, "label", alone, min_sample, sample), "r"))
        for prop in dictio_all_results_LAMA:
            for query in dictio_all_results_LAMA[prop]:
                indices_LAMA = dictio_all_results_LAMA[prop][query]["bert-base-cased-finetuned"]["correct_answers_indices"]
                if query in dictio_all_results_label[prop]:
                    indices_label = dictio_all_results_label[prop][query]["bert-base-cased-finetuned"]["correct_answers_indices"]
                else:
                    print("ERROR not the same queries at LAMA and label templates even though alone == false")
                    return None, None, None
                if len(indices_LAMA) > 0 and 0 in indices_LAMA:
                    count_correct_queries_LAMA = count_correct_queries_LAMA + 1
                if len(indices_label) > 0 and 0 in indices_label:
                    count_correct_queries_label = count_correct_queries_label + 1
                #check if lm answered the current query correctly with LAMA and label template
                if len(indices_LAMA) > 0 and 0 in indices_LAMA and len(indices_label) > 0 and 0 in indices_label:
                    count_queries = count_queries + 1
        return count_queries, count_correct_queries_LAMA, count_correct_queries_label
    else:
        return None, None, None



if __name__ == "__main__":
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-queries',action="store_true", default=False, help="set flag if queries should be evaluated")
    parser.add_argument('-train', action="store_true", default=False, help="set flag if training data should be evaluated")
    parser.add_argument('-min_sample', help="set how many triple at least should exist of each property in wikidata_onetoken_missing")
    parser.add_argument('-sample', help="set how many triple should be used of each property (e.g. 10000)")
    parser.add_argument('-epoch', help="set how many epoches should be executed")
    parser.add_argument('-template', help="set which template should be used (LAMA or label or auto)")
    parser.add_argument('-alone',action="store_true", default=False, help="set flag training data should be used which is only for one template")
    parser.add_argument('-pers_pronoun',action="store_true", default=False, help="set flag if personal pronouns should be analyzed")
    parser.add_argument('-string_token', help="set if obj and subj labels should consist of only one word (oneword) and are also in vocab file (onetoken)")
    parser.add_argument('-perc_prop', help="set how many props should be used for training (e.g. 100 for all props or 90-0 for first random_prop selection with 90% of the props)")
    
    args = parser.parse_args()
    print(args)
    epoch = args.epoch
    template = args.template
    alone = args.alone
    if alone:
        alone = "alone"
    else:
        alone = ""
    pers_pronoun = args.pers_pronoun
    if pers_pronoun:
        pers_pronoun = "_p"
    else:
        pers_pronoun = ""
    sample = args.sample
    min_sample = args.min_sample
    string_token = args.string_token
    assert(string_token in ["onetoken", "oneword", ""])
    perc_prop = args.perc_prop
    if perc_prop != "100" and not (int(sample)==500 or int(sample)==100 or template=="LAMA"):
        exit("ERROR training on less than 100% of the props is only possible with sample=500 or sample=100 and template=label")
    
    if args.queries:
        #results of the lms for the queries
        #how many results of the lm should be ovserved (must be > 0)
        count_results = 1
        dictio_all_results = json.load(open("results/queries/results_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, sample, pers_pronoun), "r"))
        if template != "auto":
            training_dataset = read_dataset("/data/fichtel/projektarbeit/training_datasets/training_dataset_{}_{}_{}{}_{}_{}.json".format(perc_prop, string_token, template, alone, min_sample, sample))  
            dictio_eval_results = get_results(dictio_all_results, training_dataset, count_results, string_token, perc_prop, min_sample)
        else:
            dictio_eval_results = get_results_auto_template(dictio_all_results, count_results, string_token, perc_prop)
        #save evalulation results
        if os.path.exists("results/queries/eval_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, sample, pers_pronoun)):
            print("remove results/queries/eval_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, sample, pers_pronoun))
            os.remove("results/queries/eval_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, sample, pers_pronoun))
        eval_file = open("results/queries/eval_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, sample, pers_pronoun), "w")
        json.dump(dictio_eval_results, eval_file)
        
    if args.train:
        if template == "auto":
            exit("BERT has not be trained with auto templates, so the training data memorization cannot be examined")
        
        #print("\nevaluation memorization of training data\n")
        #results of the finetuned lm for training queries
        dictio_all_results = json.load(open("results/train_data/results_train_queries_{}_{}_{}_{}{}_{}_{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, sample), "r"))
        
        count_memorized_correctly_normal, count_memorized_correctly_finetuned, count_all_queries = get_count_memorized_correctly(dictio_all_results)
        #print("#train queries which are memorized correctly (strict: {})".format(bool_strict))
        #print("normal: {}/{} --> {}".format(count_memorized_correctly_normal, len(dictio_all_results),count_memorized_correctly_normal/len(dictio_all_results)))
        #print("finetuned: {}/{} --> {}".format(count_memorized_correctly_finetuned, len(dictio_all_results),count_memorized_correctly_finetuned/len(dictio_all_results)))
        
        dictio_eval_results = {}
        dictio_eval_results["correctly_memorized"] = {}
        dictio_eval_results["correctly_memorized"]["normal"] = {"count_train_queries": count_memorized_correctly_normal, "count_all_train_queries": count_all_queries}
        dictio_eval_results["correctly_memorized"]["finetuned"] = {"count_train_queries": count_memorized_correctly_finetuned, "count_all_train_queries": count_all_queries}

        #save evalulation results
        if os.path.exists("results/train_data/eval_train_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, sample, pers_pronoun)):
            print("remove results/train_data/eval_train_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, sample, pers_pronoun))
            os.remove("results/train_data/eval_train_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, sample, pers_pronoun))
        eval_file = open("results/train_data/eval_train_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, sample, pers_pronoun), "w")
        json.dump(dictio_eval_results, eval_file)

    