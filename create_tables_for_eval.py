import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import pandas as pd
import math
import os
import matplotlib.colors as mcolors

def round_half_up(n, decimals=0):
    multiplier = 10** decimals
    return math.floor(n*multiplier + 0.5)/multiplier

if __name__ == "__main__":
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-precision',action="store_true", default=False, help="set flag if plot for precision with different amount if training data should be created")
    parser.add_argument('-transfer_learning',action="store_true", default=False, help="set flag if plot for precision with different amount if training data should be created")
    parser.add_argument('-distribution',action="store_true", default=False, help="set flag if the distribution should be evaluated")

    parser.add_argument('-train_file', help="training dataset name (LPAQAfiltered41/LPAQAfiltered25 or wikidata41/wikidata25)")
    parser.add_argument('-sample', help="set how many triple should be used of each property at maximum (e.g. 500 (=500 triples per prop for each query type) or all (= all given triples per prop for each query type))")
    parser.add_argument('-epoch', help="set how many epoches should be executed")
    parser.add_argument('-template', help="set which template should be used (LAMA or label)")
    parser.add_argument('-query_type', help="set which queries should be used during training (subjobj= subject and object queries, subj= only subject queries, obj= only object queries)")
    parser.add_argument('-LAMA_UHN', action="store_true", default=False, help="set this flag to evaluate only on the filtered LAMA UHN dataset")

    args = parser.parse_args()
    print(args)
    train_file = args.train_file
    epoch = int(args.epoch)
    template = args.template
    sample = args.sample
    query_type = args.query_type
    assert(query_type in ["subjobj", "subj", "obj"])
    lama_uhn = args.LAMA_UHN
    
    if args.precision:
        lm_name = 'bert-base-cased' 
        lm_name_short = lm_name.split("-")
        lm_name_capitals = lm_name_short[0].upper()[0] + lm_name_short[1].upper()[0] + lm_name_short[2].upper()[0]
        props_string = ""

        templates = ["LAMA", "label"]
        samples = ["1", "10", "30", "50", "100", "200", "300", "400", "500", "600", "700", "800", "900", "all"]
        x_axis = []
        for sample in samples:
            x_axis.append(str(sample))
        x_axis_labels = x_axis

        for template in templates:
            y_axis_baseline = []
            y_axis_finetuned = []
            #get results for LAMA and label templates
            for sample in samples:
                #get avg prec@1 of baseline model
                #result_baseline = dict((pd.read_csv("/home/fichtel/BERTriple/results/bert_base_{}.csv".format(template), sep = ',', header = None)).values)
                #y_axis_baseline.append(round_half_up(result_baseline["avg"]*100, 2))
                
                #get avg prec@1 of finetuned model
                model_dir = "{}F_{}_{}_{}_{}_{}{}".format(lm_name_capitals, train_file, sample, query_type, epoch, template, props_string)
                result_finetuned = dict((pd.read_csv("/home/fichtel/BERTriple/results/{}.csv".format(model_dir), sep = ',', header = None)).values)
                y_axis_finetuned.append(round_half_up(result_finetuned["avg"]*100, 2))
            if template == "LAMA":
                #p1, = plt.plot(x_axis, y_axis_baseline, "b-", label="manual templates")
                p2, = plt.plot(x_axis, y_axis_finetuned, "b-", marker='x', label="manual prompts")
            elif template == "label":
                #p3, = plt.plot(x_axis, y_axis_baseline, "b--", label="triple templates")
                p4, = plt.plot(x_axis, y_axis_finetuned, "r--",  marker='x', label="triple prompts")
        
        plt.xlabel('sample size', fontsize = 12)
        plt.ylabel('P@1 [%]', fontsize = 12)
        #plt.xscale("log")
        plt.xticks(ticks=x_axis, labels=x_axis_labels, fontsize = 11)
        plt.yticks(np.arange(0, 55, 5), fontsize = 11)
        #l1 = plt.legend(handles=[p1,p3], title='bert-base-cased', bbox_to_anchor=(1.01, 0.75), loc='upper left')
        #l1._legend_box.align = "left"
        #plt.gca().add_artist(l1)
        l2 = plt.legend(handles=[p2,p4], title='BERTriple', frameon=False)
        l2._legend_box.align = "left"
        plt.gca().add_artist(l2)

        #save plot
        file_name = "{}_prec@1".format(train_file)
        plt.savefig("{}.pdf".format(file_name), bbox_inches='tight', format="pdf")
        plt.clf()
        print("saved {}.pdf".format(file_name))

    if args.transfer_learning:
        results = {}
        lm_name = 'bert-base-cased' 
        lm_name_short = lm_name.split("-")
        lm_name_capitals = lm_name_short[0].upper()[0] + lm_name_short[1].upper()[0] + lm_name_short[2].upper()[0]
        props_string = ""
        if lama_uhn:
            lama_uhn = "_uhn"
        else:
            lama_uhn = ""
        protocol = json.load(open("/home/fichtel/BERTriple/results/transfer_learning_protocols/{}F_{}_{}_{}_{}_{}{}{}.json".format(lm_name_capitals, train_file, sample, query_type, epoch, template, props_string, lama_uhn), "r"))
        for experiment in protocol["round1"]:
            for prop in experiment["tested_prop"]:
                results[prop] = {}
                results[prop]["BERT"] = protocol["round0"]["tested_prop"][prop]["baseline_prec@1"]
                results[prop]["BERTriple"] = protocol["round0"]["tested_prop"][prop]["trained_prec@1"]
                results[prop]["omitted"] = experiment["tested_prop"][prop]["omitted_prec@1"]
        dictio = {}
        file_relations = open("/data/fichtel/BERTriple/relations.jsonl")
        for line in file_relations:
            data = json.loads(line)
            prop = data["relation"]
            dictio[prop] = data["label"]
        print("hier", dictio["P178"])
        properties_a = {}
        for prop in results:
            if results[prop]["BERT"] > 0:
                if 0.85 < results[prop]["omitted"]/results[prop]["BERT"] < 1.12:
                    properties_a[prop] = dictio[prop]
        print(properties_a)
        for prop in results:
            if results[prop]["BERT"] > 0:
                if results[prop]["omitted"] < 0.5 * results[prop]["BERT"]:
                    print(prop, dictio[prop])
        
        exit()
        df = pd.DataFrame.from_dict(results, orient='index')
        df['indexNumber'] = df.index.str.replace("P", "").astype(int)
        df = df.sort_values(['indexNumber']).drop('indexNumber', axis=1) 
        df1 = df.iloc[:21, :]
        df2 = df.iloc[21:, :]
        print(df1)
        print(df2)
        df1.to_latex("precision_per_props_{}F_{}_{}_{}_{}_{}{}{}_1.tex".format(lm_name_capitals, train_file, sample, query_type, epoch, template, props_string, lama_uhn))
        df2.to_latex("precision_per_props_{}F_{}_{}_{}_{}_{}{}{}_2.tex".format(lm_name_capitals, train_file, sample, query_type, epoch, template, props_string, lama_uhn))

    if args.distribution:
        lm_name = 'bert-base-cased' 
        lm_name_short = lm_name.split("-")
        lm_name_capitals = lm_name_short[0].upper()[0] + lm_name_short[1].upper()[0] + lm_name_short[2].upper()[0]
        props_string = ""
        if lama_uhn:
            logging_dir = "/home/fichtel/BERTriple/models/{}F_{}_{}_{}_{}_{}{}/logging_lama_uhn/".format(lm_name_capitals, train_file, sample, query_type, epoch, template, props_string)
        else:
            logging_dir = "/home/fichtel/BERTriple/models/{}F_{}_{}_{}_{}_{}{}/logging_lama/".format(lm_name_capitals, train_file, sample, query_type, epoch, template, props_string)
        distribution_queries = {}
        distribution_results = {}
        print(logging_dir)
        for prop_file in os.listdir(logging_dir):
            prop = prop_file.split(".")[0]
            distribution_queries[prop] = {}
            distribution_results[prop] = {}
            with open(logging_dir+"/"+prop_file, "r") as f:
                for line in f.readlines():
                    dictio = json.loads(line)
                    #get the obj label distribution of the test queries
                    if dictio["query"]["obj_label"] not in distribution_queries[prop]:
                        distribution_queries[prop][dictio["query"]["obj_label"]] = 1
                    else:
                        distribution_queries[prop][dictio["query"]["obj_label"]] += 1
                    #get the distribution of the results for the [MASK]-token
                    if dictio["result"]["token_word_form"] not in distribution_results[prop]:
                        distribution_results[prop][dictio["result"]["token_word_form"]] = 1
                    else:
                        distribution_results[prop][dictio["result"]["token_word_form"]] += 1
        
        if lama_uhn:
            model_dir = "{}F_{}_{}_{}_{}_{}{}_uhn".format(lm_name_capitals, train_file, sample, query_type, epoch, template, props_string)
        else:
            model_dir = "{}F_{}_{}_{}_{}_{}{}".format(lm_name_capitals, train_file, sample, query_type, epoch, template, props_string)
        result_finetuned = dict((pd.read_csv("/home/fichtel/BERTriple/results/{}.csv".format(model_dir), sep = ',', header = None)).values)
                
        dataset_path = "/data/kalo/akbc2021/training_datasets/{}_{}.json".format(train_file, sample)
        with open(dataset_path, "r") as dataset_file:
            dictio_prop_triple = json.load(dataset_file)
        distribution_train_queries = {}
        for prop in dictio_prop_triple["obj_queries"]:
            distribution_train_queries[prop] = {}
            for dictio in dictio_prop_triple["obj_queries"][prop]:
                if dictio["obj"] not in distribution_train_queries[prop]:
                    distribution_train_queries[prop][dictio["obj"]] = 1
                else:
                    distribution_train_queries[prop][dictio["obj"]] += 1
        avg = 0
        for prop in distribution_train_queries:
            label = max(distribution_train_queries[prop], key=distribution_train_queries[prop].get)
            count = 0
            for dings in distribution_queries[prop]:
                count = count + distribution_queries[prop][dings]
            if label in distribution_queries[prop]:
                print(prop, distribution_queries[prop][label]/count*100, result_finetuned[prop])
                avg = avg + distribution_queries[prop][label]/count*100
            else:
                print(prop, 0, result_finetuned[prop])
                avg = avg + 0
        avg = avg/41
        print(avg, result_finetuned["avg"])

                    
                    
                