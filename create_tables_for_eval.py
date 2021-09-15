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
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-precision',action="store_true", default=False, help="set flag if plot for precision with different amount if training data should be created")
    parser.add_argument('-transfer_learning',action="store_true", default=False, help="set flag if plot for precision with different amount if training data should be created")
    parser.add_argument('-distribution',action="store_true", default=False, help="set flag if the distribution should be evaluated")
    parser.add_argument('-precision_3_runs',action="store_true", default=False, help="set flag if plot for precision with different amount if training data should be created (for each sample size 3 runs)")

    parser.add_argument('-train_file', help="training dataset name (AUTOPROMPT41)")
    parser.add_argument('-sample', help="set how many triple should be used of each property at maximum (e.g. 500 (=500 triples per prop for each query type) or all (= all given triples per prop for each query type))")
    parser.add_argument('-epoch', help="set how many epoches should be executed")
    parser.add_argument('-template', help="set which template should be used (LAMA or label or ID)")
    parser.add_argument('-query_type', help="set which queries should be used during training (subjobj= subject and object queries, subj= only subject queries, obj= only object queries)")
    parser.add_argument('-lama_uhn', action="store_true", default=False, help="set this flag to evaluate only on the filtered LAMA UHN dataset")

    args = parser.parse_args()
    print(args)
    train_file = args.train_file
    epoch = int(args.epoch)
    template = args.template
    sample = args.sample
    query_type = args.query_type
    assert(query_type in ["subjobj", "subj", "obj"])
    lama_uhn = args.lama_uhn
    
    if args.precision_3_runs:
        lm_name = 'bert-base-cased'
        vocab_type = "common"
        lm_name_initials = get_initials(lm_name)
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
                #get avg prec@1 of finetuned model
                if sample == "all":
                    model_dir = "{}F_{}_{}_{}_{}_{}_{}{}".format(lm_name_initials, train_file, vocab_type, sample, query_type, epoch, template, props_string)
                    result_finetuned = dict((pd.read_csv("results/{}.csv".format(model_dir), sep = ',', header = None)).values)
                    y_axis_finetuned.append(round_half_up(result_finetuned["avg"]*100, 2))
                else:
                    results = []
                    for index in ["", "2_", "3_"]:
                        model_dir = "{}{}F_{}_{}_{}_{}_{}_{}{}".format(index, lm_name_initials, train_file, vocab_type, sample, query_type, epoch, template, props_string)
                        result_finetuned = dict((pd.read_csv("results/{}.csv".format(model_dir), sep = ',', header = None)).values)
                        results.append(round_half_up(result_finetuned["avg"]*100, 2))
                    y_axis_finetuned.append(np.mean(results))
            if template == "LAMA":
                p1, = plt.plot(x_axis, y_axis_finetuned, "b-", marker='x', label="manual prompts")
            elif template == "label":
                p2, = plt.plot(x_axis, y_axis_finetuned, "r--",  marker='x', label="triple prompts")
        
        plt.xlabel('sample size', fontsize = 12)
        plt.ylabel('P@1 [%]', fontsize = 12)
        #plt.xscale("log")
        plt.xticks(ticks=x_axis, labels=x_axis_labels, fontsize = 11)
        plt.yticks(np.arange(0, 55, 5), fontsize = 11)
        l1 = plt.legend(handles=[p1,p2], title='BERTriple', frameon=False)
        l1._legend_box.align = "left"
        plt.gca().add_artist(l1)

        #save plot
        file_name = "{}F_{}_prec@1_3_runs".format(lm_name_initials, train_file)
        plt.savefig("{}.pdf".format(file_name), bbox_inches='tight', format="pdf")
        plt.clf()
        print("saved {}.pdf".format(file_name))

    if args.transfer_learning:
        results = {}
        lm_name = 'bert-base-cased'
        vocab_type = "common"
        lm_name_initials = get_initials(lm_name)
        props_string = ""
        lama_uhn = ""
        protocol = json.load(open("results/transfer_learning_protocols/{}F_{}_{}_{}_{}_{}_{}{}.json".format(lm_name_initials, train_file, vocab_type, sample, query_type, epoch, template, props_string), "r"))
        for experiment in protocol["round1"]:
            for prop in experiment["tested_prop"]:
                results[prop] = {}
                results[prop]["BERT"] = protocol["round0"]["tested_prop"][prop]["baseline_prec@1"]
                results[prop]["BERTriple"] = protocol["round0"]["tested_prop"][prop]["trained_prec@1"]
                results[prop]["omitted"] = experiment["tested_prop"][prop]["omitted_prec@1"]
        dictio = {}
        file_relations = open("data/relations.jsonl")
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
        
        
        df = pd.DataFrame.from_dict(results, orient='index')
        df['indexNumber'] = df.index.str.replace("P", "").astype(int)
        df = df.sort_values(['indexNumber']).drop('indexNumber', axis=1) 
        df1 = df.iloc[:21, :]
        df2 = df.iloc[21:, :]
        print(df1)
        print(df2)
        df1.to_latex("prec_per_props_{}F_{}_{}_{}_{}_{}_{}{}{}_1.tex".format(lm_name_initials, train_file, vocab_type, sample, query_type, epoch, template, props_string, lama_uhn))
        df2.to_latex("prec_per_props_{}F_{}_{}_{}_{}_{}_{}{}{}_2.tex".format(lm_name_initials, train_file, vocab_type, sample, query_type, epoch, template, props_string, lama_uhn))

                    
                    
                