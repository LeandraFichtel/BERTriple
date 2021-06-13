import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import pandas as pd
import math

def round_half_up(n, decimals=0):
    multiplier = 10** decimals
    return math.floor(n*multiplier + 0.5)/multiplier

if __name__ == "__main__":
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-precision',action="store_true", default=False, help="set flag if plot for precision with different amount if training data should be created")
    parser.add_argument('-transfer_learning',action="store_true", default=False, help="set flag if plot for precision with different amount if training data should be created")

    parser.add_argument('-train_file', help="training dataset name (LPAQAfiltered41/LPAQAfiltered25 or wikidata41/wikidata25)")
    parser.add_argument('-sample', help="set how many triple should be used of each property at maximum (e.g. 500 (=500 triples per prop for each query type) or all (= all given triples per prop for each query type))")
    parser.add_argument('-epoch', help="set how many epoches should be executed")
    parser.add_argument('-template', help="set which template should be used (LAMA or label)")
    parser.add_argument('-query_type', help="set which queries should be used during training (subjobj= subject and object queries, subj= only subject queries, obj= only object queries)")

    args = parser.parse_args()
    print(args)
    train_file = args.train_file
    epoch = int(args.epoch)
    template = args.template
    sample = args.sample
    query_type = args.query_type
    assert(query_type in ["subjobj", "subj", "obj"])
    
    if args.precision:
        lm_name = 'bert-base-cased' 
        lm_name_short = lm_name.split("-")
        lm_name_capitals = lm_name_short[0].upper()[0] + lm_name_short[1].upper()[0] + lm_name_short[2].upper()[0]
        props_string = ""

        templates = ["LAMA", "label"]
        #samples = ["all", "800", "600", "500", "400", "300", "200", "100", "50"]
        samples = ["1", "10", "30", "50", "100", "200", "300", "400", "500", "600", "800"]
        x_axis = []
        for sample in samples:
            if sample == "all":
                sample = 1000
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
                p2, = plt.plot(x_axis, y_axis_finetuned, "r-", marker='x', label="manual templates")
            elif template == "label":
                #p3, = plt.plot(x_axis, y_axis_baseline, "b--", label="label templates")
                p4, = plt.plot(x_axis, y_axis_finetuned, "b--", marker='x', label="label templates")
        
        plt.xlabel('sample size')
        plt.ylabel('P@1 [%]')
        #plt.xscale("log")
        plt.xticks(ticks=x_axis, labels=x_axis_labels, rotation='vertical')
        plt.yticks(np.arange(0, 52, 2))
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
        print("WARNING nothing implemented yet to create transfer learning heatmap")