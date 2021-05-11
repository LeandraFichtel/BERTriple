import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import pandas as pd
import math

def save_precision_figure(dictio_template_sample_precision, file_name):
    x_axis = list(int(x)*2 for x in dictio_template_sample_precision["LAMA"])
    x_axis_labels = list(int(x)*2 for x in dictio_template_sample_precision["LAMA"])
    for template in dictio_template_sample_precision:
        y_axis_normal = []
        y_axis_finetuned = []
        for sample in dictio_template_sample_precision[template]:
            y_axis_normal.append(round_half_up(dictio_template_sample_precision[template][sample]["normal"]*100, 2))
            if "finetuned" in dictio_template_sample_precision[template][sample]:
                y_axis_finetuned.append(round_half_up(dictio_template_sample_precision[template][sample]["finetuned"]*100, 2))
        if template == "LAMA":
            p1, = plt.plot(x_axis, y_axis_normal, "b-", label="manual templates")
            p2, = plt.plot(x_axis, y_axis_finetuned, "r-", marker='x', label="manual templates")
        elif template == "label":
            p3, = plt.plot(x_axis, y_axis_normal, "b--", label="label templates")
            p4, = plt.plot(x_axis, y_axis_finetuned, "r--", marker='x', label="label templates")
        elif template == "auto":
            p5, = plt.plot(x_axis, y_axis_normal, "b:", marker='x', label="auto templates")
    
    plt.xlabel('sample size')
    plt.ylabel('average precision@1 [%]')
    plt.xscale("log")
    plt.xticks(ticks=x_axis, labels=x_axis_labels, rotation='vertical')
    plt.yticks(np.arange(0, 37, 2))
    l1 = plt.legend(handles=[p1,p3, p5], title='BBC', bbox_to_anchor=(1.01, 0.75), loc='upper left')
    l1._legend_box.align = "left"
    plt.gca().add_artist(l1)
    l2 = plt.legend(handles=[p2,p4], title='BBCF', bbox_to_anchor=(1.01, 1), loc='upper left')
    l2._legend_box.align = "left"
    plt.gca().add_artist(l2)
    plt.savefig("{}.pdf".format(file_name), bbox_inches='tight', format="pdf")
    plt.clf()
    print("saved {}.pdf".format(file_name))

def save_precision_random_props_figure(dictio_sample_random_props_precision, template):
    x_axis = list(int(x) for x in dictio_sample_random_props_precision["500"])
    x_axis_labels = list(str(x) for x in dictio_sample_random_props_precision["500"])
    p1 = None
    p2 = None
    for sample in dictio_sample_random_props_precision:
        y_axis = []
        for percent in dictio_sample_random_props_precision[sample]:
            y_axis.append(round_half_up(dictio_sample_random_props_precision[sample][percent]*100, 2))
        if sample == "100":
            p1, = plt.plot(x_axis, y_axis, "r--", marker='x', label="sample size 200," + "\n" + "{} templates".format(template))
        elif sample == "500":
            p2, = plt.plot(x_axis, y_axis, "r--", marker='x', label="sample size 1000," + "\n" + "{} templates".format(template))
    
    plt.xlabel('amount of properties during training [%]')
    plt.ylabel('average precision@1 [%]')
    plt.xticks(ticks=x_axis, labels=x_axis_labels)
    plt.yticks(np.arange(0, 37, 2))
    if p1 and p2:
        l = plt.legend(handles=[p1,p2], title='BBCF', bbox_to_anchor=(1.01, 1), loc='upper left')
    elif p1:
        l = plt.legend(handles=[p1], title='BBCF', bbox_to_anchor=(1.01, 1), loc='upper left')
    elif p2:
        l = plt.legend(handles=[p2], title='BBCF', bbox_to_anchor=(1.01, 1), loc='upper left')
    l._legend_box.align = "left"
    plt.gca().add_artist(l)
    plt.gca().invert_xaxis()
    plt.savefig("precision_random_props_{}.pdf".format(template), bbox_inches='tight', format="pdf")
    print("saved precision_random_props_{}.pdf".format(template))

def save_probability_figure(dictio_probability, template, min_sample, sample, query_type):
    fig, subplots = plt.subplots(1, 2, figsize=(14, 6)) 
    #get correct probabilities
    normal_correct_queries = dictio_probability["normal"]["correct"]
    normal_probabilities_correct = list(round_half_up(x*100, 2) for x in normal_correct_queries)
    finetuned_correct_queries = dictio_probability["finetuned"]["correct"]
    finetuned_probabilities_correct = list(round_half_up(x*100, 2) for x in finetuned_correct_queries)
    #get incorrect probabilities
    normal_incorrect_queries = dictio_probability["normal"]["incorrect"]
    normal_probabilities_incorrect = list(round_half_up(x*100, 2) for x in normal_incorrect_queries)
    finetuned_incorrect_queries = dictio_probability["finetuned"]["incorrect"]
    finetuned_probabilities_incorrect = list(round_half_up(x*100, 2) for x in finetuned_incorrect_queries)

    subplots[0].boxplot([normal_probabilities_correct, finetuned_probabilities_correct], positions=[0, 1], showmeans=True, showfliers=False)
    subplots[0].set_ylabel('probability [%]', fontsize = 15)
    subplots[0].set_yticks(np.arange(0, 110, 10))
    subplots[0].set_yticklabels(np.arange(0, 110, 10), fontsize = 15)
    x_axis_labels = ["BBC, {} queries".format(len(normal_correct_queries)), "BBCF, {} queries".format(len(finetuned_correct_queries))]
    subplots[0].set_xticklabels(x_axis_labels, fontsize = 15)
    x = np.arange(len(x_axis_labels))
    subplots[0].set_xticks(ticks=x)
    subplots[0].set_title("correct answers", size=15)

    subplots[1].boxplot([normal_probabilities_incorrect, finetuned_probabilities_incorrect], positions=[0, 1], showmeans=True, showfliers=False)
    subplots[1].set_ylabel('probability [%]', fontsize = 15)
    subplots[1].set_yticks(np.arange(0, 110, 10))
    subplots[1].set_yticklabels(np.arange(0, 110, 10), fontsize = 15)
    x_axis_labels = ["BBC, {} queries".format(len(normal_incorrect_queries)), "BBCF, {} queries".format(len(finetuned_incorrect_queries))]
    subplots[1].set_xticklabels(x_axis_labels, fontsize = 15)
    x = np.arange(len(x_axis_labels))
    subplots[1].set_xticks(ticks=x)
    subplots[1].set_title("incorrect answers", size=15)
    
    #handles, labels = subplots[0].get_legend_handles_labels()
    #fig.legend(handles, labels, bbox_to_anchor=(1, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig("probability_{}_{}_{}_{}.pdf".format(template, min_sample*2, sample*2, query_type), bbox_inches='tight', format="pdf")
    print("saved probability_{}_{}_{}_{}.pdf".format(template, min_sample*2, sample*2, query_type))

def save_ece_figure(dictio_sample_ece, template, query_type):
    x_axis = list(str(x) for x in dictio_sample_ece)
    x_axis_labels = list(str(int(x)*2) for x in dictio_sample_ece)
    y_axis_normal = []
    y_axis_finetuned = []
    for sample in dictio_sample_ece:
        y_axis_normal.append(dictio_sample_ece[sample]["normal"])
        y_axis_finetuned.append(dictio_sample_ece[sample]["finetuned"])

    if template == "LAMA":
        p1, = plt.plot(x_axis, y_axis_normal, "b-", label="manual templates")
        p2, = plt.plot(x_axis, y_axis_finetuned, "r-", marker='x', label="manual templates")
    elif template == "label":
        p1, = plt.plot(x_axis, y_axis_normal, "b--", label="label templates")
        p2, = plt.plot(x_axis, y_axis_finetuned, "r--", marker='x', label="label templates")

    print(y_axis_normal, y_axis_finetuned)
    plt.xlabel('sample size')
    plt.ylabel('ECE')
    plt.xticks(ticks=x_axis, labels=x_axis_labels, rotation='vertical')
    plt.yticks(np.arange(0, 1.01, 0.05))
    l1 = plt.legend(handles=[p1], title='BBC', bbox_to_anchor=(1.01, 0.8), loc='upper left')
    l1._legend_box.align = "left"
    plt.gca().add_artist(l1)
    l2 = plt.legend(handles=[p2], title='BBCF', bbox_to_anchor=(1.01, 1), loc='upper left')
    l2._legend_box.align = "left"
    plt.gca().add_artist(l2)            
    plt.savefig("ece_{}_{}.pdf".format(template, query_type), bbox_inches='tight', format="pdf")
    plt.clf()
    print("saved ece_{}_{}.pdf".format(template,query_type))

def round_half_up(n, decimals=0):
    multiplier = 10** decimals
    return math.floor(n*multiplier + 0.5)/multiplier

if __name__ == "__main__":
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-training_data_memorization',action="store_true", default=False, help="set flag to get count how good the finetuned lm memorized the training data")
    parser.add_argument('-template_eval',action="store_true", default=False, help="set flag to count queries which are answered by the finetuned lm correctly with LAMA *and* label template")
    parser.add_argument('-seen_in_training',action="store_true", default=False, help="set flag to count queries whiche are answered with answers which are already seen during training")
    parser.add_argument('-precision',action="store_true", default=False, help="set flag if plot for precision with different amount if training data should be created")
    parser.add_argument('-probability',action="store_true", default=False, help="set flag if plot for probability should be created")
    parser.add_argument('-random_props',action="store_true", default=False, help="set flag if plot for precision witper_propsh random propbs should be created")
    parser.add_argument('-per_props_template',action="store_true", default=False, help="set flag if plot for precision per prop evaluating templates should be created")
    parser.add_argument('-per_props_random_props',action="store_true", default=False, help="set flag if plot for precision per prop evaluating random props should be created")
    parser.add_argument('-ece',action="store_true", default=False, help="set flag if plot for ece for each sample should be created")

    parser.add_argument('-min_sample', help="set how many triple at least should exist of each property in wikidata_onetoken_missing")
    parser.add_argument('-sample', help="set how many triple should be used of each property (e.g. 10000 or all)")
    parser.add_argument('-epoch', help="set how many epoches should be executed")
    parser.add_argument('-template', help="set which template should be used (LAMA or label or both)")
    parser.add_argument('-alone',action="store_true", default=False, help="set flag training data should be used which is only for one template")
    parser.add_argument('-pers_pronoun',action="store_true", default=False, help="set flag if personal pronouns should be analyzed")
    parser.add_argument('-string_token', help="set if obj and subj labels should consist of only one word (oneword) and are also in vocab file (onetoken)")
    parser.add_argument('-perc_prop', help="set how many props should be used for training (e.g. all or 100 for all props or 90-0 for first random_prop selection with 90% of the props)")
    

    args = parser.parse_args()
    print(args)
    epoch = args.epoch
    template = args.template
    if template == "all":
        templates = ["LAMA", "label", "auto"]
    else:
        templates = None
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
    if sample == "all":
        samples = [1, 5, 10, 20, 30, 50, 100, 200, 300, 400, 500]
    elif sample == "random_props":
        samples = [100, 500]
    else:
        samples = None
    min_sample = args.min_sample
    string_token = args.string_token
    assert(string_token in ["onetoken", "oneword", ""])
    perc_prop = args.perc_prop
    if perc_prop != "all" and perc_prop != "100" and (sample != "500" and sample != "100" and template!="label"):
        exit("ERROR training on less than 100% of the props is only possible with sample=500 or sample=100 and template=label")
    
    if args.seen_in_training:
        if alone or samples != None or templates != None or template == "auto":
            exit("WARNING No plot for more than one sample or more than one template")
        else:
            dictio_eval_results = json.load(open("results/queries/eval_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, str(sample), pers_pronoun), "r"))
            
            #get count which queries are answered with answers which are already seen during training TODO make table
            count_queries = dictio_eval_results["seen_in_training"]["count_queries"]
            count_all_queries = dictio_eval_results["seen_in_training"]["count_all_queries"]
            new_answers = dictio_eval_results["seen_in_training"]["new_answers"]
            print("\n{}/{} queries are answered by the lm with an answer already seen during training --> {}".format(count_queries, count_all_queries, count_queries/count_all_queries))
            print(new_answers)

    if args.training_data_memorization:
        if alone or samples != None or templates != None or template == "auto":
            exit("WARNING No plot for more than one sample or more than one template")
        else:
            dictio_eval_results = json.load(open("results/train_data/eval_train_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, str(sample), pers_pronoun), "r"))
            
            #get count which queries are memorized correctly
            count_all_queries = dictio_eval_results["correctly_memorized"]["normal"]["count_all_train_queries"]
            count_queries_normal = dictio_eval_results["correctly_memorized"]["normal"]["count_train_queries"]
            count_queries_finetuned = dictio_eval_results["correctly_memorized"]["finetuned"]["count_train_queries"]
            
            print("\n{}/{} queries are are memorized correctly by normal lm --> {}".format(count_queries_normal, count_all_queries, count_queries_normal/count_all_queries))
            print("\n{}/{} queries are are memorized correctly by finetuned lm --> {}".format(count_queries_finetuned, count_all_queries, count_queries_finetuned/count_all_queries))

    if args.template_eval:
        if alone or perc_prop != "100" or samples != None or templates != None or template == "auto":
            exit("WARNING No plot for more than one sample or more than one template or less than 100% of props during training")
        else:
            dictio_eval_results = json.load(open("results/queries/eval_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, str(sample), pers_pronoun), "r"))
                        
            #get count which queries are answered by the finetuned lm correctly with LAMA *and* label template
            count_queries = dictio_eval_results["correct_both_templates"]["count_queries"]
            count_correct_queries_LAMA = dictio_eval_results["correct_both_templates"]["count_correct_queries_LAMA"]
            count_correct_queries_label = dictio_eval_results["correct_both_templates"]["count_correct_queries_label"]
            if count_queries and count_correct_queries_LAMA and count_correct_queries_label:
                print("\n{} queries are answered by the finetuned lm correctly with LAMA *and* label template".format(count_queries))
                print("{} queries are answered by the finetuned lm correctly with LAMA template".format(count_correct_queries_LAMA))
                print("{} queries are answered by the finetuned lm correctly with label template".format(count_correct_queries_label))
    if args.precision:
        if samples == None or templates == None or perc_prop != "100":
            exit("WARNING No plot for only one sample value or only one template or less than 100% of props during training")
        else:
            for file_name in ["precision@1", "precision_bups"]:
                dictio_template_sample_precision = {}
                for template in templates:
                    dictio_template_sample_precision[template] = {}
                    #get results for LAMA, label and auto template
                    for sample in samples:
                        dictio_eval_results = json.load(open("results/queries/eval_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, str(sample), pers_pronoun), "r"))
                        dictio_template_sample_precision[template][str(sample)] = dictio_eval_results["subjobj"][file_name]["overall"]                
                save_precision_figure(dictio_template_sample_precision, file_name)

    if args.probability:
        if samples != None or templates != None or template == "auto":
            exit("WARNING No plot for more than one sample or more than one template")
        else:
            for query_type in ["subjobj", "subject", "object"]:
                dictio_probability = {"normal": {}, "finetuned": {}}
                dictio_eval_results = json.load(open("results/queries/eval_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, str(sample), pers_pronoun), "r"))
                dictio_probability["normal"]["correct"] = dictio_eval_results["correct"][query_type]["probability@1"]["normal"]
                dictio_probability["normal"]["incorrect"] = dictio_eval_results["incorrect"][query_type]["probability@1"]["normal"]
                dictio_probability["finetuned"]["correct"] = dictio_eval_results["correct"][query_type]["probability@1"]["finetuned"]
                dictio_probability["finetuned"]["incorrect"] = dictio_eval_results["incorrect"][query_type]["probability@1"]["finetuned"]
            
                save_probability_figure(dictio_probability, template, int(min_sample), int(sample), query_type)
    
    if args.random_props:
        if template != "label" or (sample != "500" and sample != "100" and sample != "random_props"):
            exit("WARNING Only plot for label templates and sample=100 and sample=500 (--> sample=random_props")
        if perc_prop != "all":
            print("WARNING value for perc_prop will be ignored because all perc_props are evaluated")
        dictio_sample_random_props_precision = {}
        if sample != "random_props":
            samples = [sample]
        for sample in samples:
            dictio_sample_random_props_precision[str(sample)] = {}
            #eval results for 100% of the props
            dictio_eval_results = json.load(open("results/queries/eval_queries_100_{}_{}_{}{}_{}_{}{}.json".format(string_token, epoch, template, alone, min_sample, sample, pers_pronoun), "r"))
            dictio_sample_random_props_precision[str(sample)]["100"] = dictio_eval_results["subjobj"]["precision@1"]["overall"]["finetuned"]
            #eval results for less then 100% of the props
            for percent in [90, 80, 70, 60, 50, 40, 30, 20, 10]:
                print(percent)
                average_precision = 0
                for i in range(5):
                    perc_prop = "{}-{}".format(percent, i)
                    dictio_eval_results = json.load(open("results/queries/eval_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, sample, pers_pronoun), "r"))
                    print(dictio_eval_results["subjobj"]["precision@1"]["overall"]["finetuned"])
                    average_precision = average_precision + dictio_eval_results["subjobj"]["precision@1"]["overall"]["finetuned"]
                average_precision = average_precision / 5
                dictio_sample_random_props_precision[str(sample)][str(percent)] = average_precision

        save_precision_random_props_figure(dictio_sample_random_props_precision, template)

    if args.per_props_template:
        if samples != None or perc_prop != "100" or template != "all":
            exit("WARNING No plot for more than one sample or for less than 100% of props")
        else:
            # Defining custom function which returns 
            # the list for df.style.apply() method
            #def highlight_max(s):
            #    is_max = s == s.max()
            #    return ['font-weight: bold' if cell else '' for cell in is_max]

            dictio_eval_results = json.load(open("results/queries/eval_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, templates[0], alone, min_sample, sample, pers_pronoun), "r"))
            for query_type in ["subjobj", "subject", "object"]:
                props = dictio_eval_results["props"]
                columns = []
                precision_matrix = np.zeros(shape=(len(props),len(templates)*2-1), dtype='object')
                for j, template in enumerate(templates):
                    if template == "LAMA":
                        columns.append("BBC_{}".format("manual"))
                        columns.append("BBCF_{}".format("manual"))
                    elif template == "label":
                        columns.append("BBC_{}".format("label"))
                        columns.append("BBCF_{}".format("label"))
                    elif template == "auto":
                        columns.append("BBC_{}".format("auto"))
                    dictio_eval_results = json.load(open("results/queries/eval_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, sample, pers_pronoun), "r"))
                    precision_per_prop_normal = dictio_eval_results[query_type]["precision@1"]["per_prop"]["normal"]
                    if template != "auto": precision_per_prop_finetuned = dictio_eval_results[query_type]["precision@1"]["per_prop"]["finetuned"]
                    precision_avg_normal = dictio_eval_results[query_type]["precision@1"]["overall"]["normal"]
                    if template != "auto": precision_avg_finetuned = dictio_eval_results[query_type]["precision@1"]["overall"]["finetuned"]
                    for i, prop in enumerate(props):
                        #precison  of bert-base-cased normal
                        if precision_per_prop_normal[prop]*100 < 0.01 and not precision_per_prop_normal[prop]*100 == 0:
                            precision_matrix[i][j*2] = "<0.01"
                        else:
                            precision_matrix[i][j*2] =  round_half_up(precision_per_prop_normal[prop]*100, 2)
                        #precison per prop of bert-base-cased finetuned
                        if template != "auto": precision_matrix[i][j*2+1] = round_half_up(precision_per_prop_finetuned[prop]*100, 2)

                pd.options.display.float_format = '{:,.2f}'.format   
                df = pd.DataFrame(precision_matrix, index=props, columns=columns)
                df['indexNumber'] = df.index.str.replace("P", "").astype(int)
                df = df.sort_values(['indexNumber']).drop('indexNumber', axis=1)                
                #df.style.highlight_max(color = 'lightgreen', axis = 1)
                df.loc['avg'] = df.mean()
                
                print(df) 
                df.to_latex("per_props_template_{}_{}.tex".format(sample, query_type))

    if args.per_props_random_props:
        if template != "label" or (sample != "500" and sample != "100"):
            exit("WARNING Only plot for label templates and sample=100 or sample=500")
        if perc_prop != "all":
            print("WARNING value for perc_prop will be ignored because all perc_props are evaluated")
        
        columns = [100, 90, 80, 70, 60, 0]
        dictio_eval_results = json.load(open("results/queries/eval_queries_{}_{}_{}_{}{}_{}_{}{}.json".format("100", string_token, epoch, template, alone, min_sample, sample, pers_pronoun), "r"))
        props = set(dictio_eval_results["props"])

        precision_matrix = np.zeros(shape=(len(props), len(columns)), dtype='object')

        dictio_eval_results = json.load(open("results/queries/eval_queries_100_{}_{}_{}{}_{}_{}{}.json".format(string_token, epoch, template, alone, min_sample, sample, pers_pronoun), "r"))
        precision_per_prop_finetuned = dictio_eval_results["subjobj"]["precision@1"]["per_prop"]["finetuned"]
        for i, prop in enumerate(props):
            precision_matrix[i][0] = format(round_half_up(precision_per_prop_finetuned[prop]*100, 2), '.2f')
        
        index_random_props = 0

        dataset_file = json.load(open("/data/fichtel/projektarbeit/dataset_{}_{}.json".format(string_token, min_sample), "r"))
        missing_props = {}
        for percent in dataset_file["random_props"]:
            used_props = set(dataset_file["random_props"][percent][index_random_props])
            missing_props[percent] = props.difference(used_props)
        print(missing_props)
        
        for j, percent in enumerate(columns[1:]):
            if percent == 0:
                dictio_eval_results = json.load(open("results/queries/eval_queries_100_{}_{}_{}{}_{}_{}{}.json".format(string_token, epoch, template, alone, min_sample, sample, pers_pronoun), "r"))
                precision_per_prop_normal = dictio_eval_results["subjobj"]["precision@1"]["per_prop"]["normal"]
                for i, prop in enumerate(props):
                    #precison per prop of bert-base-cased 
                    precision_matrix[i][j+1] = format(round_half_up(precision_per_prop_normal[prop]*100, 2), '.2f')
            else:
                dictio_eval_results = json.load(open("results/queries/eval_queries_{}-{}_{}_{}_{}{}_{}_{}{}.json".format(percent, index_random_props, string_token, epoch, template, alone, min_sample, sample, pers_pronoun), "r"))
                precision_per_prop_finetuned = dictio_eval_results["subjobj"]["precision@1"]["per_prop"]["finetuned"]
                for i, prop in enumerate(props):
                    #precison per prop of bert-base-cased-finetuned
                    if prop in missing_props[str(percent)]:
                        precision_matrix[i][j+1] = str(format(round_half_up(precision_per_prop_finetuned[prop]*100, 2), '.2f'))+"x"
                    else:
                        precision_matrix[i][j+1] = format(round_half_up(precision_per_prop_finetuned[prop]*100, 2), '.2f')
        
        df = pd.DataFrame(precision_matrix, index=props, columns=columns)
        df['indexNumber'] = df.index.str.replace("P", "").astype(int)
        df = df.sort_values(['indexNumber']).drop('indexNumber', axis=1)
        df.to_latex("per_props_random_props_{}.tex".format(sample))

    if args.ece:
        if sample != "all" or templates != None or template == "auto":
            exit("WARNING No plot for only one sample or more than one template or template == auto")

        for query_type in ["subjobj", "subject", "object"]:
            dictio_sample_ece = {}
            for sample in samples:
                dictio_eval_results = json.load(open("results/queries/eval_queries_{}_{}_{}_{}{}_{}_{}{}.json".format(perc_prop, string_token, epoch, template, alone, min_sample, str(sample), pers_pronoun), "r"))
                dictio_sample_ece[str(sample)] = dictio_eval_results[query_type]["ece"]                
            save_ece_figure(dictio_sample_ece, template, query_type)