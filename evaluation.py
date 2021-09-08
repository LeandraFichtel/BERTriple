import argparse
import os
import json
import shutil
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
from tqdm import tqdm       

def start_evaluation(template, model_path, results_file_name, omitted_props=None, lama_uhn=False):
    if omitted_props:
        props = omitted_props
    else:
        props = ["P1001", "P106", "P1303", "P1376", "P1412", "P178", "P19", "P276", "P30", "P364", "P39", "P449", "P495", "P740", "P101", "P108", "P131", "P138", "P159", "P17", "P20", "P279", "P31", "P36", "P407", "P463", "P527", "P937", "P103", "P127", "P136", "P140", "P176", "P190", "P264", "P27", "P361", "P37", "P413", "P47", "P530"]
    assert len(props) > 0
    print("considering {} properties".format(len(props)))
    #use the huggingface pipeline to predict tokens for the mask-token
    fill_mask = pipeline("fill-mask", model=model_path, device=0)
    #choose the dataset und logging path
    if lama_uhn:
        test_data_path = "data/test_datasets/TREx_UHN/"
        results_file_name = results_file_name+"_uhn"
        logging_path = model_path+"/logging_lama_uhn"
        print("\nT-REx UHN evaluation: {}".format(model_path))
    else:
        test_data_path = "data/test_datasets/TREx/"
        logging_path = model_path+"/logging_lama"
        print("\nT-REx evaluation: {}".format(model_path))
    
    common_vocab = set(open("data/common_vocab_cased.txt", "r").read().splitlines())
    if os.path.exists(logging_path):
        print("remove logging dir of model")
        shutil.rmtree(logging_path)
    os.mkdir(logging_path)
    #iterate through all test_data for all properties
    avg_prec_at_1 = 0
    results_file = open("results/{}.csv".format(results_file_name), "w+")
    for file_name in os.listdir(test_data_path):
        prop = file_name.split(".")[0]
        if prop in props: 
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
            assert len(valid_triples) > 0
            #get the right template for the prop
            dictio_prop_template = json.load(open("data/templates.json", "r"))
            query_template = dictio_prop_template[prop][template]
            print("using {} template: {}".format(template, query_template))
            #evaluation (precision@1 per prop and overall avg precison@1)
            for dictio_triple in tqdm(valid_triples):
                mask_query = query_template.replace("[X]", dictio_triple["subj_label"]).replace("[Y]", fill_mask.tokenizer.mask_token)
                
                found_valid_token = False
                predict_try_no = 1
                while not found_valid_token:
                    dictio_result = fill_mask(mask_query, top_k=predict_try_no)
                    predict_try_no += 2
                    for result in dictio_result:
                        if result["token_str"] in common_vocab:
                            found_valid_token = True
                            #create dictio for logging
                            dictio_logging = {}
                            dictio_triple["masked_sentences"] = [mask_query]
                            dictio_logging["query"] = dictio_triple
                            dictio_logging["result"] = result 
                            #check whether the predicted token is correct
                            if result["token_str"] == dictio_triple["obj_label"]:
                                prec_at_1 = prec_at_1 + 1
                                dictio_logging["prec@1"] = 1
                            else:
                                dictio_logging["prec@1"] = 0
                            break
                assert found_valid_token
                json.dump(dictio_logging, logging_file)
                logging_file.write("\n")
            #calculate precision@1 of each prop averaged over all test queries
            prec_at_1 = prec_at_1/len(valid_triples)
            print("prec@1: {}%\n".format(round(prec_at_1 * 100, 2)))
            results_file.write("{},{}\n".format(prop, round(prec_at_1 * 100, 2)))
            avg_prec_at_1 = avg_prec_at_1 + prec_at_1
    #calculate overall precision@1 averaged over all props
    avg_prec_at_1 = avg_prec_at_1/len(props)
    print("avg prec@1 over all props: {}\n\n".format(avg_prec_at_1))
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
    baseline_lms = ["bert-base-cased"]#, "distilbert-base-cased"]
    #check that all baseline language models are cased because the templates for the queries are cased
    for lm_name in baseline_lms:
        assert "uncased" not in lm_name
    
    if not os.path.exists("data/"):
        exit("Please download the data dir from this url: TODO")

    user_input = input("All baseline models and results will be deleted. Is this okay? :) [N/y]")
    if user_input != "y":
        exit("Okay, it is not okay. I stopped the script for you. :)")
    #remove and create model dir and result dir for baseline models
    if os.path.exists("models/baselines/"):
        print("remove dir of baseline models")
        shutil.rmtree("models/baselines/")
    os.mkdir("models/baselines/")
    if os.path.exists("results/baselines/"):
        print("remove results dir of baseline models")
        shutil.rmtree("results/baselines/")
    os.mkdir("results/baselines/")
    
    #use the common vocab of LAMA as basis and find intersection of all vocabs of baseline language models
    common_vocab = set(open("data/LAMA_common_vocab_cased.txt", "r").read().splitlines())
    for lm_name in baseline_lms:
        tokenizer = AutoTokenizer.from_pretrained(lm_name)
        vocab = tokenizer.get_vocab().keys()
        common_vocab = common_vocab.intersection(vocab)
    
    assert len(common_vocab) > 0
    print("save common_vocab_cased.txt")
    with open("data/common_vocab_cased.txt", "w+") as common_vocab_file:
        for token in common_vocab:
            common_vocab_file.write(token+"\n") 

    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-template', help="set which template should be used (LAMA or label or ID)")
    args = parser.parse_args()
    template = args.template
    assert template in ["LAMA", "label", "ID"]

    print("evaluating the baseline models: {}".format(baseline_lms))
    for lm_name in baseline_lms:
        lm_name_initials = get_initials(lm_name)
        model_path = "models/baselines/{}_{}".format(lm_name_initials, template)
        results_file_name = "baselines/{}_{}".format(lm_name_initials, template)
        model = AutoModelForMaskedLM.from_pretrained(lm_name)
        model.save_pretrained(model_path, config=True)
        tokenizer = AutoTokenizer.from_pretrained(lm_name)
        tokenizer.save_pretrained(model_path)
        start_evaluation(template, model_path, results_file_name)
        #start_evaluation(template, model_path, results_file_name, lama_uhn=True)


