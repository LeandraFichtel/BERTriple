import argparse
import os
import json
import re
import shutil
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
from tqdm import tqdm    

def valid_label(obj_label, vocab_type, common_vocab, tokenizer):
    if vocab_type == "common":
        return obj_label in common_vocab
    elif vocab_type == "different":
        return len(tokenizer(obj_label, add_special_tokens=False)["input_ids"]) == 1

def start_evaluation(template, vocab_type, model_path, results_file_name, omitted_props=None, lama_uhn=False):
    if omitted_props:
        props = omitted_props
    else:
        props = ["P1001", "P106", "P1303", "P1376", "P1412", "P178", "P19", "P276", "P30", "P364", "P39", "P449", "P495", "P740", "P101", "P108", "P131", "P138", "P159", "P17", "P20", "P279", "P31", "P36", "P407", "P463", "P527", "P937", "P103", "P127", "P136", "P140", "P176", "P190", "P264", "P27", "P361", "P37", "P413", "P47", "P530"]
    assert len(props) > 0
    print("\nconsidering {} properties".format(len(props)))
    #use the huggingface pipeline to predict tokens for the mask-token
    fill_mask = pipeline("fill-mask", model=model_path, device=0)
    #choose the dataset und logging path
    if lama_uhn:
        test_data_path = "data/test_datasets/TREx_UHN/"
        results_file_name = results_file_name+"_uhn"
        logging_path = model_path+"/logging_lama_uhn"
        print("T-REx UHN evaluation: {}".format(model_path))
    else:
        test_data_path = "data/test_datasets/TREx/"
        logging_path = model_path+"/logging_lama"
        print("T-REx evaluation: {}".format(model_path))
        
    #load LAMA common vocab to filter results, e.g. not have stop_words like "nothing" or "him" as predictions
    LAMA_common_vocab = set(open("data/LAMA_common_vocab_cased.txt", "r").read().splitlines())
    if os.path.exists(logging_path):
        print("remove logging dir of model")
        shutil.rmtree(logging_path)
    os.mkdir(logging_path)
    #get common_vocab and tokenizer of current model
    common_vocab = set(open("data/common_vocab_cased.txt", "r").read().splitlines())
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    #iterate through all test_data for all properties
    avg_prec_at_1 = 0
    results_file = open("results/{}.csv".format(results_file_name), "w+")
    for file_name in os.listdir(test_data_path):
        prop = file_name.split(".")[0]
        if prop in props: 
            logging_file = open(logging_path+"/"+file_name, "w+")
            #a triple is valid if the object label of the triple is contained in the vocab
            valid_triples = []
            with open(test_data_path+file_name, "r") as file:
                for line in file:
                    dictio_triple = json.loads(line)
                    if valid_label(dictio_triple["obj_label"], vocab_type, common_vocab, tokenizer):
                        subj_label = dictio_triple["sub_label"]
                        obj_label = dictio_triple["obj_label"]
                        subj_qid = dictio_triple["sub_uri"]
                        obj_qid = dictio_triple["obj_uri"]
                        valid_triples.append({"subj_label": subj_label, "subj_qid": subj_qid, "obj_label": obj_label, "obj_qid": obj_qid})            
            print("property {}: {} test queries ".format(prop, len(valid_triples)))
            #get the right template for the prop
            dictio_prop_template = json.load(open("data/templates.json", "r"))
            query_template = dictio_prop_template[prop][template]
            print("using {} template: {}".format(template, query_template))
            #evaluation (precision@1 per prop and overall avg precison@1)
            #note: if len(valid_triples) == 0, then the prec_at_1 for this property is 0
            prec_at_1 = 0
            for dictio_triple in tqdm(valid_triples):
                mask_query = query_template.replace("[X]", dictio_triple["subj_label"]).replace("[Y]", fill_mask.tokenizer.mask_token)
                found_valid_token = False
                predict_try_no = 100
                index = 0
                while not found_valid_token:
                    dictio_result = fill_mask(mask_query, top_k=predict_try_no)
                    for result in dictio_result[index:]:
                        #delete blankspace at beginning and at end of token
                        predicted_token = result["token_str"].strip()
                        if predicted_token in LAMA_common_vocab:
                            found_valid_token = True
                            #create dictio for logging
                            dictio_logging = {}
                            dictio_triple["masked_sentences"] = [mask_query]
                            dictio_logging["query"] = dictio_triple
                            dictio_logging["result"] = result 
                            #check whether the predicted token is correct
                            if predicted_token == dictio_triple["obj_label"]:
                                prec_at_1 = prec_at_1 + 1
                                dictio_logging["prec@1"] = 1
                            else:
                                dictio_logging["prec@1"] = 0
                            break
                    index = predict_try_no
                    predict_try_no += predict_try_no*2
                assert found_valid_token
                json.dump(dictio_logging, logging_file)
                logging_file.write("\n")
            #calculate precision@1 of each prop averaged over all test queries
            if len(valid_triples) > 0:
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
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-lm_names', default=[], nargs='+', help="names of the baseline language models (use the huggingface identifiers: https://huggingface.co/transformers/pretrained_models.html)")
    parser.add_argument('-vocab', help="set whether a common vocab or different vocabs of the baseline language models should be used")
    args = parser.parse_args()
    baseline_lms = args.lm_names
    #check that all baseline language models are cased because the templates for the queries are cased
    for lm_name in baseline_lms:
        assert "uncased" not in lm_name
    vocab_type = args.vocab
    assert vocab_type in ["common", "different"]

    if vocab_type == "common":
        user_input = input("All baseline models and results will be deleted. Is this okay? :) [N/y]")
        if user_input != "y":
            exit("Okay, it's not okay. I stopped the script for you. :)")
        #remove and create model dir and result dir for baseline models
        if os.path.exists("models/baselines_common_vocab/"):
            print("remove dir of common baseline models")
            shutil.rmtree("models/baselines_common_vocab/")
        os.mkdir("models/baselines_common_vocab/")
        if os.path.exists("results/baselines_common_vocab/"):
            print("remove results dir of common baseline models")
            shutil.rmtree("results/baselines_common_vocab/")
        os.mkdir("results/baselines_common_vocab/")
        for file_name in os.listdir("data/train_datasets"):
            if "common" in file_name:
                print("remove common vocab dataset", "data/train_datasets/"+file_name)
                os.remove("data/train_datasets/"+file_name)
        #use the common vocab of LAMA as basis and find intersection of all vocabs of baseline language models
        common_vocab = set(open("data/LAMA_common_vocab_cased.txt", "r").read().splitlines())
        print("length of common vocab before intersection:", len(common_vocab))
        for lm_name in baseline_lms:
            vocab = common_vocab.copy()
            tokenizer = AutoTokenizer.from_pretrained(lm_name, add_prefix_space=True)
            for token in common_vocab:
                if len(tokenizer(token, add_special_tokens=False)["input_ids"]) > 1:
                    vocab.remove(token)
            common_vocab = common_vocab.intersection(vocab)
            print("length of common vocab after intersection with {}: {}".format(lm_name, len(common_vocab)))
        
        assert len(common_vocab) > 0
        print("save common_vocab_cased.txt")
        with open("data/common_vocab_cased.txt", "w+") as common_vocab_file:
            for token in common_vocab:
                common_vocab_file.write(token+"\n")
    elif vocab_type == "different":
        if not os.path.exists("models/baselines_different_vocab/"):
            os.mkdir("models/baselines_different_vocab/")
        if not os.path.exists("results/baselines_different_vocab/"):
            os.mkdir("results/baselines_different_vocab/")
    
    print("evaluating the baseline models: {}".format(baseline_lms))
    for lm_name in baseline_lms:
        lm_name_initials = get_initials(lm_name)
        for template in ["LAMA", "label", "ID"]:
            model_path = "models/baselines_{}_vocab/{}_{}".format(vocab_type, lm_name_initials, template)
            results_file_name = "baselines_{}_vocab/{}_{}".format(vocab_type, lm_name_initials, template)
            model = AutoModelForMaskedLM.from_pretrained(lm_name)
            model.save_pretrained(model_path, config=True)
            tokenizer = AutoTokenizer.from_pretrained(lm_name)
            tokenizer.save_pretrained(model_path)
            start_evaluation(template, vocab_type, model_path, results_file_name)
            start_evaluation(template, vocab_type, model_path, results_file_name, lama_uhn=True)


