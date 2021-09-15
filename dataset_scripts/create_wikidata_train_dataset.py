import json
import os

def create_entity2label_dict():
    # build a rdf dictionary from wikidata truthy
    entity_dictionary = {}
    with open("/data/wikidata/latest-truthy.nt", "r") as f:
        for line in f:
            if "rdf-schema#label" in line and "@en " in line:
                try:
                    s,p,o = line.split("> ")
                    s = s.replace("<", "")
                    o = o.replace("\"@en .\n", "")
                    start = o.index( "\"" ) + 1
                    #end = o.index( "\"", start )
                    o = o[start:]
                    s = s.encode('utf-8').decode('unicode-escape')
                    o = o.encode('utf-8').decode('unicode-escape')
                        
                    s = s.replace("http://www.wikidata.org/entity/","")
                    o = o.replace("http://www.wikidata.org/entity/","")
                    
                    entity_dictionary[s] = o
                except ValueError:
                    continue
    return entity_dictionary

def change_to_trex_label(entity_dictionary, redirects_path, LAMA_test_dataset_path):
    redirects = {}
    with open(redirects_path, "r") as f:
        for line in f:
            line = line.replace(".\n", "")
            s,p,o = line.split("> ")[0:3]
            s = s.replace("<", "")
            o = o.replace("<", "")            
            s = s.replace("http://www.wikidata.org/entity/","")
            o = o.replace("http://www.wikidata.org/entity/","")
            redirects[s] = o
    no_entry = []
    for file in os.listdir(LAMA_test_dataset_path):
        with open(LAMA_test_dataset_path+file+"/test.jsonl", "r") as f:
            trex_data = []
            for line in f.readlines():
                trex_data.append(json.loads(line))
            print("{} test triple for {}".format(len(trex_data), file))
            #change wikidata labels to TREX labels
            for triple_dict in trex_data:
                s_qid = triple_dict["sub_uri"]
                o_qid = triple_dict["obj_uri"]
                s_label_trex = triple_dict["sub_label"]
                o_label_trex = triple_dict["obj_label"]
                if s_qid in entity_dictionary:
                    s_label = entity_dictionary[s_qid]
                    #change s_label to TREX label to have the same labels of testset and trainset
                    if s_label_trex != s_label:
                        entity_dictionary[s_qid] = s_label_trex
                elif s_qid in redirects:
                    redirected_s_qid = redirects[s_qid]
                    if s_label_trex != entity_dictionary[redirected_s_qid]:
                        entity_dictionary[redirected_s_qid] = s_label_trex
                else:
                    print("no entry for", s_qid)
                
                if o_qid in entity_dictionary:
                    o_label = entity_dictionary[o_qid]
                    #change o_label to TREX label to have the same labels of testset and trainset
                    if o_label_trex != o_label:
                        entity_dictionary[o_qid] = o_label_trex
                elif o_qid in redirects:
                    redirected_o_qid = redirects[o_qid]
                    if o_label_trex != entity_dictionary[redirected_o_qid]:
                        entity_dictionary[redirected_o_qid] = o_label_trex
                else:
                    print("no entry for", o_qid)
    return entity_dictionary

def remove_duplicates(prop_triples):
    for p_qid in prop_triples:
        triples = prop_triples[p_qid]
        seen = set()
        new_l = []
        for d in triples:
            t = tuple(d.items())
            if t not in seen:
                seen.add(t)
                new_l.append(d)
        prop_triples[p_qid] = new_l
    return prop_triples

def substract_triples(original_prop_triples, substract_prop_triples):
    for p_qid in original_prop_triples:
        original_triples = original_prop_triples[p_qid]
        substract_triples = substract_prop_triples[p_qid]
        
        to_be_removed = set()
        for d in substract_triples:
            to_be_removed.add(tuple(d.items()))

        new_l = []
        for d in original_triples:
            t = tuple(d.items())
            if t not in to_be_removed:
                new_l.append(d)
        original_prop_triples[p_qid] = new_l
    return original_prop_triples

def load_trex_file(dataset, trex_data, vocab):
    counter = 0
    for triple_dict in trex_data:
        p_qid = triple_dict["predicate_id"]
        s_label = triple_dict["sub_label"]
        o_label = triple_dict["obj_label"]
        if o_label not in vocab:
            print("WARNING: object label ({}) is not in bert-base-cased vocab = is not one-token, but added triple anyway".format(o_label))
        triple = {"subj": s_label, "prop": p_qid, "obj": o_label}
        counter += 1
        if p_qid not in dataset:
            dataset[p_qid] = [triple]
        else:
            dataset[p_qid].append(triple)
    print("Added {} triples of property {}.".format(counter, p_qid))

def get_wikidata_train_data(vocab, entity2label_trexlabel, LAMA_test_dataset_path):
    train_dataset = {}
    train_dataset["subj_queries"] = {}
    train_dataset["obj_queries"] = {}
    props = {"P1001", "P106", "P1303", "P1376", "P1412", "P178", "P19", "P276", "P30", "P364", "P39", "P449", "P495", "P740", "P101", "P108", "P131", "P138", "P159", "P17", "P20", "P279", "P31", "P36", "P407", "P463", "P527", "P937", "P103", "P127", "P136", "P140", "P176", "P190", "P264", "P27", "P361", "P37", "P413", "P47", "P530"}
    with open("/data/wikidata/latest-truthy.nt", "r") as f:
        for i, line in enumerate(f):
            if i%1000000 == 0:
                print(i)
            if len(line.split("> <")) == 3:
                s,p,o = line.split("> <")
            else:
                #lines that do contain literals
                continue
            if "wikidata.org/entity" in o and "wikidata.org/entity" in s:
                s_qid = s.replace("<", "").replace(">", "").replace("http://www.wikidata.org/entity/","")
                o_qid = o.replace("> .\n","").replace("<", "").replace("http://www.wikidata.org/entity/","")
                p_qid = p.replace("http://www.wikidata.org/prop/direct/","").replace(">","")
                if p_qid in props:
                    if s_qid in entity2label_trexlabel and o_qid in entity2label_trexlabel:
                        s_label = entity2label_trexlabel[s_qid]
                        o_label = entity2label_trexlabel[o_qid]

                        if o_label in vocab:
                            triple = {"subj": s_label, "prop": p_qid, "obj": o_label}
                            if p_qid not in train_dataset["obj_queries"]:
                                train_dataset["obj_queries"][p_qid] = [triple]
                            else:
                                train_dataset["obj_queries"][p_qid].append(triple)

    #remove duplicates of triples
    train_dataset["obj_queries"] = remove_duplicates(train_dataset["obj_queries"])

    #get LAMA test dataset
    LAMA_test_dataset = {}
    for file in os.listdir(LAMA_test_dataset_path):
        with open(LAMA_test_dataset_path+file+"/test.jsonl", "r") as f:
            trex_data = []
            for line in f.readlines():
                trex_data.append(json.loads(line))
            load_trex_file(LAMA_test_dataset, trex_data, vocab)

    #remove testdata from training data
    count_all_triples = 0
    for p_qid in train_dataset["obj_queries"]:
        count_all_triples = count_all_triples + len(train_dataset["obj_queries"][p_qid])
    print("Lengths before substraction test from training data {}.".format(count_all_triples))
    
    train_dataset["obj_queries"] = substract_triples(train_dataset["obj_queries"], LAMA_test_dataset)
    count_all_triples = 0
    for p_qid in train_dataset["obj_queries"]:
        count_all_triples = count_all_triples + len(train_dataset["obj_queries"][p_qid])
    print("Lengths after substraction test from training data {}.".format(count_all_triples))
    return train_dataset

if __name__ == "__main__":
    entity_dictionary_path = "/data/kalo/akbc2021/entity_dictionary"
    if os.path.exists(entity_dictionary_path):
        print("load existing", entity_dictionary_path)
        with open(entity_dictionary_path, "r") as f:
            entity_dictionary = json.load(f)
    else:
        print("create", entity_dictionary_path)
        entity_dictionary = create_entity2label_dict()
        with open(entity_dictionary_path, 'w+') as fp:
            json.dump(entity_dictionary, fp, indent=4, sort_keys=True)

    entity2label_trexlabel_path = "/data/kalo/akbc2021/entity2label_trexlabel.json"
    if os.path.exists(entity2label_trexlabel_path):
        print("load existing", entity2label_trexlabel_path)
        with open(entity2label_trexlabel_path, "r") as f:
            entity2label_trexlabel = json.load(f)
    else:
        print("create", entity2label_trexlabel_path)
        autoprompt_dataset_path = "/data/kalo/akbc2021/AUTOPROMPT/original/"
        redirects_path = "/data/wikidata/sameAsLinks_wikidata.nt"
        entity2label_trexlabel = change_to_trex_label(entity_dictionary, redirects_path, autoprompt_dataset_path)
        with open(entity2label_trexlabel_path, "w+") as f:
            json.dump(entity2label_trexlabel, f)
    
    
    wikidata_train_dataset_path = "/data/kalo/akbc2021/training_datasets/wikidata41_all.json"
    if not os.path.exists(wikidata_train_dataset_path):
        print("create", wikidata_train_dataset_path)
        autoprompt_dataset_path = "/data/kalo/akbc2021/AUTOPROMPT/original/"
        vocab_file = "/home/fichtel/BERTriple/LAMA/pre-trained_language_models/bert/cased_L-12_H-768_A-12/vocab.txt"
        with open(vocab_file, "r") as f:
            vocab = set()
            for line in f:
                vocab.add(line.strip())
        wikidata_train_dataset = get_wikidata_train_data(vocab, entity2label_trexlabel, autoprompt_dataset_path)
        with open(wikidata_train_dataset_path, "w+") as f:
            json.dump(wikidata_train_dataset, f)