import json

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

def get_wikidata_train_data():
    training_data = {}
    training_data["subj_queries"] = {}
    training_data["obj_queries"] = {}
    properties = {"P1001", "P106", "P1303", "P1376", "P1412", "P178", "P19", "P276", "P30", "P364", "P39", "P449", "P495", "P740", "P101", "P108", "P131", "P138", "P159", "P17", "P20", "P279", "P31", "P36", "P407", "P463", "P527", "P937", "P103", "P127", "P136", "P140", "P176", "P190", "P264", "P27", "P361", "P37", "P413", "P47", "P530"}
    #wikidata_training_path = "/home/kalo/conferences/akbc2021/data/data/TREx/wikidata_training.json"
    common_vocab_file = "/home/kalo/conferences/akbc2021/common_vocab_cased.txt"
    wikidata_training_path = "/data/fichtel/BERTriple/training_datasets/wikidata41.json"
    entity_dictionary_trex_label_file = "/data/fichtel/BERTriple/entity2label_trexlabel.json"
    trex_triple_file = "/data/fichtel/BERTriple/test_datasets/"+"LAMA_trex_test.json"

    with open(entity_dictionary_trex_label_file, "r") as f:
        entity_dictionary_trex_label = json.load(f)
        with open(common_vocab_file, "r") as f:
            common_vocab = set()
            for line in f:
                common_vocab.add(line.strip())
            with open("/data/wikidata/latest-truthy.nt", "r") as f:
                for i, line in enumerate(f):
                    if i%1000000 == 0:
                        print(i)
                    s,p,o = line.split("> <")
                    if "wikidata.org/entity" in o and "wikidata.org/entity" in s:
                        s_qid = s.replace("<", "").replace("http://www.wikidata.org/entity/","")
                        o_qid = o.replace("> .\n","").replace("http://www.wikidata.org/entity/","")
                        p_qid = p.replace("http://www.wikidata.org/prop/direct/","").replace(">","")
                        if p_qid in properties:
                            if s_qid in entity_dictionary_trex_label and o_qid in entity_dictionary_trex_label:
                                s_label = entity_dictionary_trex_label[s_qid]
                                o_label = entity_dictionary_trex_label[o_qid]
                            
                                if o_label in common_vocab:
                                    triple = {"subj": s_label, "prop": p_qid, "obj": o_label}
                                    if p_qid not in training_data["obj_queries"]:
                                        training_data["obj_queries"][p_qid] = [triple]
                                    else:
                                        training_data["obj_queries"][p_qid].append(triple)

    #remove duplicates of triples
    training_data["obj_queries"] = remove_duplicates(training_data["obj_queries"])

    with open(trex_triple_file, "r") as f:
        trex_test = json.load(f)
        #remove testdata from training data
        count_all_triples = 0
        for p_qid in training_data["obj_queries"]:
            for triple in training_data["obj_queries"][p_qid]:
                count_all_triples = count_all_triples + 1
        print("Lengths before substraction test from training data {}.".format(count_all_triples))
        
        training_data["obj_queries"] = substract_triples(training_data["obj_queries"], trex_test)
        count_all_triples = 0
        for p_qid in training_data["obj_queries"]:
            for triple in training_data["obj_queries"][p_qid]:
                count_all_triples = count_all_triples + 1
        print("Lengths after substraction test from training data {}.".format(count_all_triples))

    with open(wikidata_training_path, "w+") as f:
        json.dump(training_data, f, indent=4, sort_keys=True)

if __name__ == "__main__":
    get_wikidata_train_data()