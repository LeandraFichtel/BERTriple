import json
import os

def create_entity2label_dict():
    # build a rdf dictionary from wikidata truthy
    entity_dictionary_file = "/home/kalo/conferences/akbc2021/data/data/entity_dictionary"
    entity_dictionary = {}
    with open("/data/wikidata/latest-truthy.nt", "r") as f:
        for line in f:
            if "rdf-schema#label" in line and "@en " in line:
                try:
                    s,p ,o = line.split("> ")
                    s = s.replace("<", "")
                    o = o.replace("@en", "")
                    start = o.index( "\"" ) + 1
                    end = o.index( "\"", start )
                    o = o[start:end]
                    s = s.encode('utf-8').decode('unicode-escape')
                    o = o.encode('utf-8').decode('unicode-escape')
                        
                    s = s.replace("http://www.wikidata.org/entity/","")
                    o = o.replace("http://www.wikidata.org/entity/","")
                    
                    entity_dictionary[s] = o
                except ValueError:
                    continue
    with open(entity_dictionary_file, 'w+') as fp:
        json.dump(entity_dictionary, fp, indent=4, sort_keys=True)
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

def get_label(uri,entity_dictionary):
        return entity_dictionary[uri]

def load_trex_file(LAMA_test, trex_data, entity_dictionary, common_vocab):
        counter = 0
        for triple_dict in trex_data:
            s_qid = triple_dict["sub_uri"]
            p_qid = triple_dict["predicate_id"]
            o_qid = triple_dict["obj_uri"]
            s_label_trex = triple_dict["sub_label"]
            o_label_trex = triple_dict["obj_label"]
            if s_qid in entity_dictionary:
                s_label = entity_dictionary[s_qid]
                #change s_label to TREX label to have the same labels of testset and trainset
                if s_label_trex != s_label:
                    entity_dictionary[s_qid] = s_label_trex
                    
                if o_qid in entity_dictionary:
                    o_label = entity_dictionary[o_qid]
                    #change o_label to TREX label to have the same labels of testset and trainset
                    if o_label_trex != o_label:
                        entity_dictionary[o_qid] = o_label_trex

                    #add tripel with current labels if the o_label is in common_vocab to have obj_queries
                    s_label = entity_dictionary[s_qid]
                    o_label = entity_dictionary[o_qid]
                    if o_label in common_vocab:
                        triple = {"subj": s_label, "prop": p_qid, "obj": o_label}
                        counter += 1
                        if p_qid not in LAMA_test:
                            LAMA_test[p_qid] = [triple]
                        else:
                            LAMA_test[p_qid].append(triple)
                else:
                    print("no label found in wikidata:(", o_qid, o_label_trex)
            else:
                print("no label found wikidata :(", s_qid, s_label_trex)
        print("Added {} triples of property {}.".format(counter, p_qid))

def get_LAMA_test_data():
    trex_folder = "/home/kalo/conferences/akbc2021/data/data/TREx/"
    entity_dictionary_file = "/home/kalo/conferences/akbc2021/data/data/entity_dictionary"
    common_vocab_file = "/home/kalo/conferences/akbc2021/common_vocab_cased.txt"
    #trex_triple_file = "/home/kalo/conferences/akbc2021/data/"+"LAMA_test.json"
    trex_triple_file = "/data/fichtel/BERTriple/test_datasets/"+"LAMA_trex_test.json"
    entity_dictionary_trex_label_file = "/data/fichtel/BERTriple/entity2label_trexlabel.json"
    LAMA_test = {}

    with open(entity_dictionary_file, "r") as f:
        entity_dictionary = json.load(f)
        with open(common_vocab_file, "r") as f:
            common_vocab = set()
            for line in f:
                common_vocab.add(line.strip())
            for file in os.listdir(trex_folder):
                if file != "wikidata_training.json":
                    with open(trex_folder+file, "r") as f:
                        trex_data = []
                        for line in f.readlines():
                            trex_data.append(json.loads(line))
                        load_trex_file(LAMA_test, trex_data, entity_dictionary, common_vocab)
            
            count_all_triples = 0
            for p_qid in LAMA_test:
                for triple in LAMA_test[p_qid]:
                    count_all_triples = count_all_triples + 1
            print("Lengths before removal {}.".format(count_all_triples))
                
            #remove duplicates of triples
            LAMA_test = remove_duplicates(LAMA_test)
            count_all_triples = 0
            for p_qid in LAMA_test:
                for triple in LAMA_test[p_qid]:
                    count_all_triples = count_all_triples + 1
            print("Lengths after removal {}.".format(count_all_triples))
            
    #save the test dataset of LAMA
    with open(trex_triple_file, "w") as f:
        json.dump(LAMA_test, f, indent=4, sort_keys=True)

    #save the dictionary where the wikidata rdf labels are replaced with the trex labels when they were different
    with open(entity_dictionary_trex_label_file, 'w') as fp:
        json.dump(entity_dictionary, fp, indent=4, sort_keys=True)

if __name__ == "__main__":
    #create_entity2label_dict()
    get_LAMA_test_data()
