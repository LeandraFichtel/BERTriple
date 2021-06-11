import json, os
from transformers import BertTokenizer



trex_folder = "/home/kalo/conferences/akbc2021/data/data/TREx/"

common_vocab_file = "/home/kalo/conferences/akbc2021/common_vocab_cased.txt"

trex_triple_file = "/home/kalo/conferences/akbc2021/data/"+"trex_test.json"
testdata = []


def remove_duplicates(triples):
    seen = set()
    new_l = []
    for d in triples:
        t = tuple(d.items())
        if t not in seen:
            seen.add(t)
            new_l.append(d)
    return new_l


def load_trex_file(trex_data, common_vocab):
    global testdata
    counter = 0
    for triple_dict in trex_data:
        #print(triple_dict)
        p_qid = triple_dict["predicate_id"]
        s_label = triple_dict["sub_label"]
        o_label = triple_dict["obj_label"]
        triple = {"subj": s_label, "prop": p_qid, "obj": o_label}
        if o_label in common_vocab:
            counter += 1
            testdata.append(triple)
    print("Added {} triples of property {}.".format(counter, p_qid))
    

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
                load_trex_file(trex_data, common_vocab)

    print("Lengths before removal {}.".format(len(testdata)))
    testdata = remove_duplicates(testdata)
    print("Lengths after removal {}.".format(len(testdata)))
    #with open(trex_triple_file, "w") as f:
    #    json.dump(testdata,f, indent=4, sort_keys=True)
print(testdata[0])
testdata.append({'subj': 'Belgium', 'prop': 'P17', 'obj': 'Belgium'})

traindata = json.load(open("/data/fichtel/BERTriple/training_datasets/wikidata41.json", "r"))
print(traindata.keys())
print(traindata["obj_queries"][0])
for datapoint in traindata["obj_queries"]:
    if datapoint in testdata:
        print("ALARM", datapoint)
        exit()
