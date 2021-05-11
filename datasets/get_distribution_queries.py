import json
import os
from os import path

min_sample = "500"
string_token = "onetoken"
dictio_query_answer = json.load(open("/data/fichtel/projektarbeit/queries_{}_{}.json".format(string_token, min_sample), "r"))
dictio_id_label = json.load(open("/data/fichtel/projektarbeit/entity2label_onlyrdflabel.json", "r"))

dictio_prop_distibution = {}
count = 0
for query in dictio_query_answer:
    triple = query.split("_")
    subj = triple[0]
    prop = triple[1]
    obj = triple[2]
    if prop == "P276":
        count = count + 1
    if prop not in dictio_prop_distibution:
        dictio_prop_distibution[prop] = {"subj_label": {}, "obj_label": {}}
    #dictio_prop_distibution[prop]["count_queries"] = dictio_prop_distibution[prop]["count_queries"] + 1

    if subj == "?":
        used_labels = set()
        for answer in dictio_query_answer[query]:
            subj_label = dictio_id_label[answer][0].capitalize()
            if subj_label not in used_labels:
                if subj_label not in dictio_prop_distibution[prop]["subj_label"]:
                    dictio_prop_distibution[prop]["subj_label"][subj_label] = 1
                else:
                    dictio_prop_distibution[prop]["subj_label"][subj_label] = dictio_prop_distibution[prop]["subj_label"][subj_label] + 1
            used_labels.add(subj_label)
    elif obj == "?":
        used_labels = set()
        for answer in dictio_query_answer[query]:
            obj_label = dictio_id_label[answer][0]
            if obj_label not in used_labels:
                if obj_label not in dictio_prop_distibution[prop]["obj_label"]:
                    dictio_prop_distibution[prop]["obj_label"][obj_label] = 1
                else:
                    dictio_prop_distibution[prop]["obj_label"][obj_label] = dictio_prop_distibution[prop]["obj_label"][obj_label] + 1
            used_labels.add(obj_label)

for prop in dictio_prop_distibution:
    dictio_prop_distibution[prop]["subj_label"] = {k: v for k, v in sorted(dictio_prop_distibution[prop]["subj_label"].items(), reverse=True, key=lambda item: item[1])}
    dictio_prop_distibution[prop]["obj_label"] = {k: v for k, v in sorted(dictio_prop_distibution[prop]["obj_label"].items(), reverse=True, key=lambda item: item[1])}

if path.exists("/data/fichtel/projektarbeit/distribution_queries_{}_{}.json".format(string_token, min_sample)):
    print("removed distribution_queries_{}_{}.json".format(string_token, min_sample))
    os.remove("/data/fichtel/projektarbeit/distribution_queries_{}_{}.json".format(string_token, min_sample))
distribution_file = open("/data/fichtel/projektarbeit/distribution_queries_{}_{}.json".format(string_token, min_sample), "w")
json.dump(dictio_prop_distibution, distribution_file)