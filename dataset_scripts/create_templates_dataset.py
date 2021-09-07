import os
import json
import operator

def get_dictio_prop_data():
    dictio_prop_template = {}
    file_relations = open("/data/fichtel/BERTriple/relations.jsonl")
    for line in file_relations:
        data = json.loads(line)
        prop = data["relation"]
        label = data["label"]
        template = data["template"]
        dictio_prop_template[prop] = {}
        dictio_prop_template[prop]["LAMA"] = template
        dictio_prop_template[prop]["label"] = "[X] {} [Y] .".format(label)
        dictio_prop_template[prop]["ID"] = "[X] {} [Y] .".format(prop)
    return dictio_prop_template

if __name__ == '__main__':
    with open("../data/templates.json", "w+") as templates_file:
        dictio_prop_template = get_dictio_prop_data()
        json.dump(dictio_prop_template, templates_file)
