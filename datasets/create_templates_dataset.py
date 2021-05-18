import os
import json
import operator

def get_dictio_prop_data():
    dictio_prop_template = {}
    dictio_prop_label = {}
    file_relations = open("/data/fichtel/BERTriple/relations.jsonl")
    for line in file_relations:
        data = json.loads(line)
        prop = data["relation"]
        label = data["label"]
        template = data["template"]
        dictio_prop_template[prop] = {}
        dictio_prop_template[prop]["LAMA"] = template.replace("[X]", "[S]").replace("[Y]", "[O]")
        dictio_prop_template[prop]["label"] = "[S] {} [O] .".format(label)
    return dictio_prop_template

if __name__ == '__main__':
    if os.path.exists("/data/fichtel/BERTriple/templates.json"):
        os.remove("/data/fichtel/BERTriple/templates.json")
        print("removed /data/fichtel/BERTriple/templates.json")
    templates_file = open("/data/fichtel/BERTriple/templates.json", "w")
    
    dictio_prop_template = get_dictio_prop_data()
    json.dump(dictio_prop_template, templates_file)
