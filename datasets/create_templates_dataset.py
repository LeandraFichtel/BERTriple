import os
import json
import operator

def get_dictio_prop_data():
    dictio_prop_template = {}
    dictio_prop_label = {}
    file_relations = open("/data/fichtel/projektarbeit/relations.jsonl")
    min_sample = "500"
    auto_templates = json.load(open("/data/fichtel/projektarbeit/auto_templates/auto_templates_{}.json".format(min_sample), "r"))
    for line in file_relations:
        data = json.loads(line)
        prop = data["relation"]
        label = data["label"]
        template = data["template"]
        dictio_prop_template[prop] = {}
        dictio_prop_template[prop]["LAMA"] = template.replace("[X]", "[S]").replace("[Y]", "[O]")
        dictio_prop_template[prop]["label"] = "[S] {} [O] .".format(label)
        dictio_prop_template[prop]["auto"] = {}
        dictio_prop_template[prop]["auto"][min_sample] = {}
        for sample in auto_templates:
            if prop in auto_templates[sample]:
                dictio_prop_template[prop]["auto"][min_sample][sample] = max(auto_templates[sample][prop].items(), key=operator.itemgetter(1))[0]
        dictio_prop_label[prop] = label
    return dictio_prop_template, dictio_prop_label
     
    

if __name__ == '__main__':
    if os.path.exists("/data/fichtel/projektarbeit/templates.json"):
        os.remove("/data/fichtel/projektarbeit/templates.json")
        print("removed templates.json")
    templates_file = open("/data/fichtel/projektarbeit/templates.json", "w")
    if os.path.exists("/data/fichtel/projektarbeit/prop2label.json"):
        os.remove("/data/fichtel/projektarbeit/prop2label.json")
        print("removed prop2label.json")
    labels_file = open("/data/fichtel/projektarbeit/prop2label.json", "w")

    dictio_prop_template, dictio_prop_label = get_dictio_prop_data()
    
    json.dump(dictio_prop_template, templates_file)
    json.dump(dictio_prop_label, labels_file)
