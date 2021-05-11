import json
import sys
import os
import dill
import random
from itertools import islice
from transformers import BertTokenizer
from transformers import pipeline, BertForMaskedLM
from os import path
import copy

def get_ents_in_sent(
        sent_b,
        ents,
        mandatory_uri_substr="wikidata.org/entity/"):
    # filter entities which are in boundaries
    # and whose uris contain the mandatory uri substring
    # (default: to filter only entities, not relations)
    ents_copy = copy.deepcopy(ents)
    sent_ents = list(filter(
        lambda ent: (
            ent["boundaries"][0] >= sent_b[0] and
            ent["boundaries"][0] < sent_b[1] and
            mandatory_uri_substr in ent["uri"]
            ),
        ents_copy
        ))

    # align boundaries from abstract level to sentence level
    for i in range(0, len(sent_ents)):
        sent_ents[i]["boundaries"][0] -= sent_b[0]
        sent_ents[i]["boundaries"][1] -= sent_b[0]
            
    return sent_ents


def index_sentences(dictio_id_label, dictio_prop_triple):
    maxLength = 10
    index = {}
    for i, file in enumerate(os.listdir("/data/fichtel/projektarbeit/trex/")):
        print("index sentences in file {}".format(i))
        path = os.path.join("/data/fichtel/projektarbeit/trex/", file)
        trex_dataset = json.load(open(path, "r"))
        # go through every abstract associated to an entity
        for abstract in trex_dataset:
            # get sentences from "sentences_boundaries"
            for sent_b in abstract["sentences_boundaries"]:
                sent = abstract["text"][sent_b[0]:sent_b[1]]
                if len(sent.split()) <= maxLength:
                    # get entities in this boundary
                    sent_ents = get_ents_in_sent(sent_b, abstract["entities"])
                    # index sentences containing entity pairs (e1,e2) from input relation r
                    if len(sent_ents) == 2:
                        for e1 in sent_ents:
                            for e2 in sent_ents:
                                if e1 != e2:
                                    try:
                                        e1['uri'] = e1['uri'].replace("http://www.wikidata.org/entity/", "")
                                        e2['uri'] = e2['uri'].replace("http://www.wikidata.org/entity/", "")
                                        #check for sentence whether it belongs to one of the input relations in props
                                        for prop in dictio_prop_triple:
                                            entityPairs = dictio_prop_triple[prop]
                                            if (e1['uri'],e2['uri']) in entityPairs:
                                                    #check if entities are overlapping (e.g. "This novel won the John Newbery Medal in 1996.", e1=John Newbery, e2=Newbery Medal)
                                                overlapping = False
                                                if e1['boundaries'][0] < e2['boundaries'][0] and e2['boundaries'][0] <= e1['boundaries'][1]:
                                                    overlapping = True
                                                elif e1['boundaries'][0] > e2['boundaries'][0] and e1['boundaries'][0] <= e2['boundaries'][1]:
                                                    overlapping = True
                                                #check if label extracted with boundaries is equal to surfaceform
                                                e1_too_short_surfaceform = False
                                                if e1['boundaries'][1] - e1['boundaries'][0] != len(dictio_id_label[e1['uri']][0]):
                                                    e1_label_token = dictio_id_label[e1['uri']][0].split(" ")
                                                    for token in e1_label_token:
                                                        if token in e1["surfaceform"].split(" "):
                                                            e1_too_short_surfaceform = True
                                                            break
                                                e2_too_short_surfaceform = False
                                                if e2['boundaries'][1] - e2['boundaries'][0] != len(dictio_id_label[e2['uri']][0]):
                                                    e2_label_token = dictio_id_label[e2['uri']][0].split(" ")
                                                    for token in e2_label_token:
                                                        if token in e2["surfaceform"].split(" "):
                                                            e2_too_short_surfaceform = True
                                                            break
                                                if e1_too_short_surfaceform == False and e2_too_short_surfaceform == False and overlapping == False:
                                                    entry = {}
                                                    #replace the current surfaceform with the label of wikidata, which are used
                                                    if e1['boundaries'][0] < e2['boundaries'][0]:
                                                        sentence = sent[:e1['boundaries'][0]] + dictio_id_label[e1['uri']][0] + sent[e1['boundaries'][1]:e2['boundaries'][0]] + dictio_id_label[e2['uri']][0] + sent[e2['boundaries'][1]:]
                                                    elif e1['boundaries'][0] > e2['boundaries'][0]:
                                                        sentence = sent[:e2['boundaries'][0]] + dictio_id_label[e2['uri']][0] + sent[e2['boundaries'][1]:e1['boundaries'][0]] + dictio_id_label[e1['uri']][0] + sent[e1['boundaries'][1]:]
                                                    else:
                                                        sentence = -1
                                                        print("subject==object --> sentence: {}, surfaceform: {}".format(sent ,e1["surfaceform"]))
                                                    if sentence != -1:
                                                        entry['sentence'] = sentence
                                                        entry['entities'] = (e1['uri'],e2['uri'])

                                                        #adjust boundaries
                                                        e1_copy = copy.deepcopy(e1)
                                                        e2_copy = copy.deepcopy(e2)
                                                        if e1_copy['boundaries'][0] < e2_copy['boundaries'][0]:
                                                            e1_boundary_diff = len(dictio_id_label[e1_copy['uri']][0]) - (e1_copy['boundaries'][1] - e1_copy['boundaries'][0])
                                                            entry['e1'] = e1_copy['boundaries'][:]
                                                            entry['e1'][1] = entry['e1'][0] + len(dictio_id_label[e1_copy['uri']][0])
                                                            entry['e2'] = e2_copy['boundaries']
                                                            entry['e2'][0] = entry['e2'][0] + e1_boundary_diff
                                                            entry['e2'][1] = entry['e2'][0] + len(dictio_id_label[e2_copy['uri']][0])
                                                        elif e1_copy['boundaries'][0] > e2_copy['boundaries'][0]:
                                                            e2_boundary_diff = len(dictio_id_label[e2_copy['uri']][0]) - (e2_copy['boundaries'][1] - e2_copy['boundaries'][0])
                                                            entry['e2'] = e2_copy['boundaries'][:]
                                                            entry['e2'][1] = entry['e2'][0] + len(dictio_id_label[e2_copy['uri']][0])
                                                            entry['e1'] = e1_copy['boundaries']
                                                            entry['e1'][0] = entry['e1'][0] + e2_boundary_diff
                                                            entry['e1'][1] = entry['e1'][0] + len(dictio_id_label[e1_copy['uri']][0])
                                                        if prop in index:
                                                            index[prop].append(entry)
                                                        else:
                                                            index[prop] = [entry]
                                    except KeyError:
                                        print("WARNING: KeyError")
                                        continue
    return index

from difflib import SequenceMatcher
def similar_string(a, b):
    return SequenceMatcher(None, a, b).ratio()

def filter_similar_templates(index, dictio_id_label):
    filtered_index = {}
    for prop in index:
        similar_templates = {}
        unique_templates = {}
        filtered_index[prop] = []
        for i, index_entry in enumerate(index[prop]):
            orig_sentence = index_entry["sentence"]
            if index_entry['e1'][0] < index_entry['e2'][0]:
                subject_object_template = orig_sentence[:index_entry['e1'][0]] + "[S]" + orig_sentence[index_entry['e1'][1]:index_entry['e2'][0]] + "[O]" + orig_sentence[index_entry['e2'][1]:]
            else:
                subject_object_template = orig_sentence[:index_entry['e2'][0]] + "[O]" + orig_sentence[index_entry['e2'][1]:index_entry['e1'][0]] + "[S]" + orig_sentence[index_entry['e1'][1]:]
            if subject_object_template not in unique_templates:
                unique_templates[subject_object_template] = index_entry

        for template in unique_templates:
            find_suitable_key_template = False
            for key_template in similar_templates:
                if find_suitable_key_template == True:
                    break
                elif similar_string(key_template, template) > 0.8:
                    find_suitable_key_template = True
                    similar_templates[key_template].append(template)
                else:
                    for similar_template in similar_templates[key_template]:
                        if similar_string(similar_template, template) > 0.8:
                            find_suitable_key_template = True
                            similar_templates[key_template].append(template)
                            break
            if find_suitable_key_template == False:
                similar_templates[template] = []

        #for sent in similar_templates:
        #    print("{} --> {}\n".format(sent, similar_templates[sent]))
        #print(len(index))
        #print(len(similar_templates))
        for key_template in similar_templates:
            shortest_template = key_template
            for similar in similar_templates[key_template]:
                    if len(similar) < len(shortest_template):
                        shortest_template = similar
            index_entry = unique_templates[shortest_template]
            index_entry["template"] = shortest_template
            filtered_index[prop].append(index_entry)
    #print(filtered_index)
    return filtered_index


def check_tokenizer(filtered_index, tokenizer):
    for prop in filtered_index:
        filtered_index_copy = filtered_index[prop].copy()
        for index_entry in filtered_index_copy:
            template = index_entry["template"]
            tokenized_template = tokenizer.tokenize(template)
            if "[UNK]" in tokenized_template:
                print(tokenized_template)
                filtered_index[prop].remove(index_entry)
    return filtered_index

#rank templates with regard to first metric of paper from Bouroui et al.
def rank_template(index_entry, random_onetoken_triple, unmasker):
    output_rank = 0
    template = index_entry["template"]
    for (subj_label, obj_label) in random_onetoken_triple["subj"]:
        subj_query = (template.replace("[S]", "[MASK]")).replace("[O]", obj_label)
        #get answers of lm for [MASK] token
        subj_lm_results = unmasker(subj_query)
        for result in subj_lm_results:
            lm_answer = result["token_str"]
            if lm_answer.lower() == subj_label.lower():
                output_rank = output_rank + 1
                break
    for (subj_label, obj_label) in random_onetoken_triple["obj"]:
        obj_query = (template.replace("[S]", subj_label.capitalize())).replace("[O]", "[MASK]")
        #get answers of lm for [MASK] token
        obj_lm_results = unmasker(obj_query)
        for result in obj_lm_results:
            lm_answer = result["token_str"]
            if lm_answer.lower() == obj_label.lower():
                output_rank = output_rank + 1
                break
    return output_rank

def get_all_not_used_triplets(string_token):
    wikidata_all_triplets = open("/data/fichtel/projektarbeit/gold_dataset.nt", "r")
    dictio_prop_template = json.load(open("/data/fichtel/projektarbeit/templates.json", "r"))

    #collect all triplets of wikidata in a dict
    #I do not use wikidata_onetoken_missing_all.json because to find enough matching sentences there are not onetoken triplets like (Angela Merkel, speaks, english) necessary
    dictio_prop_triple = {}
    for line in wikidata_all_triplets:
        triple = ((line.replace("\n", "")).replace(".", "")).split(" ")
        prop = str(triple[1]).split('/')[-1].replace('>', "")
        if prop in dictio_prop_template:
            subj = str(triple[0]).split('/')[-1].replace('>', "")
            obj = str(triple[2]).split('/')[-1].replace('>', "")
            if prop not in dictio_prop_triple:
                dictio_prop_triple[prop] = set()
            dictio_prop_triple[prop].add((subj, obj))
    #get the used triplets of the queries
    dictio_prop_query_answer = json.load(open("/data/fichtel/projektarbeit/queries_{}_all.json".format(string_token), "r"))       
    dictio_prop_used_triple = {}
    for prop in dictio_prop_query_answer:
        dictio_prop_used_triple[prop] = set()
        #get query
        for query in dictio_prop_query_answer[prop]:
            for answer in dictio_prop_query_answer[prop][query]:
                triple = query.replace("?", answer).split("_")
                subj = triple[0]
                obj = triple[2]
                subj_label = dictio_id_label[subj][0]
                obj_label = dictio_id_label[obj][0]
                dictio_prop_used_triple[prop].add((subj_label, obj_label))
    #get the not used triple to use them to find the sentences for the templates
    dictio_prop_not_used_triple = {}
    for prop in dictio_prop_triple:
        not_used_triple = dictio_prop_triple[prop] - dictio_prop_used_triple[prop]
        dictio_prop_not_used_triple[prop] = not_used_triple
    
    return dictio_prop_not_used_triple            

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    min_sample = 500
    samples = [1, 5, 10, 20, 30, 50, 100, 200, 300, 400, 500]
    string_token = "onetoken"
    lm_name = "bert-base-cased"

    dictio_id_label = json.load(open("/data/fichtel/projektarbeit/entity2label_onlyrdflabel.json", "r"))
    if not path.exists("/data/fichtel/projektarbeit/trex_filtered_index.json"):
        print("create new filtered trex index")
        #all not used triple (ids) for each prop
        dictio_prop_all_triple = get_all_not_used_triplets(string_token)
        index = index_sentences(dictio_id_label, dictio_prop_all_triple)
        #initializing bert tokenizer
        filtered_index = filter_similar_templates(index, dictio_id_label)
        tokenizer = BertTokenizer.from_pretrained(lm_name)
        filtered_index = check_tokenizer(filtered_index, tokenizer)
        trex_filtered_index_file = open("/data/fichtel/projektarbeit/trex_filtered_index.json", "w")
        json.dump(filtered_index, trex_filtered_index_file)
        trex_filtered_index_file.close()
    else:
        print("load existing trex index")
        filtered_index = json.load(open("/data/fichtel/projektarbeit/trex_filtered_index.json", "r"))
    
    print("found possible sentences for templates")
    
    
    dictio_prop_rank_templates = {}
    #initializing bert unmasker
    unmasker = pipeline('fill-mask', model=lm_name, device=0, top_k=50)
    for sample in samples:
        print("sample: {}".format(sample))
        dictio_prop_rank_templates[str(sample)] = {}
        #only onetoken not used triple (onetoken label) for each prop
        dictio_prop_onetoken_tuples = json.load(open("/data/fichtel/projektarbeit/auto_templates/tuples_for_ranking_{}_{}_{}_{}.json".format(string_token, "auto", min_sample, sample), "r"))
        for prop in dictio_prop_onetoken_tuples:
            if prop not in filtered_index:
                print("no sentences for property {} :(".format(prop))
                continue

            intermediate_results = {}
            print("start to rank {} templates for property {}".format(len(filtered_index[prop]), prop))
            for index_entry in filtered_index[prop]:
                score = rank_template(index_entry, dictio_prop_onetoken_tuples[prop], unmasker)
                template = index_entry["template"]
                intermediate_results[template] = score
            sorted_results = {k: v for k, v in sorted(intermediate_results.items(), reverse=True, key=lambda item: item[1])}
            dictio_prop_rank_templates[str(sample)][prop] = sorted_results
        if os.path.exists("/data/fichtel/projektarbeit/auto_templates/auto_templates_{}_{}.json".format(min_sample, sample)):
            os.remove("/data/fichtel/projektarbeit/auto_templates/auto_templates_{}_{}.json".format(min_sample, sample))    
        template_file = open("/data/fichtel/projektarbeit/auto_templates/auto_templates_{}_{}.json".format(min_sample, sample), "w")
        json.dump(dictio_prop_rank_templates[str(sample)], template_file)
    if os.path.exists("/data/fichtel/projektarbeit/auto_templates/auto_templates_{}.json".format(min_sample)):
        os.remove("/data/fichtel/projektarbeit/auto_templates/auto_templates_{}.json".format(min_sample))    
    template_file = open("/data/fichtel/projektarbeit/auto_templates/auto_templates_{}.json".format(min_sample), "w")
    json.dump(dictio_prop_rank_templates, template_file)