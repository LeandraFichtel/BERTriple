from transformers import pipeline, BertForMaskedLM
import json
lm_name = "bert-base-cased"

perc_prop = 100
string_token = "onetoken"
epoch = 3
min_sample = 500
sample = 100
template = "label"
alone = ""

def read_dataset(dataset_path):
    dataset_file = open(dataset_path, "r")
    for line in dataset_file:
        dictio_query_answer = json.loads(line)
        prop = dictio_query_answer["prop"]
        if prop == "P17":
            if dictio_query_answer["answer"] == "Germany":
                print(dictio_query_answer["query"])

read_dataset("/data/fichtel/projektarbeit/training_datasets/training_dataset_{}_{}_{}{}_{}_{}.json".format(perc_prop, string_token, template, alone, min_sample, sample))

query_1 = "Berlin lies in [MASK]."
query_2 = "Berlin country [MASK]."
query_3 = "Berlin lies in the country of [MASK]."

unmasker_normal = pipeline('fill-mask', model=lm_name, device=0, top_k=1)
print(query_1, "-->", unmasker_normal(query_1))
print(query_2, "-->", unmasker_normal(query_2))
print(query_3, "-->", unmasker_normal(query_3))


finetuned_lm_path = "/data/fichtel/projektarbeit/bert_base_cased_finetuned_{}_{}_{}_{}{}_{}_{}".format(perc_prop, string_token, epoch, template, alone, min_sample, sample)
unmasker_finetuned = pipeline('fill-mask', tokenizer= lm_name, model = BertForMaskedLM.from_pretrained(finetuned_lm_path), device=0, top_k=1)
print(query_2, "-->", unmasker_finetuned(query_2))