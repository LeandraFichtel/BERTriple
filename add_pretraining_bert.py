from transformers import BertForMaskedLM, Trainer, TrainingArguments, BertConfig
import json
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import dill
import os
import shutil
import argparse


#functions
def read_dataset(dataset_path):
    queries = []
    answers = []
    dataset_file = open(dataset_path, "r")
    for line in dataset_file:
        dictio_query_answer = json.loads(line)
        queries.append(dictio_query_answer["query"])
        answers.append(dictio_query_answer["query"].replace("[MASK]", dictio_query_answer["answer"]))
    return queries, answers

class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-min_sample', help="set how many triple at least should exist of each property in wikidata_onetoken_missing")
    parser.add_argument('-sample', help="set how many triple should be used of each property (e.g. 10000)")
    parser.add_argument('-epoch', help="set how many epoches should be executed")
    parser.add_argument('-template', help="set which template should be used (LAMA or label)")
    parser.add_argument('-alone',action="store_true", default=False, help="set flag training data should be used which is only for one template")
    args = parser.parse_args()
    print(args)
    epoch_number = int(args.epoch)
    template = args.template
    alone = args.alone
    if alone:
        alone = "alone"
    else:
        alone = ""
    sample = args.sample
    min_sample = args.min_sample
    if int(sample) > int(min_sample):
        exit("ERROR: the sample size cannot be bigger than the min_sample size")
    #used LM
    lm_name = 'bert-base-cased'
    lm_name_path = lm_name.replace("-", "_")
    #pepare training dataset
    #read datasets from path
    train_queries, train_answers = read_dataset("/data/fichtel/projektarbeit/training_dataset_{}{}_{}_{}.json".format(template, alone, min_sample, sample))
    
    #use tokenizer to get encodings
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    train_question_encodings = tokenizer(train_queries, truncation=True, padding='max_length', max_length=256)
    train_answer_encodings = tokenizer(train_answers, truncation=True, padding='max_length', max_length=256)["input_ids"]

    #get final datasets for training
    train_dataset = MaskedDataset(train_question_encodings, train_answer_encodings)
    
    #additional pretraining
    print("start training")
    model = BertForMaskedLM.from_pretrained(lm_name)
    if torch.cuda.is_available():
        print("using gpu")
        device = torch.device('cuda')
    else:
        print("using cpu")
        device = torch.device('cpu')
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)
    for epoch in range(epoch_number):
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()
        print("epoch {} ready".format(epoch))

    model_path = "/data/fichtel/projektarbeit/{}_finetuned_{}_{}{}_{}_{}".format(lm_name_path, epoch_number, template, alone, min_sample, sample)
    model.save_pretrained(model_path)
    #print(model.eval())