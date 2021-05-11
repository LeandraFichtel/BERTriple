from transformers import BertForMaskedLM, BertModel, Trainer, TrainingArguments
import json
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import dill
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import shutil
import argparse
from torchsummary import summary


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
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-min_sample', help="set how many triple at least should exist of each property in wikidata_onetoken_missing")
    parser.add_argument('-sample', help="set how many triple should be used of each property (e.g. 10000)")
    parser.add_argument('-epoch', help="set how many epoches should be executed")
    parser.add_argument('-template', help="set which template should be used (LAMA or label)")
    parser.add_argument('-alone',action="store_true", default=False, help="set flag training data should be used which is only for one template")
    parser.add_argument('-string_token', help="set if obj and subj labels should consist of only one word (oneword) and are also in vocab file (onetoken)")
    parser.add_argument('-perc_prop', help="set how many props should be used for training (e.g. 100 for all props or 90-0 for first random_prop selection with 90% of the props)")
    
    args = parser.parse_args()
    print(args)
    epoch = int(args.epoch)
    template = args.template
    alone = args.alone
    if alone:
        alone = "alone"
    else:
        alone = ""
    sample = args.sample
    min_sample = args.min_sample
    if int(sample) > int(min_sample):
        raise("ERROR the sample size cannot be bigger than the min_sample size")
    string_token = args.string_token
    assert(string_token in ["onetoken", "oneword", ""])
    perc_prop = args.perc_prop
    if perc_prop != "100" and not (int(sample)==500 or int(sample)==100 or template=="LAMA"):
        exit("ERROR training on less than 100% of the props is only possible with sample=500 or sample=100 and template=label")
    
    #used LM
    lm_name = 'bert-base-cased'
    #pepare training dataset
    #read datasets from path
    train_queries, train_answers = read_dataset("/data/fichtel/projektarbeit/training_datasets/training_dataset_{}_{}_{}{}_{}_{}.json".format(perc_prop, string_token, template, alone, min_sample, sample))
    #use tokenizer to get encodings
    tokenizer = BertTokenizer.from_pretrained(lm_name)
    train_question_encodings = tokenizer(train_queries, truncation=True, padding='max_length', max_length=256)
    train_answer_encodings = tokenizer(train_answers, truncation=True, padding='max_length', max_length=256)["input_ids"]
    #get final datasets for training
    train_dataset = MaskedDataset(train_question_encodings, train_answer_encodings)
    
    model = BertModel.from_pretrained(lm_name)
    print(summary(model))
    exit()

    #prepare test dataset
    #read datasets from path
    #test_queries, test_answers = read_dataset("/data/fichtel/projektarbeit/test_dataset_{}{}_{}_{}.json".format(template, alone, min_sample, sample))
    #use tokenizer to get encodings
    #test_question_encodings = tokenizer(test_queries, truncation=True, padding='max_length', max_length=256)
    #test_answer_encodings = tokenizer(test_answers, truncation=True, padding='max_length', max_length=256)["input_ids"]
    #get final datasets for testing
    #test_dataset = MaskedDataset(test_question_encodings, test_answer_encodings)

    print("start training")
    lm_name_path = lm_name.replace("-", "_")
    model_path = "/data/fichtel/projektarbeit/{}_finetuned_{}_{}_{}_{}{}_{}_{}".format(lm_name_path, perc_prop, string_token, epoch, template, alone, min_sample, sample)
    if os.path.exists(model_path):
        print("remove dir of model")
        shutil.rmtree(model_path)
    os.mkdir(model_path)
    training_args = TrainingArguments(
    output_dir=model_path+'/results',          # output directory
    num_train_epochs=epoch,           # total number of training epochs
    per_device_train_batch_size=20,   # batch size per device during training
    per_device_eval_batch_size=64,    # batch size for evaluation
    warmup_steps=500,                 # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                # strength of weight decay
    logging_dir=model_path+'/logs',   # directory for storing logs
    logging_steps=10,
    save_total_limit=0
    )
    
    model = BertForMaskedLM.from_pretrained(lm_name)

    trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset          # training dataset
    )

    trainer.train()
    trainer.save_model(model_path)