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
def read_dataset(dataset_path, dictio_prop_template, template):
    queries = []
    answers = []
    dataset_file = json.load(open(dataset_path, "r"))
    for datapoint in dataset_file:
        subj_label = datapoint["subj"]
        prop =  datapoint["prop"]
        obj_label = datapoint["obj"]
        query = dictio_prop_template[prop][template]
        queries.append(query.replace("[S]", subj_label).replace("[O]", "[MASK]"))
        answers.append(query.replace("[S]", subj_label).replace("[O]", obj_label))
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
    dictio_prop_template = json.load(open("/data/fichtel/BERTriple/templates.json", "r"))

    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_file', help="filename of the training dataset")
    parser.add_argument('-sample', help="set how many triple should be used of each property (e.g. 10000)")
    parser.add_argument('-epoch', help="set how many epoches should be executed")
    parser.add_argument('-template', help="set which template should be used (LAMA or label)")
    parser.add_argument('-perc_prop', help="set how many props should be used for training (e.g. 100 for all props or 90-0 for first random_prop selection with 90% of the props)")
    
    args = parser.parse_args()
    print(args)
    train_file = args.train_file
    epoch = int(args.epoch)
    template = args.template
    sample = args.sample
    perc_prop = args.perc_prop
    if perc_prop != "100" and not (int(sample)==500 or int(sample)==100 or template=="LAMA"):
        exit("ERROR training on less than 100% of the props is only possible with sample=500 or sample=100 and template=label")
    
    #used LM
    lm_name = 'bert-base-cased'
    #pepare training dataset
    #read datasets from path
    train_queries, train_answers = read_dataset("/data/fichtel/BERTriple/training_datasets/{}".format(train_file), dictio_prop_template, template)
    
    #use tokenizer to get encodings
    tokenizer = BertTokenizer.from_pretrained(lm_name)
    train_question_encodings = tokenizer(train_queries, truncation=True, padding='max_length', max_length=256)
    train_answer_encodings = tokenizer(train_answers, truncation=True, padding='max_length', max_length=256)["input_ids"]
    #get final datasets for training
    train_dataset = MaskedDataset(train_question_encodings, train_answer_encodings)
    
    model = BertModel.from_pretrained(lm_name)

    print("start training")
    lm_name_path = lm_name.replace("-", "_")
    model_path = "/data/fichtel/BERTriple/models/{}_finetuned_{}_{}_{}".format(lm_name_path, train_file.split(".")[0], epoch, template)
    
    if os.path.exists(model_path):
        print("remove dir of model")
        shutil.rmtree(model_path)
    os.mkdir(model_path)
    training_args = TrainingArguments(
    output_dir=model_path+'/results',          # output directory
    num_train_epochs=epoch,           # total number of training epochs
    per_device_train_batch_size=16,   # batch size per device during training
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