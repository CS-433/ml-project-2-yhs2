from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, GPT2ForSequenceClassification
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import TrainingArguments, Trainer, AutoTokenizer
import torch
from torch.utils.data import Dataset
import tqdm
import wandb
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import os
import math

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


class GPT2Dataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


def pretraining():
    wandb.login()
    wandb.init(project="gpt2-tuning")

    X_train = []
    with open("./Data/twitter-datasets/train_pos_full.txt", "r", encoding='utf-8') as f:
        for line in f:
            X_train.append(line)
    with open("./Data/twitter-datasets/train_neg_full.txt", "r", encoding='utf-8') as f:
        for line in f:
            X_train.append(line)

    np.random.seed(42)
    np.random.shuffle(X_train)

    X_eval = X_train[:int(len(X_train)*0.01)]
    X_train = X_train[int(len(X_train)*0.01):]

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_encodings = tokenizer(X_train, truncation=True, padding=True)
    eval_encodings = tokenizer(X_eval, truncation=True, padding=True)

    train_set = GPT2Dataset(train_encodings)
    eval_set = GPT2Dataset(eval_encodings)

    configuration = GPT2Config.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', config=configuration)
    training_args = TrainingArguments(output_dir='gpt2-tuning',
                                    evaluation_strategy="steps",
                                    eval_steps=10000,                                  
                                    num_train_epochs=1,
                                    per_device_train_batch_size=8,
                                    per_device_eval_batch_size=8,
                                    learning_rate=2.5e-4,
                                    lr_scheduler_type='cosine',
                                    warmup_ratio=0.05,
                                    adam_beta1=0.9,
                                    adam_beta2=0.999,                                  
                                    weight_decay=0.01,                                  
                                    logging_strategy="steps",
                                    logging_steps = 500,
                                    save_steps=5000,
                                    save_total_limit=10,                                  
                                    report_to='wandb',                                  
                                    ) 

    trainer = Trainer(model=model, args=training_args, train_dataset=train_set, eval_dataset=eval_set, tokenizer=tokenizer)
    trainer.train()

    model.save_pretrained("gpt2-tuning")

def compute_probs(model, tokenizer, sentences):
    # Calculate the token-wise probabilities of the sentences
    probs = []
    for sentence in tqdm.tqdm(sentences):
        input_ids = tokenizer.encode(sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        # Calculate the probability for each token
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        log_probs = log_probs[0, range(len(input_ids[0])), input_ids[0]]
        print(log_probs)

        # Calculate the probability of the sentence
        sentence_prob = torch.sum(log_probs, dim=-1)
        probs.append(sentence_prob)
        print(sentence_prob)
    return probs

def calculate_perplexity(sentence, model, tokenizer, device):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()  # Cross-entropy loss
    return math.exp(loss)  # Convert to perplexity

def plot_perplexities(perplexities_train, perplexities_test, title, filename):
    plt.hist(perplexities_train, bins=np.arange(0, 500, 10), alpha=0.5, label="Train", density=True)
    plt.hist(perplexities_test, bins=np.arange(0, 500, 10), alpha=0.5, label="Test", density=True)
    plt.xlabel("Perplexity")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.savefig(filename)

if __name__ == "__main__":
    '''
    X_test = []
    with open("./Data/twitter-datasets/test_data.txt", "r", encoding='utf-8') as f:
        for line in f:
            X_test.append(line)
    '''
    '''
    X_train = []
    with open("./Data/twitter-datasets/train_pos_full.txt", "r", encoding='utf-8') as f:
        for line in f:
            X_train.append(line)
    with open("./Data/twitter-datasets/train_neg_full.txt", "r", encoding='utf-8') as f:
        for line in f:
            X_train.append(line)

    np.random.seed(42)
    np.random.shuffle(X_train)

    X_train = X_train[:100000]

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2-tuning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    model.to(device)
    
    perplexities = []
    for sentence in tqdm.tqdm(X_train):
        perplexity = calculate_perplexity(sentence, model, tokenizer, device)
        perplexities.append(perplexity)
    
    np.save("perplexities_train.npy", perplexities)
    '''
    plot_perplexities(np.load("perplexities_train.npy"), np.load("perplexities_test.npy"), "Perplexity distribution of train and test data", "perplexities_train_test.png")
    
    