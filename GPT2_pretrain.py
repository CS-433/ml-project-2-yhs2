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
from scipy import stats
import seaborn as sns

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
    wandb.init(project="gpt2-tuning-test")

    X_train = []
    with open("./Data/twitter-datasets/train_pos_full.txt", "r", encoding='utf-8') as f:
        for line in f:
            X_train.append(line)
    with open("./Data/twitter-datasets/train_neg_full.txt", "r", encoding='utf-8') as f:
        for line in f:
            X_train.append(line)

    np.random.seed(42)
    np.random.shuffle(X_train)

    X_eval = X_train[:int(len(X_train)*0.05)]
    X_train = X_train[int(len(X_train)*0.05):]

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_encodings = tokenizer(X_train, truncation=True, padding=True)
    eval_encodings = tokenizer(X_eval, truncation=True, padding=True)

    train_set = GPT2Dataset(train_encodings)
    eval_set = GPT2Dataset(eval_encodings)

    configuration = GPT2Config.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', config=configuration)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(output_dir='gpt2-tuning',
                                    evaluation_strategy="steps",
                                    eval_steps=5000,                                  
                                    num_train_epochs=10,
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

    model.save_pretrained("gpt2-pretrain-test")

def compute_probs(model, tokenizer, sentences, device):
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
    return probs

def calculate_perplexity(text, model, tokenizer, device):
    encodings = tokenizer(text, return_tensors="pt")
    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)
    return ppl.item()

def plot_perplexities(perplexities_train, perplexities_test, title, filename):
    sns.set_palette("Set2")
    bins = np.arange(0, 300, 6)

    plt.hist(perplexities_train, bins=bins, alpha=0.5, label="Train", density=True, color=sns.color_palette()[0])
    plt.hist(perplexities_test, bins=bins, alpha=0.5, label="Test", density=True, color=sns.color_palette()[1])

    plt.xlabel("Perplexity")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.show()

def calculate_perplexities_for_train_test_data():
    X_test = []
    with open("./Data/twitter-datasets/test_data.txt", "r", encoding='utf-8') as f:
        for line in f:
            line = line.split(",", 1)[1]
            X_test.append(line)

    X_train = []
    with open("./Data/twitter-datasets/train_pos_full.txt", "r", encoding='utf-8') as f:
        for line in f:
            X_train.append(line)
    with open("./Data/twitter-datasets/train_neg_full.txt", "r", encoding='utf-8') as f:
        for line in f:
            X_train.append(line)

    np.random.seed(42)
    np.random.shuffle(X_train)

    X_train = X_train[:10000]

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2-tuning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    model.to(device)

    perplexities = []
    for sentence in tqdm.tqdm(X_test):
        perplexity = calculate_perplexity(sentence, model, tokenizer, device)
        perplexities.append(perplexity)
    np.save("perplexities_test.npy", perplexities)
    
    perplexities = []
    for sentence in tqdm.tqdm(X_train):
        perplexity = calculate_perplexity(sentence, model, tokenizer, device)
        perplexities.append(perplexity)
    
    np.save("perplexities_train.npy", perplexities)

def calculate_statistics_of_perplexities():
    perplexities_train = np.load("perplexities_train.npy")
    perplexities_test = np.load("perplexities_test.npy")
    print(f"Mean perplexity of train data: {np.mean(perplexities_train)}")
    print(f"Mean perplexity of test data: {np.mean(perplexities_test)}")
    print(f"Standard deviation of perplexity of train data: {np.std(perplexities_train)}")
    print(f"Standard deviation of perplexity of test data: {np.std(perplexities_test)}")

if __name__ == "__main__":
    plot_perplexities(np.load("perplexities_train.npy"), np.load("perplexities_test.npy"), "Perplexity distribution of train and test data", "perplexities_train_test.png")
    calculate_statistics_of_perplexities()
    

    