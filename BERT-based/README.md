## Installation
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
python -m pip install --upgrade pip
python -m pip install transformers datasets evaluate wandb scikit-learn
python -m pip install --upgrade accelerate
```

## Running the code
this will automatically download the twitter dataset (not full) and fine-tune the bert-base-uncased model using cross-entropy loss.
```bash
python run.py
```
If you want to change some hyperparameters, please refer config.py file.

## Wandb link
[text](https://wandb.ai/hsunyu/epfl_ml_project2?nw=nwuserhsunyu)