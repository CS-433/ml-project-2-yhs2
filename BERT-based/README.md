This code is primarily adapted from the [Huggingface GitHub repository](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) and the [supplementary material of our new paper](https://openreview.net/forum?id=oI5tZaWkF9). Please note that the code has not been officially released and the paper is still under review. Therefore, do not distribute the code or the OpenReview link.

## Installation
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
python -m pip install --upgrade pip
python -m pip install transformers datasets evaluate wandb scikit-learn
python -m pip install --upgrade accelerate
```

## Running the code for our best model
this will automatically download the twitter dataset and the model weights
```bash
python run.py
```
After running the code, the results will be saved in wandb. Additionally, you can find the submission file in the output folder at `./output/test_submission.txt`.
If you want to change some hyperparameters, please refer config.py file.

## Running the code for training the bert model with DIMP-Loss
This example is on our experimental dataset (full dataset will date a long time), you can change the dataset in the config.py file.

### First step, train the bert model on validation set
```bash
python run.py config_val.json
```
the result will automatically save the model in wandb. you should copy the model api to the config_train.json file. e.g. `hsunyu/epfl_ml_project2/twitter_1_only_valid_bert_base:v1`

### Second step, train the bert model with DIMP-Loss
To use the model you have trained from last step, update the `twomodelloss_wandb_model2` key in the `config_train.json` file. Alternatively, you can run the following command to use the provided validation model.
```bash
python run.py config_train.json
```
Note: this will not provide the submission file, because the testing is from our experimental dataset.

## Important parameters in config.py

## Citation
Please cite our paper if you use DIMP-Loss in your work:
```bibtex
@misc{kuo2024llmgenerated,
    title={Not All LLM-Generated Data Are Equal: Rethinking Data Weighting in Text Classification},
    author={Hsun-Yu Kuo and Yin-Hsiang Liao and Yu-Chieh Chao and Wei-Yun Ma and Pu-Jen Cheng},
    year={2024},
    eprint={2410.21526},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```