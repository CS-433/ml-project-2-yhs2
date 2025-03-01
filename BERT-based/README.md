This code is adapted from the [Huggingface GitHub repository](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) and the [supplementary material of our new paper](https://openreview.net/forum?id=oI5tZaWkF9). Please note that the code and the paper are not officially released, as the paper is still under review. Do not distribute the code or the OpenReview link.

---

## Running the Code for Our Best Model

To automatically download the Twitter dataset and the model weights, run:

```bash
python run.py
```

- **Results:** After running, results will be saved in Weights & Biases (W&B).
- **Submission File:** A submission file will be available at `./output/test_submission.txt`.

To modify hyperparameters, refer to the `config.py` file.

---

## Training the BERT Model with DIMP-Loss

This example uses our experimental dataset (full dataset training may take longer). You can change the dataset in the `config.py` file.

### Step 1: Train the BERT Model on the Validation Set

Run the following command:

```bash
python run.py config_val.json
```

- **Results:** The trained model will be automatically saved in W&B.
- **Next Step:** Copy the model API to the `config_train.json` file. Example: `hsunyu/epfl_ml_project2/twitter_1_only_valid_bert_base:v1`.

### Step 2: Train the BERT Model with DIMP-Loss

To use the previously trained model, update the `twomodelloss_wandb_model2` key in `config_train.json`. Alternatively, use the provided validation model by running:

```bash
python run.py config_train.json
```

- **Note:** This will not generate a submission file because testing uses our experimental dataset.

---

## Important Parameters in `config.py`

- **`model_name_or_path`**:  
  Specifies the pretrained model path or identifier from Hugging Face's model hub (e.g., `bert-base-uncased`, `vinai/bertweet-base`, `hsunyu/epfl_ml_project2/twitter_full_bertweet_large:v1`).
  If using a W&B model, set `use_wandb_model` to `True` and specify the model name in the `wandb_model` key.

- **`problem_type`**:  
  Defines the task type. Examples include:
  - `"single_label_classification"`: For text classification with cross-entropy loss (CE-Loss).
  - `"single_label_classification_myloss_v2"`: For the DIMP-Loss approach.

- **`wandb_dataset`**:  
  Specifies the W&B dataset artifact name for training and evaluation. Examples:
  - `hsunyu/epfl_ml_project2/twitter_full_datasets:v3`: Full dataset.
  - `hsunyu/epfl_ml_project2/twitter_dataset_1:v0`: Experimental dataset.

- **`use_wandb_model`**:  
  Boolean indicating whether to load a pretrained model from a W&B artifact. Useful for reproducibility.

- **`twomodelloss_wandb_model2`**:  
  Refers to the W&B artifact for the validation model used in DIMP-Loss training. Example: `hsunyu/epfl_ml_project2/twitter_1_only_valid_bert_base:v1`.

- **`per_device_train_batch_size`**:  
  Defines the batch size per device during training. Default: `128`.

- **`num_train_epochs`**:  
  Specifies the total number of training epochs. Default: `3.0`.

### Contrastive Learning Parameters
We also tried self-supervise contrastive learning ([SimCSE](https://arxiv.org/abs/2104.08821)) in this project. However, it does not impact a lot in this case thus we did not discuss in main part of report. following parameters are related to contrastive learning:

- **`contrastive_learning`**:  
A boolean indicating whether to use contrastive learning during training. This approach improves representation learning by encouraging the model to distinguish between similar and dissimilar samples.

- **`contrastive_learning_weight`**:  
Defines the weight assigned to the contrastive loss component during training. Higher values emphasize the contrastive loss more.

- **`temperature`**:
Specifies the temperature parameter for scaling logits in the contrastive loss computation. Lower values make the model more confident in its predictions.


You can adjust these parameters in the `config.json` file to fine-tune the model behavior and experiment settings.

---

### Hyperparameter Tuning and Experiment Results in Weights & Biases
Most of the experiment results for BERT-based approaches use default hyperparameters from Hugging Face. However, we also conducted hyperparameter tuning in this project. You can explore the experiment results in Weights & Biases using the following links:
* [Dataset Size](https://wandb.ai/hsunyu/epfl_ml_project2/sweeps/24iegrm8?nw=nwuserhsunyu)
* [Contrastive Weight](https://wandb.ai/hsunyu/epfl_ml_project2/sweeps/7eu8aso3?nw=nwuserhsunyu)

## Citation

If you use DIMP-Loss in your work, please cite our paper:

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

