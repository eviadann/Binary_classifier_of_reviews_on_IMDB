import yaml
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from src.main_model import TextClassificationModel
from src.data_processing import TextDataset
import numpy as np
import random
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(train_df, val_df, config_path="configs/config.yaml", seed_all:int=None):

    if seed_all:
        set_seed(seed_all)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)


    model = TextClassificationModel(**config["model"])

    train_dataset = TextDataset(train_df, model.tokenizer, max_length=model.max_length)
    val_dataset = TextDataset(val_df, model.tokenizer, max_length=model.max_length)

    training_args = TrainingArguments(**config["training"])
    #
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=model.tokenizer,
        compute_metrics=model.compute_metrics,
        callbacks=[EarlyStoppingCallback(**config["callbacks"]["es"])]
    )

    # Запуск обучения
    # trainer.train()
    return model

if __name__=="__main__":
    import pandas as pd
    from datasets import load_dataset
    dataset = load_dataset("imdb")
    train_df = pd.DataFrame(dataset['train']).sample(10000, random_state=42).reset_index(drop=True)
    test_df = pd.DataFrame(dataset['test']).sample(2000, random_state=42).reset_index(drop=True)
    train_model(train_df=train_df, val_df=test_df, config_path="configs/config.yaml")


