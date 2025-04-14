import torch
from datasets import Dataset

def prepare_dataset(df, tokenizer, text_column="text", label_column="label", max_length=512):
    dataset = Dataset.from_pandas(df[[text_column, label_column]])

    def tokenize_function(example):
        return tokenizer(
            example[text_column],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns([text_column])
    tokenized_dataset.set_format("torch")
    return tokenized_dataset

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=512, text_col="text", label_col="label"):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_col = text_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, self.text_col]
        label = self.df.loc[idx, self.label_col]

        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze() for key, val in tokens.items()}
        item[self.label_col] = torch.tensor(label, dtype=torch.long)
        return item
