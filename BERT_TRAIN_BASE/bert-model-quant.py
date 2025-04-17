# %% [markdown]
# # IMPORTS

# %%
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import torch
from tqdm.auto import tqdm

# %%
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # DATA

# %%
model_path = "../model/bert_sentiment_model"
onnx_path = "../model/onnx/model.onnx"
openvino_model_path = "../model/model_openvino/model_openvino.xml"


# %%
# для проверки пайплайна
# dataset = load_dataset("imdb")
# train_df = pd.DataFrame(dataset['train']).sample(10000, random_state=42).reset_index(drop=True)
# test_df = pd.DataFrame(dataset['test']).sample(2000, random_state=42).reset_index(drop=True)
train_df = pd.read_csv('../datasets/train.csv', sep=';').reset_index(drop=True)
test_df = pd.read_csv('../datasets/test.csv', sep=';').reset_index(drop=True)

# %%
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

# %%
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

# %%
device_cpu = torch.device("cpu")

# %% [markdown]
# ### Экспортируем модель в ONNX формат

# %%
# Загружаем обученную модель
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
model = model.to(device_cpu)

# %%
# print(type(model))
# print(type(tokenizer))

# %%

dummy_input = (
    torch.randint(0, 100, (1, 512)),
    torch.ones((1, 512), dtype=torch.int64)               
)
torch.onnx.export(
    model, dummy_input, onnx_path, 
    input_names=["input_ids", "attention_mask"], output_names=["output"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=14
)

print(f"Модель сохранена в {onnx_path}")

# %% [markdown]
# ### Экспотируем в OpenVINO и выполняем квантизацию

# %%
import openvino as ov
from openvino import convert_model
import nncf
import onnx


# %%
# quantized_model_path = "../model/model_openvino_int8/model_openvino_int8.xml"  # Файл для сохранения квантизированной модели

print(onnx_path)

# %%
onnx_model = onnx.load(onnx_path)

# %%
# Проверка модели и вывод структуры
# onnx.checker.check_model(onnx_model)
# print(onnx.helper.printable_graph(onnx_model.graph))

# %%
model_ir = convert_model(onnx_path)
ov.serialize(model_ir, openvino_model_path)
print(f"Модель сохранена в {openvino_model_path}")

# %%
# Загружаем модель OpenVINO
core = ov.runtime.Core()
model_ov = core.read_model(openvino_model_path)
print("Модель загружена!")

# %%
# # import openvino.runtime as ov

# # core = ov.Core()
# model_ov = core.read_model(openvino_model_path)

# for input_node in model_ov.inputs:
#     print(f"Input name: {input_node.get_any_name()}, shape: {input_node.get_partial_shape()}")


# %%
# Используем уже загруженный tokenizer
calibration_size = 300
calibration_df = train_df.sample(calibration_size, random_state=42).reset_index(drop=True)

# %%
# Создаем DataLoader
calibration_loader = torch.utils.data.DataLoader(
    TextDataset(calibration_df, tokenizer, max_length=512), 
    batch_size=4, 
    shuffle=False
)

# %%
# Функция трансформации для калибровки
def transform_fn(data_item):
    input_ids = data_item["input_ids"].numpy() # to_numpy()
    attention_mask = data_item["attention_mask"].numpy()
    return {"input_ids": input_ids, "attention_mask": attention_mask}

# %%
# Создание калибровочного датасета для NNCF
calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)

# %%
# sample = next(iter(calibration_dataset.get_inference_data()))
# print(sample.keys())  # Должен содержать 'input'
# print(sample['input_ids'].shape)  # Должно быть (batch_size, 128)

# %%
# sample = next(iter(calibration_dataset))
# print(sample.keys())  # Должно содержать 'input'
# print(sample['input'].shape)  # Должно быть (batch_size, 128)

# %%
quantized_model = nncf.quantize(
    model_ov, 
    calibration_dataset, 
    model_type=nncf.ModelType.TRANSFORMER, 
    target_device=nncf.TargetDevice.CPU,
    fast_bias_correction=True,
    preset=nncf.quantization.QuantizationPreset.PERFORMANCE,
    advanced_parameters=nncf.quantization.advanced_parameters.AdvancedQuantizationParameters(
        batchwise_statistics=False
    )    
)

# %%
# ov.serialize(quantized_model, 
#              "../model/bert_quantized_model/quantized_model.xml", 
#              "../model/bert_quantized_model/quantized_model.bin")
# print("Квантизация завершена.")


# %%

# === 5. Компиляция и трансформация в реальные INT8-операции ===
compiled_model = core.compile_model(quantized_model, "CPU")

# === 6. Сохранение ИМЕННО компилированной модели (теперь это реально INT8) ===
ov.save_model(compiled_model, "./quantized_model")  # Сохраняет quantized_model.xml/bin

# ov.tensor_from_file
# 