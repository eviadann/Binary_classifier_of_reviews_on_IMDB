# %% [markdown]
# # IMPORTS

# %%
# !pip install transformers datasets scikit-learn pandas openvino onnx nncf
# !pip install accelerate

# %%
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
# для проверки пайплайна
dataset = load_dataset("imdb")
train_df = pd.DataFrame(dataset['train']).sample(5000, random_state=42).reset_index(drop=True)
test_df = pd.DataFrame(dataset['test']).sample(1000, random_state=42).reset_index(drop=True)

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
def generate_classification_report(model, tokenizer, df, text_column="text", label_column="label", max_length=512, batch_size=32):
    predictions = []
    labels = df[label_column].tolist()
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch_texts = df[text_column].iloc[i:i + batch_size].tolist()
        
        tokens = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        tokens = {k: v.to(model.device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = model(**tokens)
            batch_predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            predictions.extend(batch_predictions)
        del tokens
        torch.cuda.empty_cache()
    
    report = classification_report(labels, predictions, target_names=["Class 0", "Class 1"])
    
    return report

# %% [markdown]
# ===============================================================================================

# %%
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average="weighted")
    # accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    
    return {"f1": f1,
            "precision": precision,
            "recall": recall,
           }

# %%
device_cpu = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# %%
# Предобученная модель
model_name = "cointegrated/LaBSE-en-ru"
# model_name = "distilbert/distilbert-base-multilingual-cased"  # https://huggingface.co/distilbert/distilbert-base-multilingual-cased
# model_name = "cointegrated/rubert-tiny"  # https://huggingface.co/cointegrated/rubert-tiny

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model = model.to(device)

# %%
# print(generate_classification_report(model, tokenizer, test_df, text_column="text", label_column="label"))

# %%
train_dataset = TextDataset(train_df, tokenizer, max_length=512)
test_dataset = TextDataset(test_df, tokenizer, max_length=512)
# train_dataset = prepare_dataset(train_df, tokenizer, text_column="text", label_column="label", max_length=512)
# test_dataset = prepare_dataset(test_df, tokenizer, text_column="text", label_column="label", max_length=512)

# %%
training_args = TrainingArguments(
    output_dir='./results',                      # Директория для сохранения результатов
    eval_strategy="epoch",                       # Оценка модели на каждой эпохе
    learning_rate=2e-5,                          # Скорость обучения
    per_device_train_batch_size=32,              # Размер батча для обучения
    per_device_eval_batch_size=32,               # Размер батча для валидации
    num_train_epochs=10,                         # Количество эпох
    weight_decay=0.00001,                        # Коэффициент регуляризации
    logging_dir='./logs',                        # Директория для логов (TensorBoard)
    logging_steps=50,                            # Логирование каждые 500 шагов
    load_best_model_at_end=True,                 # Загрузка лучшей модели по окончанию обучения
    save_total_limit=2,                          # Сохраняем только 2 лучшие модели
    metric_for_best_model="f1",                  # Ключевая метрика для выбора лучшей модели
    greater_is_better=True,                      # Лучшая модель — та, где метрика больше
    save_strategy="epoch",                       # Сохраняем модель на каждой эпохе
    report_to="tensorboard",                     # Используем TensorBoard для логирования
    optim="adamw_torch",                         # Явно указываем AdamW как оптимизатор
    warmup_steps=100,
)

# %%
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,                                   # Тренировочный датасет
#     eval_dataset=test_dataset,                                     # Валидационный датасет
#     tokenizer=tokenizer,                                           # Токенизатор
#     compute_metrics=compute_metrics,                               # Функция вычисления метрик
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # Ранняя остановка (2 эпохи)
#     # data_collator=data_collator
# )

# # Запуск тренировки
# trainer.train()

# %%
model_path = "../model/bert_sentiment_model"

# Сохранение модели
# trainer.save_model(model_path)
# # Сохранение токенизатора
# tokenizer.save_pretrained(model_path)

print(f"Модель сохранена в: {model_path}")

# %% [markdown]
# ### Экспортируем модель в ONNX формат

# %%
print('Загружаем обученную модель')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
model = model.to(device_cpu)

# %%
print(type(model))
print(type(tokenizer))

# %%
onnx_path = "../model/onnx/model.onnx"

print('Экспортируем модель в ONNX')
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
quantized_model_path = "../model/model_openvino_int8/model_openvino_int8.xml"  # Файл для сохранения квантизированной модели
onnx_path = "../model/onnx/model.onnx"
openvino_model_path = "../model/model_openvino/model_openvino.xml"

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
calibration_size = 500
calibration_df = train_df.sample(calibration_size, random_state=42).reset_index(drop=True)

# %%
# Создаем DataLoader
calibration_loader = torch.utils.data.DataLoader(
    TextDataset(calibration_df, tokenizer, max_length=512), 
    batch_size=32, 
    shuffle=False
)

# %%
# Функция трансформации для калибровки
def transform_fn(data_item):
    input_ids = data_item["input_ids"].numpy()
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
    target_device=nncf.TargetDevice.CPU
)

# %%
ov.serialize(quantized_model, 
             "../model/quantized_model/quantized_model.xml", 
             "../model/quantized_model/quantized_model.bin")
print("Квантизация завершена.")


