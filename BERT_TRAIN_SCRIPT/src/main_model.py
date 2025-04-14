import os
import time
from datetime import datetime

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import numpy as np
from tqdm.auto import tqdm


class TextClassificationModel:
    def __init__(self, model_name, num_labels, tokenizer=None, max_length=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.model_name = model_name
        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)
        self.model = self.model.to(self.device)

    def generate_classification_report(self, df, text_column="text", label_column="label", batch_size=32,
                                       threshold_dict=None, save_to_file=False):
        """
        Генерирует отчеты классификации модели для различных порогов, указывая метрики для каждого порога и,
        при необходимости, сохраняет результаты в текстовые файлы.
        """

        if threshold_dict is None:
            threshold_dict = {"default": 0.5}

        labels = df[label_column].tolist()
        all_probabilities = []

        for i in tqdm(range(0, len(df), batch_size)):
            batch_texts = df[text_column].iloc[i:i + batch_size].tolist()

            tokens = self.tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            with torch.no_grad():
                outputs = self.model(**tokens)
                probabilities = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
                all_probabilities.extend(probabilities)

            del tokens
            torch.cuda.empty_cache()

        model_short_name = self.model_name.split("/")[-1]
        self._generate_reports_from_probabilities(labels, all_probabilities, threshold_dict, save_to_file,
                                                  model_short_name)

    @staticmethod
    def _generate_reports_from_probabilities(labels, all_probabilities, threshold_dict, save_to_file=False,
                                             model_name="model"):
        """
        Генерирует и выводит отчеты классификации для различных порогов, а также сохраняет их в текстовые файлы при необходимости.

        """
        reports = {}
        for name, threshold in threshold_dict.items():
            predictions = [1 if prob >= threshold else 0 for prob in all_probabilities]
            report = classification_report(labels, predictions, target_names=["Class 0", "Class 1"])
            reports[name] = report
            print(f"Classification Report for threshold '{name}' ({threshold}):\n{report}")

            if save_to_file:
                timestamp = int(datetime.now().timestamp())
                filename = f"{model_name}_{threshold}_thr_{timestamp}.txt"
                filepath = os.path.join("./reports", filename)

                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, "w") as f:
                    f.write(f"Classification Report for threshold '{name}' ({threshold}):\n")
                    f.write(report)

    def predict_text(self, texts):
        """
        Вычисляет вероятности классов для заданного текста или списка текстов.
        """
        # if device:
        #     self.device = torch.device(device) if isinstance(device, str) else device
        #     self.model.to(self.device)

        if isinstance(texts, str):
            texts = [texts]

        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.model(**tokens)
            probabilities = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()

        return probabilities

    def evaluate_prediction_speed(self, texts, device="cpu", max_texts=100):
        # Установка устройства вручную
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        texts = texts[:max_texts]

        start_time = time.time()

        with torch.no_grad():
            for text in texts:
                tokens = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)

                _ = self.model(**tokens)

        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_prediction = total_time / len(texts)
        print(f"Общее время предсказания для {len(texts)} текстов: {total_time:.4f} секунд")
        print(f"Среднее время на одно предсказание: {avg_time_per_prediction:.4f} секунд")

    def evaluate_batch_prediction_speed(self, texts, device="cpu", max_texts=100):

        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        texts = texts[:max_texts]

        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        start_time = time.time()

        with torch.no_grad():
            _ = self.model(**tokens)

        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_prediction = total_time / len(texts)
        print(f"Общее время предсказания для {len(texts)} текстов: {total_time:.4f} секунд")
        print(f"Среднее время на одно предсказание: {avg_time_per_prediction:.4f} секунд")

    @staticmethod
    def compute_metrics(eval_pred, average="weighted"):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        f1 = f1_score(labels, predictions, average=average)
        # accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average=average)
        recall = recall_score(labels, predictions, average=average)

        return {"f1": f1,
                "precision": precision,
                "recall": recall,
                }



    def find_optimal_threshold(self,
                               df,
                               text_column="text",
                               label_column="label",
                               batch_size=32,
                               metric="f1",
                               thresholds=None,
                               average="weighted",
                               show_plot=False):
        """
        Подбирает оптимальный порог для бинарной классификации на основе указанной метрики.
        """
        if thresholds is None:
            thresholds = np.arange(0.35, 0.75, 0.05)

        labels = df[label_column].tolist()
        all_probabilities = []

        for i in tqdm(range(0, len(df), batch_size)):
            batch_texts = df[text_column].iloc[i:i + batch_size].tolist()
            probabilities = self.predict_text(batch_texts)
            all_probabilities.extend(probabilities)

        best_threshold = self._find_best_threshold(
            labels, all_probabilities, metric=metric, thresholds=thresholds, average=average, show_plot=show_plot
        )

        return best_threshold

    @staticmethod
    def _find_best_threshold(labels, probabilities, metric="f1", thresholds=None,
                             show_plot=False, average="weighted", ):
        """
        Подбирает оптимальный порог для бинарной классификации на основе указанной метрики.
        При необходимости отображает графики подбора порога.
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.05)

        best_threshold = 0.5
        best_score = 0

        plot_thresholds = []
        precisions = []
        recalls = []
        f1_scores = []

        for threshold in thresholds:
            predictions = [1 if prob >= threshold else 0 for prob in probabilities]

            precision = precision_score(labels, predictions, average=average, zero_division=0)
            recall = recall_score(labels, predictions, average=average, zero_division=0)
            f1 = f1_score(labels, predictions, average=average, zero_division=0)

            if show_plot:
                plot_thresholds.append(threshold)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)

            score = {"f1": f1, "precision": precision, "recall": recall}.get(metric)
            if score is None:
                raise ValueError("Unsupported metric. Choose 'f1', 'precision', or 'recall'.")

            if score > best_score:
                best_score = score
                best_threshold = round(threshold, 3)

        print(f"Best threshold: {best_threshold} with {metric} score: {best_score}")

        if show_plot:
            import matplotlib.pyplot as plt
            plt.style.use('ggplot')
            plt.rcParams["font.family"] = "Times New Roman"

            scores_table = pd.DataFrame({
                'f1': f1_scores,
                'precision': precisions,
                'recall': recalls,
                'probability': plot_thresholds
            }).sort_values(by=metric, ascending=False).round(3)

            figure = plt.figure(figsize=(9, 8))

            plt1 = figure.add_subplot(111)
            plt1.plot(plot_thresholds, precisions, label='Precision', linewidth=2)
            plt1.plot(plot_thresholds, recalls, label='Recall', linewidth=2)
            plt1.plot(plot_thresholds, f1_scores, label='F1', linewidth=2)
            plt1.plot((best_threshold, best_threshold), (0, 1), 'g--', label='Best Threshold')
            plt1.set_ylabel('Scores')
            plt1.set_xlabel('Probability threshold')
            plt1.set_title('Probabilities threshold calibration')
            plt1.legend(loc='upper right')

            # Отображение таблицы ниже графика
            plt_table = plt1.table(
                cellText=scores_table.values,
                colLabels=scores_table.columns,
                colLoc='center',
                cellLoc='center',
                loc='bottom',
                bbox=[0, -0.4, 1, 0.3]  # Увеличиваем отступ для таблицы
            )

            # Настройка полей, чтобы избежать наложения
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)  # Увеличиваем отступ снизу
            plt.tight_layout()
            plt.show()

        return best_threshold
