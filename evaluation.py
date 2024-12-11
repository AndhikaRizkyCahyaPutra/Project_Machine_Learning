import numpy as np

class Evaluation:
    @staticmethod
    def calculate_confusion_matrix(true_labels, predicted_labels, unique_labels):
        matrix = {label: {l: 0 for l in unique_labels} for label in unique_labels}
        for true, pred in zip(true_labels, predicted_labels):
            matrix[true][pred] += 1
        return matrix

    @staticmethod
    def calculate_metrics(true_labels, predicted_labels):
        unique_labels = np.unique(true_labels)
        confusion_matrix = Evaluation.calculate_confusion_matrix(true_labels, predicted_labels, unique_labels)
        metrics = {}

        for label in unique_labels:
            true_positive = confusion_matrix[label][label]
            false_positive = sum(confusion_matrix[l][label] for l in unique_labels if l != label)
            false_negative = sum(confusion_matrix[label][l] for l in unique_labels if l != label)
            true_negative = sum(
                confusion_matrix[l1][l2]
                for l1 in unique_labels for l2 in unique_labels
                if l1 != label and l2 != label
            )

            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            accuracy = (true_positive + true_negative) / len(true_labels)

            metrics[label] = {
                "Precision": precision,
                "Recall": recall,
                "Accuracy": accuracy,
            }

        return metrics, confusion_matrix
