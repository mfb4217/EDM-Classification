"""
Evaluation of status classification model
"""
import torch
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, roc_auc_score

from config import Config
from model import create_model
from preprocessing import DataPreprocessor


class Evaluator:
    """Model evaluator"""
    
    def __init__(self, config: Config, model, preprocessor=None, variable_length=False):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        
        if preprocessor is None:
            preprocessor = DataPreprocessor(config)
        self.X_test, self.y_test = preprocessor.preprocess_test(config.test_path, variable_length=variable_length)
        self.variable_length = variable_length
        
        if isinstance(self.X_test, list):
            print(f"Test data: {len(self.X_test)} series (variable length)")
        else:
            print(f"Test data: {len(self.X_test)} chunks")
    
    def evaluate(self):
        """Evaluate on all test chunks"""
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        all_predictions = []
        all_probabilities = []
        all_ground_truth = []
        
        self.model.eval()
        with torch.no_grad():
            for x, y in tqdm(zip(self.X_test, self.y_test), total=len(self.X_test)):
                # Convert to tensor
                x_tensor = torch.FloatTensor(x).transpose(0, 1).unsqueeze(0).to(self.device)  # (1, num_channels, length)
                
                # Predict
                logits = self.model(x_tensor)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(logits, dim=1)
                
                all_predictions.append(pred.cpu().numpy()[0])
                all_probabilities.append(probs.cpu().numpy()[0])
                all_ground_truth.append(y)
        
        # Convert to numpy
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        y_true = np.array(all_ground_truth)
        
        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        
        class_names = self.config.get_class_names()
        print(f"\nPer-class metrics:")
        print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 50)
        for i, name in enumerate(class_names):
            print(f"{name:<10} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10.0f}")
        
        print(f"\nConfusion Matrix:")
        print(f"{'':>10}", end="")
        for name in class_names:
            print(f"{name:>10}", end="")
        print()
        for i, name in enumerate(class_names):
            print(f"{name:>10}", end="")
            for j in range(len(class_names)):
                print(f"{cm[i, j]:>10}", end="")
            print()
        
        # ROC AUC (one-vs-rest for multiclass)
        try:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            print(f"\nROC AUC (macro, one-vs-rest): {roc_auc:.4f}")
        except Exception as e:
            print(f"\nROC AUC could not be calculated: {e}")
            roc_auc = 0.0
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        return {
            'accuracy': float(accuracy),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist()
        }


def evaluate_model(config_path: str, model_dir: str, experiment_name: str):
    """Evaluate a trained model"""
    config = Config.load_config(config_path)
    
    # Create preprocessor and load validation scaler
    preprocessor = DataPreprocessor(config)
    preprocessor.preprocess_train(config.train_path)  # Fit scaler
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(model_dir, f"{experiment_name}_best_model.pth")
    model = create_model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Evaluate
    evaluator = Evaluator(config, model, preprocessor)
    results = evaluator.evaluate()
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python evaluate.py <config_path> <model_dir> <experiment_name>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    model_dir = sys.argv[2]
    experiment_name = sys.argv[3]
    
    evaluate_model(config_path, model_dir, experiment_name)

