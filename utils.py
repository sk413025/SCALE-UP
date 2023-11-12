from sklearn import metrics
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np

def AUROC_Score(pred_in, pred_out, file):
    y_in = [1] * len(pred_in)
    y_out = [0] * len(pred_out)

    y = y_in + y_out

    pred = pred_in.tolist() + pred_out.tolist()
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    plt.plot(fpr, tpr, label=file)
    plt.savefig(file + ".png", bbox_inches="tight")
    auc_score = metrics.roc_auc_score(y, pred)
    print(auc_score)
    
    # Convert predictions to binary class labels
    pred_class = [1 if p >= 0.5 else 0 for p in pred]
    # Calculate F1 score
    f1 = f1_score(y, pred_class)
    print("F1 Score:", f1)
    return auc_score, f1
