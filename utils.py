import numpy as np
from sklearn.metrics import roc_auc_score
def ROC(args, y_test,y_pred):
    auc=roc_auc_score(y_test,y_pred)
    print('auroc', auc)
    return auc
