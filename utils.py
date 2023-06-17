import numpy as np
from sklearn.metrics import roc_auc_score
def ROC(args, y_test,y_pred):
    auc=roc_auc_score(y_test,y_pred)
    print('auroc', auc)
    return auc
class log():
    def __init__(self) -> None:
        self.roc_auc_max = 0
        self.f1_max = 0
    def print_result(self, y_test, y_pred, model, i, args):
        y_test = np.nan_to_num(y_test)
        y_pred = np.nan_to_num(y_pred)

        auc=roc_auc_score(y_test,y_pred)
      
        if not os.path.exists("./{}{}modelroc".format(args.model, args.name,i, auc)):
            os.makedirs("./{}{}modelroc/".format(args.model,args.name, i, auc))
  
        if self.roc_auc_max < auc:
            self.roc_auc_max = auc
       
          
        print('auroc:{:.4f}'.format(auc))
        
