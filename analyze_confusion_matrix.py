#import torch
import os
import numpy as np
import seaborn as sns

n_classes = 7
cm = np.array([
    [270,10,37,5,5,13,3],
    [45,220,30,37,28,23,2],
    [47,9,277,5,12,26,1],
    [30,23,14,282,11,19,2],
    [26,35,105,30,166,18,5],
    [46,21,68,43,16,197,0],
    [5,4,3,2,8,3,107]
])


# np.set_printoptions(suppress=True, precision=4)

import matplotlib.pyplot as plt

# Normalize the confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap='Blues', cbar=False)
plt.title('Confusion Matrix with Percentages')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



for c in range(n_classes):
    tp = cm[c,c]
    fp = sum(cm[:,c]) - cm[c,c]
    fn = sum(cm[c,:]) - cm[c,c]
    tn = sum(np.delete(sum(cm)-cm[c,:],c))

    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    specificity = tn/(tn+fp)
    f1_score = 2*((precision*recall)/(precision+recall))
    
    


    #print(f"for class {c}: acc {accuracy}, recall {recall},\
    #      precision {precision}, f1 {f1_score}")
    print("for class {}: recall {}, specificity {}\
          precision {}, f1 {}".format(c,round(recall,4), round(specificity,4), round(precision,4),round(f1_score,4)))

##    print("tp: ", tp)
##    print("fp: ", fp)
##    print("fn: ", fn)
##    print("tn: ", tn)