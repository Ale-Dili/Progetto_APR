#import torch
import os
import numpy as np
import seaborn as sns

num_classes = 7
excecution = "decision_tree"
path_to_cm = "logs/"+excecution+"/confusion_matrix.npy"

classes = []
if num_classes == 8:
    classes = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
elif num_classes == 7:
    classes = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
elif num_classes == 3:
    classes = ['Low', 'Medium', 'High']
else:
    classes = [str(i) for i in range(num_classes)]

# Load the confusion matrix as npy file
cm = np.load(path_to_cm)


# np.set_printoptions(suppress=True, precision=4)

import matplotlib.pyplot as plt

# Normalize the confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap='Blues', cbar=False, xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix with Percentages')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



for c in range(num_classes):
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