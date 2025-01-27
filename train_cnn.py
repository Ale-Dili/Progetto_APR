import tensorflow as tf # type: ignore
from tensorflow.keras import layers, models, regularizers # type: ignore
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import TensorBoard # type: ignore


def plot_history(history):
    os.makedirs('model/history', exist_ok=True)

    # Grafici di andamento del training
    plt.figure(figsize=(12, 5))

    # Grafico della loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Grafico della accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Salva i grafici
    plt.tight_layout()
    plt.savefig('model/history/training_history.png')
    print("Grafici salvati in 'model/history/training_history.png'.")

    # Mostra i grafici
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False):
    os.makedirs('model/history', exist_ok=True)

    # Calcola la confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Mostra la confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.tight_layout()

    # Salva la confusion matrix
    plt.savefig('model/history/confusion_matrix.png')
    print("Confusion matrix salvata in 'model/history/confusion_matrix.png'.")

    # Mostra la confusion matrix
    plt.show()
    
'''
def build_cnn(input_shape=(128, 128, 1), num_classes=8):
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),


        layers.Dense(num_classes, activation='softmax')
    ])
    return model
'''


def build_cnn(input_shape=(128, 128, 1), num_classes=7):
    model = models.Sequential([
        layers.Conv2D(8, (3,3), activation = 'relu', input_shape = input_shape),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(16, (3, 3), activation = 'relu'),

        layers.Dropout(0.2),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation = 'relu'),
        layers.Dropout(0.2),
        
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(512, activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(num_classes, activation = 'softmax')
    ])
    
    return model


#------MAIN--------------------

X_test = np.load("processed_data/X_test.npy")
X_train = np.load("processed_data/X_train.npy")
X_val = np.load("processed_data/X_val.npy")
y_test = np.load("processed_data/y_test.npy")
y_train = np.load("processed_data/y_train.npy")
y_val = np.load("processed_data/y_val.npy")


num_classes = len(np.unique(y_train))

'''
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
'''

#config tensorboard
log_dir = "logs/fit/" + tf.timestamp().numpy().astype(str)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
#print num classes
print(f'Number of classes: {num_classes}')
#print tensorboard log dir
print(f'Tensorboard log dir: {log_dir}')

model = build_cnn(input_shape=X_train.shape[1:], num_classes=num_classes)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              )



history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=32,callbacks=[tensorboard_callback])


test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}')
print('-----------------------------------------')

print("Saving the model...")
model.save('model/cnn_model.h5')
print("Model saved as 'cnn_model.h5'.")


print("Saving the history...")
plot_history(history)
classes = []
if num_classes == 8:
    classes = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
elif num_classes == 7:
    classes = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
elif num_classes == 3:
    classes = ['Low', 'Medium', 'High']
else:
    classes = [str(i) for i in range(num_classes)]
plot_confusion_matrix(y_test, np.argmax(model.predict(X_test), axis=1), classes=classes)
