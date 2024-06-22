import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Flatten the images for MLP
train_images = train_images.reshape((50000, 32 * 32 * 3))
test_images = test_images.reshape((10000, 32 * 32 * 3))

# Define the model architecture
def create_model():
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(32 * 32 * 3,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
    )
    return model

# Evaluate model performance
def evaluate_model(model, test_images, test_labels):
    test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(test_images, test_labels, verbose=2)
    f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    print(f'Test accuracy: {test_acc}')
    print(f'Test precision: {test_precision}')
    print(f'Test recall: {test_recall}')
    print(f'Test AUC: {test_auc}')
    print(f'Test F1 score: {f1_score}')
    
    # Classification report
    predictions = model.predict(test_images)
    y_pred = predictions.argmax(axis=1)
    y_true = test_labels.argmax(axis=1)
    print("Classification Report:\n", classification_report(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Train and evaluate with k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True)
for train_index, val_index in kf.split(train_images):
    model = create_model()
    model.fit(train_images[train_index], train_labels[train_index], epochs=10, validation_data=(train_images[val_index], train_labels[val_index]))
    evaluate_model(model, test_images, test_labels)
