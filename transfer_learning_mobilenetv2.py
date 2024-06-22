import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(train_images)

# Define the model architecture
def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    
    # Freeze all layers in the base model
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create the new model on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
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

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model_mobilenetv2.h5', save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

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

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Train and evaluate with k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True)
for train_index, val_index in kf.split(train_images):
    model = create_model()
    history = model.fit(
        datagen.flow(train_images[train_index], train_labels[train_index], batch_size=64),
        epochs=50,
        validation_data=(train_images[val_index], train_labels[val_index]),
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )
    plot_history(history)
    evaluate_model(model, test_images, test_labels)
    
    # Fine-tune the model by unfreezing the top layers
    for layer in model.layers[0].layers[-10:]:
        layer.trainable = True
    
    # Recompile the model with a lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
    )
    
    # Fine-tune the model
    history_finetune = model.fit(
        datagen.flow(train_images[train_index], train_labels[train_index], batch_size=64),
        epochs=20,
        validation_data=(train_images[val_index], train_labels[val_index]),
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )
    plot_history(history_finetune)
    evaluate_model(model, test_images, test_labels)
