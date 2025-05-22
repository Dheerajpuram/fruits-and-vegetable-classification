# main.py
import os
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path setup
base_dir = 'fruits-vegetables-images'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Image data generator
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

print("Loading datasets...")

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse', shuffle=True)
valid_gen = datagen.flow_from_directory(valid_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse', shuffle=False)

# Get class names
class_names = list(train_gen.class_indices.keys())

# Flatten images
def flatten_generator(gen):
    X = []
    y = []
    for images, labels in gen:
        X.append(images.reshape(images.shape[0], -1))  # images already rescaled by datagen
        y.append(labels)
        if len(X)*BATCH_SIZE >= gen.samples:
            break
    return np.vstack(X), np.concatenate(y)

X_train, y_train = flatten_generator(train_gen)
X_valid, y_valid = flatten_generator(valid_gen)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Models to try
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB()
}

# MLflow setup
mlflow.set_experiment("Fruit and Vegetable Classification")
best_accuracy = 0
best_model = None
best_model_name = ""

# Training and Evaluation
for name, model in models.items():
    print(f"\nTraining {name}...")
    with mlflow.start_run(run_name=name):
        if name == "LogisticRegression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_valid_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_valid)

        acc = accuracy_score(y_valid, preds)
        print(f"{name} Validation Accuracy: {acc:.4f}")

        mlflow.log_param("model", name)
        mlflow.log_metric("val_accuracy", acc)
        mlflow.sklearn.log_model(model, name)

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name

print(f"\n✅ Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# Save the best model
if not os.path.exists('models'):
    os.makedirs('models')

print(f"\n💾 Saving Best Model: {best_model_name}...")
joblib.dump(best_model, 'models/final_best_model2.pkl')
joblib.dump(scaler, 'models/scaler2.pkl')
joblib.dump(class_names, 'models/class_names2.pkl')
print("✅ Model, Scaler, and Class Names saved successfully in the 'models' folder.")