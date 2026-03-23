import os, cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def load_images(folder, label):
    X, y = [], []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.resize(img, (224,224))
        img = img / 255.0

        X.append(img)
        y.append(label)

    return X, y

real_X, real_y = load_images("dataset/real_frames", 0)
fake_X, fake_y = load_images("dataset/fake_frames", 1)

X = np.array(real_X + fake_X)
y = np.array(real_y + fake_y)

print("Total images:", len(X))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=8)

loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

model.save("ml/deepfake_model.h5")