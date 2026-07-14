import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
import matplotlib.pyplot as plt

# =====================================
# Configuration
# =====================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

DATASET_PATH = "dataset/data"

# =====================================
# Data Augmentation
# =====================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,

    validation_split=0.20,

    rotation_range=20,

    zoom_range=0.20,

    width_shift_range=0.20,

    height_shift_range=0.20,

    shear_range=0.20,

    horizontal_flip=True,

    fill_mode="nearest"
)

# =====================================
# Training Generator
# =====================================

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training",

    shuffle=True
)

# =====================================
# Validation Generator
# =====================================

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation",

    shuffle=False
)

# =====================================
# Check Dataset
# =====================================

print("\nClass Mapping")
print(train_generator.class_indices)

print("\nTraining Images :", train_generator.samples)
print("Validation Images :", validation_generator.samples)

# =====================================
# Build MobileNetV2 Model
# =====================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.3)(x)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(0.2)(x)

predictions = Dense(
    2,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n")
model.summary()
# =====================================
# Callbacks
# =====================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# =====================================
# Train Model
# =====================================

history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS,

    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint
    ]

)

# =====================================
# Evaluate
# =====================================

loss, accuracy = model.evaluate(validation_generator)

print("\nValidation Accuracy : {:.2f}%".format(accuracy * 100))
print("Validation Loss : {:.4f}".format(loss))

# =====================================
# Save Final Model
# =====================================

model.save("model.keras")

print("\nModel Saved Successfully!")

# =====================================
# Accuracy Graph
# =====================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.plot(history.history["accuracy"], label="Train")

plt.plot(history.history["val_accuracy"], label="Validation")

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

# =====================================
# Loss Graph
# =====================================

plt.subplot(1,2,2)

plt.plot(history.history["loss"], label="Train")

plt.plot(history.history["val_loss"], label="Validation")

plt.title("Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.tight_layout()

plt.show()