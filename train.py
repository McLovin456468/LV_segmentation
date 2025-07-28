import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import binary_crossentropy

from data_loader import load_data
from model import build_light_unet
from metrics import compute_soft_dice, dice_loss_metric


def combined_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss_metric(y_true, y_pred)

(x_train, y_train), (x_test, y_test), (x_val, y_val) = load_data()

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

def data_generator(images, masks, batch_size):
    generator = datagen.flow(images, masks, batch_size=batch_size, seed=42)
    while True:
        x_batch, y_batch = next(generator)
        yield x_batch, y_batch

batch_size = 8
train_gen = data_generator(x_train, y_train, batch_size=batch_size)
steps_per_epoch = len(x_train) // batch_size

model = build_light_unet(input_shape=(256, 256, 1))
optimizer = Adam(learning_rate=1e-4)

model.compile(optimizer=optimizer, loss=combined_loss, metrics=[compute_soft_dice])
model.summary()

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

history = model.fit(
    train_gen,
    validation_data=(x_val, y_val),
    epochs=30,
    batch_size=batch_size,
    steps_per_epoch=steps_per_epoch,
    callbacks=callbacks
)

save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)
model.save(os.path.join(save_dir, f'light_unet_89.h5'))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['compute_soft_dice'], label='Train Dice')
plt.plot(history.history['val_compute_soft_dice'], label='Val Dice')
plt.title('Dice Coefficient')
plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Combined Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
