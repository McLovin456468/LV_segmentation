import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import binary_crossentropy
from sklearn.metrics import accuracy_score

from data_loader import load_data
from metrics import compute_soft_dice, dice_loss_metric

def combined_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss_metric(y_true, y_pred)

(_, _), (X_test, y_test), _ = load_data()

model_path = 'saved_models/light_unet_89.h5'
model = load_model(model_path, custom_objects={
    'compute_soft_dice': compute_soft_dice,
    'dice_loss_metric': dice_loss_metric,
    'combined_loss': combined_loss
})

y_pred = model.predict(X_test)
y_pred_bin = (y_pred > 0.5).astype(np.uint8)

y_true_flat = y_test.flatten()
y_pred_flat = y_pred_bin.flatten()
accuracy = accuracy_score(y_true_flat, y_pred_flat)
print(f"Pixel-wise Accuracy on test set: {accuracy:.4f}")

def dice_score(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

dice = dice_score(y_test, y_pred_bin)
print(f"Dice Coefficient on test set: {dice:.4f}")

for i in range(5):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.title('Input')

    plt.subplot(1, 3, 2)
    plt.imshow(y_test[i].squeeze(), cmap='gray')
    plt.title('Ground Truth')

    plt.subplot(1, 3, 3)
    plt.imshow(y_pred_bin[i].squeeze(), cmap='gray')
    plt.title('Prediction')

    plt.tight_layout()
    plt.show()
