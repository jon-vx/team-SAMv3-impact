"""
INIA — Segmentation API
========================
Unified training, evaluation, and inference for UNet architectures.
Supports: unet, unet++, unet3++

Usage in Google Colab:
    !pip install keras_unet_collection
    from INIA import fit, evaluate, predict, get_bboxes, load_data, preprocess

    X_train, y_train, X_test, y_test = load_data()
    model, history = fit("unet++", X_train, y_train, epochs=50)
    results = evaluate(model, X_test, y_test)
    bboxes = get_bboxes(model, X_test)
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from keras_unet_collection import models as kuc_models

# =============================================================================
# Constants — single source of truth for all experiments
# =============================================================================
INPUT_SIZE = (320, 320, 1)
FILTER_NUM = [64, 128, 256, 512]
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.1
SMOOTH = 1e-6

# =============================================================================
# Metrics & Losses
# =============================================================================
bce = tf.keras.losses.BinaryCrossentropy()


def dice_coef(y_true, y_pred, smooth=SMOOTH):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return (2.0 * intersection + smooth) / (denominator + smooth)


def iou_coef(y_true, y_pred, smooth=SMOOTH):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return bce(y_true, y_pred) + dice_loss(y_true, y_pred)


# =============================================================================
# Data Loading & Preprocessing
# =============================================================================
def _pad_to_320(x):
    """Pad array of shape (N, 276, 300) to (N, 320, 320, 1)."""
    return np.pad(x[..., None], ((0, 0), (22, 22), (10, 10), (0, 0)))


def load_data(
    images_path="images.npz",
    masks_path="masks.npz",
    test_split=28,
    seed=42,
):
    """
    Load, preprocess, shuffle, and split the dataset.

    Parameters
    ----------
    images_path : str
        Path to images.npz file.
    masks_path : str
        Path to masks.npz file.
    test_split : int
        Number of samples reserved for test set (default 28).
    seed : int
        Random seed for reproducible shuffling.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
    """
    images = np.load(images_path)["images"]
    masks = np.load(masks_path)["masks"]

    # Crop top 24 rows (equipment overlay)
    images = images[:, 24:]
    masks = masks[:, 24:]

    # Grayscale (take first channel)
    images = images[:, :, :, 0]

    # Normalize and binarize
    images = images / 255.0
    masks = (masks > 0).astype("float32")

    # Pad to 320x320
    images = _pad_to_320(images)
    masks = _pad_to_320(masks)

    # Shuffle
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(images))
    images = images[idx]
    masks = masks[idx]

    # Split
    split_idx = len(images) - test_split
    X_train = images[:split_idx]
    y_train = masks[:split_idx]
    X_test = images[split_idx:]
    y_test = masks[split_idx:]

    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    return X_train, y_train, X_test, y_test


def preprocess(images, masks):
    """
    Apply preprocessing to custom data.
    Expects images (N, 276, 300, C) and masks (N, 276, 300, C).
    Returns padded arrays ready for the model.
    """
    if images.ndim == 4 and images.shape[-1] > 1:
        images = images[:, :, :, 0]
    images = images / 255.0 if images.max() > 1.0 else images
    masks = (masks > 0).astype("float32")
    if masks.ndim == 4:
        masks = masks[:, :, :, 0]
    images = _pad_to_320(images)
    masks = _pad_to_320(masks)
    return images, masks


# =============================================================================
# Model Factory
# =============================================================================
SUPPORTED_MODELS = ["unet", "unet++", "unet3++"]


def get_model(name):
    """
    Build a segmentation model by name.

    Parameters
    ----------
    name : str
        One of 'unet', 'unet++', 'unet3++'.

    Returns
    -------
    tf.keras.Model
    """
    name = name.lower().strip()

    if name == "unet":
        return kuc_models.unet_2d(
            input_size=INPUT_SIZE,
            filter_num=FILTER_NUM,
            n_labels=1,
            output_activation="Sigmoid",
        )
    elif name in ("unet++", "unetpp", "unet_plus"):
        return kuc_models.unet_plus_2d(
            input_size=INPUT_SIZE,
            filter_num=FILTER_NUM,
            n_labels=1,
            output_activation="Sigmoid",
        )
    elif name in ("unet3++", "unet3pp", "unet_3plus"):
        return kuc_models.unet_3plus_2d(
            input_size=INPUT_SIZE,
            filter_num_down=FILTER_NUM,
            n_labels=1,
            output_activation="Sigmoid",
        )
    else:
        raise ValueError(
            f"Unknown model '{name}'. Supported: {SUPPORTED_MODELS}"
        )


# =============================================================================
# Training
# =============================================================================
def fit(
    model_name,
    X_train,
    y_train,
    epochs=50,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    validation_split=VALIDATION_SPLIT,
    callbacks=None,
):
    """
    Build, compile, and train a segmentation model.

    Parameters
    ----------
    model_name : str
        One of 'unet', 'unet++', 'unet3++'.
    X_train : np.ndarray
        Training images, shape (N, 320, 320, 1).
    y_train : np.ndarray
        Training masks, shape (N, 320, 320, 1).
    epochs : int
    batch_size : int
    learning_rate : float
    validation_split : float
    callbacks : list or None
        Keras callbacks. If None, uses EarlyStopping + ModelCheckpoint.

    Returns
    -------
    model : tf.keras.Model
    history : tf.keras.callbacks.History
    """
    model = get_model(model_name)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=bce_dice_loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="bin_acc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            dice_coef,
            iou_coef,
        ],
    )

    if callbacks is None:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_dice_coef",
                patience=10,
                mode="max",
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f"best_{model_name.replace('+', 'p')}.keras",
                monitor="val_dice_coef",
                mode="max",
                save_best_only=True,
            ),
        ]

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks,
    )

    print(f"\n✓ {model_name} trained for {len(history.history['loss'])} epochs")
    return model, history


# =============================================================================
# Evaluation
# =============================================================================
def evaluate(model, X_test, y_test):
    """
    Evaluate a trained model on the test set.

    Returns
    -------
    dict with keys: loss, bin_acc, precision, recall, dice_coef, iou_coef
    """
    results = model.evaluate(X_test, y_test, verbose=1)
    metric_names = [m.name if hasattr(m, "name") else m for m in model.metrics_names]
    results_dict = dict(zip(metric_names, results))

    print("\n--- Evaluation Results ---")
    for k, v in results_dict.items():
        print(f"  {k:>12s}: {v:.4f}")

    return results_dict


# =============================================================================
# Prediction & Bounding Boxes (for Team 2 — SAM)
# =============================================================================
def predict(model, images, threshold=0.5):
    """
    Run inference and return binary masks.

    Parameters
    ----------
    model : tf.keras.Model
    images : np.ndarray
        Shape (N, 320, 320, 1).
    threshold : float

    Returns
    -------
    np.ndarray of shape (N, 320, 320, 1) with uint8 values {0, 1}
    """
    preds = model.predict(images, verbose=0)
    return (preds > threshold).astype(np.uint8)


def _mask_to_bbox(mask_2d):
    """
    Extract bounding box [x_min, y_min, x_max, y_max] from a single 2D mask.
    Returns None if the mask is empty (no foreground).
    """
    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    if not rows.any():
        return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def get_bboxes(model, images, threshold=0.5):
    """
    Run inference and return bounding boxes for each image.

    Parameters
    ----------
    model : tf.keras.Model
    images : np.ndarray
        Shape (N, 320, 320, 1).
    threshold : float

    Returns
    -------
    list of [x_min, y_min, x_max, y_max] or None for empty masks
    """
    masks = predict(model, images, threshold)
    bboxes = []
    for i in range(len(masks)):
        bbox = _mask_to_bbox(masks[i].squeeze())
        bboxes.append(bbox)
    return bboxes


def export_bboxes(model, images, output_path="bboxes.npy", threshold=0.5):
    """
    Run inference, extract bounding boxes, and save to disk.
    Useful for handing off to Team 2 (SAM).

    Parameters
    ----------
    model : tf.keras.Model
    images : np.ndarray
    output_path : str
    threshold : float
    """
    bboxes = get_bboxes(model, images, threshold)
    bboxes_array = np.array(
        [b if b is not None else [-1, -1, -1, -1] for b in bboxes]
    )
    np.save(output_path, bboxes_array)
    print(f"✓ Saved {len(bboxes)} bounding boxes to {output_path}")
    return bboxes_array


# =============================================================================
# Visualization
# =============================================================================
def plot_history(history, model_name="model"):
    """Plot training curves: Dice/IoU, loss/accuracy, precision/recall."""
    hist_df = pd.DataFrame(history.history)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    # Segmentation quality
    seg_cols = [c for c in ["dice_coef", "val_dice_coef", "iou_coef", "val_iou_coef"] if c in hist_df]
    hist_df[seg_cols].plot(ax=axes[0], linewidth=2)
    axes[0].set_title(f"{model_name} — Dice & IoU")
    axes[0].set_ylabel("score")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True)

    # Loss & accuracy
    loss_cols = [c for c in ["loss", "val_loss", "bin_acc", "val_bin_acc"] if c in hist_df]
    hist_df[loss_cols].plot(ax=axes[1], linewidth=2)
    axes[1].set_title("Loss & Accuracy")
    axes[1].set_ylabel("value")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True)

    # Precision & recall
    pr_cols = [c for c in ["precision", "recall", "val_precision", "val_recall"] if c in hist_df]
    if pr_cols:
        hist_df[pr_cols].plot(ax=axes[2], linewidth=2)
        axes[2].set_title("Precision & Recall")
        axes[2].set_ylabel("value")
        axes[2].set_ylim(0, 1.0)
        axes[2].grid(True)

    axes[-1].set_xlabel("epoch")
    plt.tight_layout()
    plt.show()


def plot_predictions(model, X_test, y_test, n=3):
    """Show input, ground truth, predicted mask, and overlay for n samples."""
    for i in range(n):
        idx = np.random.randint(0, len(X_test))
        image = X_test[idx]
        mask = y_test[idx]

        pred = model.predict(np.expand_dims(image, axis=0), verbose=0)
        pred_bin = (pred[0] > 0.5).astype(np.uint8)

        fig, axs = plt.subplots(1, 4, figsize=(16, 6))

        axs[0].imshow(image.squeeze(), cmap="gray")
        axs[0].set_title("Input image")
        axs[0].axis("off")

        axs[1].imshow(mask.squeeze(), cmap="gray")
        axs[1].set_title("Ground truth")
        axs[1].axis("off")

        axs[2].imshow(pred_bin.squeeze(), cmap="gray")
        axs[2].set_title("Predicted mask")
        axs[2].axis("off")

        axs[3].imshow(image.squeeze(), cmap="gray")
        axs[3].imshow(pred_bin.squeeze(), cmap="jet", alpha=0.4)
        axs[3].set_title("Overlay")
        axs[3].axis("off")

        plt.tight_layout()
        plt.show()


# =============================================================================
# Compare All Architectures
# =============================================================================
def compare(X_train, y_train, X_test, y_test, architectures=None, epochs=50):
    """
    Train and evaluate multiple architectures under identical conditions.

    Parameters
    ----------
    architectures : list of str or None
        Defaults to ['unet', 'unet++', 'unet3++'].

    Returns
    -------
    pd.DataFrame with Dice, IoU, and other metrics per architecture
    """
    if architectures is None:
        architectures = ["unet", "unet++", "unet3++"]

    all_results = {}
    all_histories = {}

    for arch in architectures:
        print(f"\n{'='*60}")
        print(f"  Training: {arch}")
        print(f"{'='*60}")

        model, history = fit(arch, X_train, y_train, epochs=epochs)
        results = evaluate(model, X_test, y_test)

        all_results[arch] = results
        all_histories[arch] = history

    # Summary table
    df = pd.DataFrame(all_results).T
    df.index.name = "architecture"
    print(f"\n{'='*60}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(df.to_string())

    return df, all_histories
