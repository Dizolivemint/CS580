import numpy as np
import tensorflow as tf
from itertools import product
from src.split import train_valid_split
from src.cnn_model import build_cnn_model # You will implement this
from src.loader import load_images # You will implement this
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_cnn_config(
    x, y,
    model_config,
    train_config,
    random_seed=42
):
    """
    Trains and evaluates a single CNN config with reproducibility.
    Returns evaluation metrics and best threshold.
    """
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    # Split data
    x_train, x_valid, y_train, y_valid = train_valid_split(
        x, y, valid_size=train_config.get('valid_size', 0.1), seed=random_seed
    )
    
    # Build model
    model = build_cnn_model(**model_config)
    
    # Early stopping callback
    callbacks = []
    if train_config.get('early_stopping', True):
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=train_config.get('patience', 2),
            restore_best_weights=True
        ))
    
    # Fit
    history = model.fit(
        x_train, y_train,
        batch_size=train_config.get('batch_size', 32),
        epochs=train_config.get('epochs', 10),
        validation_data=(x_valid, y_valid),
        shuffle=True,
        callbacks=callbacks,
        verbose=0
    )
    
    # Predict and optimize threshold
    y_pred_valid = model.predict(x_valid).flatten()
    best_threshold, best_f1 = find_best_threshold(y_valid, y_pred_valid)
    y_pred_bin = (y_pred_valid > best_threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_valid, y_pred_bin),
        'precision': precision_score(y_valid, y_pred_bin, zero_division=0),
        'recall': recall_score(y_valid, y_pred_bin, zero_division=0),
        'f1': f1_score(y_valid, y_pred_bin, zero_division=0),
        'best_threshold': best_threshold,
        'epochs': len(history.history['loss']),
    }
    return metrics

def find_best_threshold(y_true, y_probs, metric='f1'):
    """Finds the threshold with highest F1 (or other) score."""
    best_t, best_s = 0.5, 0
    for t in np.linspace(0.1, 0.9, 50):
        pred = (y_probs > t).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, pred, zero_division=0)
        else:
            score = f1_score(y_true, pred, zero_division=0)
        if score > best_s:
            best_t, best_s = t, score
    return best_t, best_s

def grid_search(x, y, model_param_grid, train_param_grid, n_seeds=3):
    """
    Evaluates all parameter grid combinations (model and train configs),
    averaging metrics across seeds.
    """
    keys_model = list(model_param_grid.keys())
    keys_train = list(train_param_grid.keys())
    combos = list(product(
        *[model_param_grid[k] for k in keys_model],
        *[train_param_grid[k] for k in keys_train]
    ))
    
    all_results = []
    for combo in combos:
        # Split model and train params
        model_config = dict(zip(keys_model, combo[:len(keys_model)]))
        train_config = dict(zip(keys_train, combo[len(keys_model):]))
        print(f"Evaluating config: {model_config} + {train_config}")
        
        # Aggregate metrics over seeds
        metrics_list = []
        for seed in range(n_seeds):
            metrics = evaluate_cnn_config(x, y, model_config, train_config, random_seed=seed)
            metrics_list.append(metrics)
        
        # Average results
        avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
        avg_metrics.update(model_config)
        avg_metrics.update(train_config)
        all_results.append(avg_metrics)
    
    return all_results
