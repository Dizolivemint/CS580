import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepchem as dc
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import warnings
import pandas as pd
from itertools import product
import time
import datetime
import os
import glob
import json

# Suppress the specific deprecation warnings
warnings.filterwarnings('ignore', message='please use MorganGenerator')

def load_latest_model_and_threshold(model_prefix="tox21_model_"):
    """Load the most recent .h5 model and its threshold JSON file. Returns (model, threshold)."""
    model_files = glob.glob(f"{model_prefix}*.h5")
    if not model_files:
        print("No saved models found.")
        return None, None

    model_files.sort(reverse=True)
    latest_model = model_files[0]
    print(f"Loading latest model: {latest_model}")

    model = tf.keras.models.load_model(latest_model)

    # Load corresponding threshold JSON
    threshold_path = latest_model.replace(".h5", "_threshold.json")
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            threshold_data = json.load(f)
            best_threshold = threshold_data.get("best_threshold", 0.5)
        print(f"Loaded threshold: {best_threshold:.2f}")
    else:
        best_threshold = 0.5
        print("Threshold file not found. Using default: 0.5")

    return model, best_threshold
  
# Function to load and prepare the Tox21 data
def load_tox21_data():
    _, (train, valid, test), _ = dc.molnet.load_tox21()
    train_X, train_y, train_w = train.X, train.y, train.w
    valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
    test_X, test_y, test_w = test.X, test.y, test.w

    # Remove extra tasks (focus on first task only)
    train_y = train_y[:, 0]
    valid_y = valid_y[:, 0]
    test_y = test_y[:, 0]
    train_w = train_w[:, 0]
    valid_w = valid_w[:, 0]
    test_w = test_w[:, 0]
    
    return train_X, train_y, train_w, valid_X, valid_y, valid_w, test_X, test_y, test_w

def find_best_threshold(y_true, y_pred_probs, metric='f1', thresholds=None, verbose=False, adaptive=True):
    """
    Finds the best threshold for binary classification predictions using coarse + fine sweep.
    
    Parameters:
    - y_true: Ground truth binary labels (0 or 1)
    - y_pred_probs: Predicted probabilities (from model.predict)
    - metric: Which metric to optimize ('f1', 'precision', 'recall')
    - thresholds: If provided, skips adaptive zooming and uses only these thresholds.
    - verbose: If True, prints all metric values per threshold
    - adaptive: If True and thresholds is None, uses coarse + fine zooming
    
    Returns:
    - best_thresh: The threshold that gave the best score
    - best_score: The corresponding metric score
    """
    def compute_score(thresh_list):
        best_t = 0.5
        best_s = 0
        for thresh in thresh_list:
            pred_binary = (y_pred_probs > thresh).astype(int).flatten()
            try:
                if metric == 'f1':
                    score = f1_score(y_true, pred_binary, zero_division=0)
                elif metric == 'precision':
                    score = precision_score(y_true, pred_binary, zero_division=0)
                elif metric == 'recall':
                    score = recall_score(y_true, pred_binary, zero_division=0)
                else:
                    raise ValueError("Unsupported metric: choose 'f1', 'precision', or 'recall'")
                
                if verbose:
                    print(f"Threshold: {thresh:.4f} -> {metric}: {score:.4f}")
                
                if score > best_s:
                    best_s = score
                    best_t = thresh
            except:
                continue
        return best_t, best_s

    if thresholds is not None:
        return compute_score(thresholds)

    if adaptive:
        # Step 1: Coarse search
        coarse_thresholds = np.linspace(0.1, 0.9, 20)
        coarse_best_thresh, _ = compute_score(coarse_thresholds)

        # Step 2: Fine zoom around the best coarse threshold (±0.05)
        lower = max(0.0, coarse_best_thresh - 0.05)
        upper = min(1.0, coarse_best_thresh + 0.05)
        fine_thresholds = np.linspace(lower, upper, 50)
        return compute_score(fine_thresholds)
    
    # If not adaptive, default to fixed sweep
    return compute_score(np.linspace(0.1, 0.9, 50))

def build_model(input_dim, hidden_units, n_layers, dropout_rate, learning_rate):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    for _ in range(n_layers):
        model.add(tf.keras.layers.Dense(hidden_units, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model
  
def evaluate_predictions(y_true, y_pred_probs, threshold=0.5):
    y_pred_binary = (y_pred_probs > threshold).astype(int).flatten()
    return {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0)
    }

def evaluate_hyperparams(hidden_units, n_layers, learning_rate, dropout_rate, 
                        weight_positives, class_weight_value, n_epochs, batch_size, 
                        early_stopping_patience, random_seed):
    
    # Set seeds for reproducibility
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # Load data
    train_X, train_y, _, valid_X, valid_y, _, *_ = load_tox21_data()

    d = train_X.shape[1]
    
    # Build and compile model
    model = build_model(d, hidden_units, n_layers, dropout_rate, learning_rate)

    class_weights = {0: 1.0, 1: class_weight_value} if weight_positives else None

    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True))

    # Train
    history = model.fit(
        train_X, train_y,
        validation_data=(valid_X, valid_y),
        class_weight=class_weights,
        batch_size=batch_size,
        epochs=n_epochs,
        callbacks=callbacks,
        verbose=0
    )

    # Predict on validation set
    valid_y_pred = model.predict(valid_X, verbose=0)

    # Metrics at default threshold
    base_metrics = evaluate_predictions(valid_y, valid_y_pred, threshold=0.5)

    # Find best threshold
    best_thresh, best_f1 = find_best_threshold(valid_y, valid_y_pred, metric='f1', verbose=False)

    # Metrics at optimal threshold
    opt_metrics = evaluate_predictions(valid_y, valid_y_pred, threshold=best_thresh)

    return {
        **base_metrics,
        'auc': roc_auc_score(valid_y, valid_y_pred),
        'best_threshold': best_thresh,
        'opt_accuracy': opt_metrics['accuracy'],
        'opt_precision': opt_metrics['precision'],
        'opt_recall': opt_metrics['recall'],
        'opt_f1': opt_metrics['f1'],
        'actual_epochs': len(history.history['loss'])
    }

# Main function to perform hyperparameter tuning
def hyperparameter_tuning():
    print("Starting hyperparameter tuning for Tox21 model...")
    
    # Define hyperparameter grid
    param_grid = {
        'hidden_units': [50, 100],                        # Size of hidden layers
        'n_layers': [1, 2],                               # Number of hidden layers
        'learning_rate': [0.001, 0.0005],                 # Learning rate
        'dropout_rate': [0.2, 0.3, 0.5],                  # Dropout rate
        'weight_positives': [True],                       # Always use class weighting
        'class_weight_value': [3.0, 5.0, 8.0],            # Weighting factor for positive class
        'n_epochs': [10],                                 # Max epochs (will use early stopping)
        'batch_size': [64],
        'early_stopping_patience': [2],                   # Early stopping patience
        'repetitions': [3]                                # Number of repetitions per config
    }
    
    # Create all possible combinations of hyperparameters
    keys = [k for k in param_grid.keys() if k != 'repetitions']
    values = [param_grid[k] for k in keys]
    combinations = list(product(*values))
    
    # Calculate total number of models to train
    total_configs = len(combinations)
    total_runs = total_configs * param_grid['repetitions'][0]
    print(f"Will evaluate {total_configs} different configurations, {total_runs} total model trainings")
    
    # Prepare results storage
    all_results = []
    best_config = None
    best_f1 = 0
    
    # Counter for progress tracking
    run_count = 0
    start_time = time.time()
    
    # Evaluate each hyperparameter combination
    for combo in combinations:
        # Convert current combination to dictionary
        config = dict(zip(keys, combo))
        
        # Track results over repetitions
        rep_results = []
        
        # Run multiple times with different seeds
        for seed in range(1, param_grid['repetitions'][0] + 1):
            run_count += 1
            
            # Display progress
            print(f"\nRunning configuration {run_count}/{total_runs} " + 
                  f"({100*run_count/total_runs:.1f}%)")
            print(f"Config: {config}, Repetition: {seed}")
            
            # Evaluate with current hyperparameters and seed
            result = evaluate_hyperparams(
                hidden_units=config['hidden_units'],
                n_layers=config['n_layers'],
                learning_rate=config['learning_rate'],
                dropout_rate=config['dropout_rate'],
                weight_positives=config['weight_positives'],
                class_weight_value=config['class_weight_value'],
                n_epochs=config['n_epochs'],
                batch_size=config['batch_size'],
                early_stopping_patience=config['early_stopping_patience'],
                random_seed=seed
            )
            
            # Add config info to result
            result.update(config)
            result['seed'] = seed
            
            # Store individual run result
            all_results.append(result)
            rep_results.append(result)
            
            # Print current results
            print(f"AUC: {result['auc']:.4f}, F1: {result['f1']:.4f}, " + 
                  f"Opt F1: {result['opt_f1']:.4f} at threshold {result['best_threshold']:.2f}")
        
        # Calculate average metrics across repetitions
        avg_opt_f1 = np.mean([r['opt_f1'] for r in rep_results])
        avg_auc = np.mean([r['auc'] for r in rep_results])
        
        print(f"\nAverage for config - AUC: {avg_auc:.4f}, Opt F1: {avg_opt_f1:.4f}")
        
        # Check if this is the best configuration so far
        if avg_opt_f1 > best_f1:
            best_f1 = avg_opt_f1
            best_config = config
            print("New best configuration found!")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nHyperparameter tuning completed in {elapsed_time/60:.2f} minutes")
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    # Average results per configuration
    avg_results = results_df.groupby([
        'hidden_units', 'n_layers', 'learning_rate', 'dropout_rate', 
        'weight_positives', 'class_weight_value', 'n_epochs', 
        'batch_size', 'early_stopping_patience'
    ]).mean().reset_index()
    
    # Sort by optimized F1 score
    avg_results = avg_results.sort_values('opt_f1', ascending=False)
    
    # Display top 5 configurations
    print("\nTop 5 configurations by average optimized F1 score:")
    for i, row in avg_results.head(5).iterrows():
        print(f"Rank {i+1}:")
        print(f"  Hidden Units: {row['hidden_units']}, Layers: {row['n_layers']}")
        print(f"  Learning Rate: {row['learning_rate']}, Dropout: {row['dropout_rate']}")
        print(f"  Class Weight: {row['class_weight_value']}, Batch Size: {row['batch_size']}")
        print(f"  Metrics - AUC: {row['auc']:.4f}, Opt F1: {row['opt_f1']:.4f}")
        print(f"  Precision: {row['opt_precision']:.4f}, Recall: {row['opt_recall']:.4f}")
        print(f"  Best Threshold: {row['best_threshold']:.2f}")
        print()
    
    # Plot distribution of F1 scores
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['opt_f1'], bins=20)
    plt.title('Distribution of Optimized F1 Scores Across All Models')
    plt.xlabel('Optimized F1 Score')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig('f1_distribution.png')
    
    # Plot relationship between key hyperparameters and F1 score
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot effect of hidden units
    axes[0, 0].boxplot([results_df[results_df['hidden_units'] == hu]['opt_f1'] 
                        for hu in param_grid['hidden_units']], 
                        labels=param_grid['hidden_units'])
    axes[0, 0].set_title('Effect of Hidden Units on F1 Score')
    axes[0, 0].set_xlabel('Number of Hidden Units')
    axes[0, 0].set_ylabel('Optimized F1 Score')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot effect of dropout rate
    axes[0, 1].boxplot([results_df[results_df['dropout_rate'] == dr]['opt_f1'] 
                        for dr in param_grid['dropout_rate']], 
                        labels=[str(dr) for dr in param_grid['dropout_rate']])
    axes[0, 1].set_title('Effect of Dropout Rate on F1 Score')
    axes[0, 1].set_xlabel('Dropout Rate')
    axes[0, 1].set_ylabel('Optimized F1 Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot effect of class weight
    axes[1, 0].boxplot([results_df[results_df['class_weight_value'] == cw]['opt_f1'] 
                        for cw in param_grid['class_weight_value']], 
                        labels=[str(cw) for cw in param_grid['class_weight_value']])
    axes[1, 0].set_title('Effect of Class Weight on F1 Score')
    axes[1, 0].set_xlabel('Class Weight Value')
    axes[1, 0].set_ylabel('Optimized F1 Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot effect of learning rate
    axes[1, 1].boxplot([results_df[results_df['learning_rate'] == lr]['opt_f1'] 
                        for lr in param_grid['learning_rate']], 
                        labels=[str(lr) for lr in param_grid['learning_rate']])
    axes[1, 1].set_title('Effect of Learning Rate on F1 Score')
    axes[1, 1].set_xlabel('Learning Rate')
    axes[1, 1].set_ylabel('Optimized F1 Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_effects.png')
    
    # Save results to CSV for further analysis
    results_df.to_csv('tuning_results.csv', index=False)
    avg_results.to_csv('avg_tuning_results.csv', index=False)
    
    return results_df, avg_results, best_config

if __name__ == "__main__":
    # Check TensorFlow version and GPU availability
    print("TensorFlow version:", tf.__version__)
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    print("GPU Details:", tf.config.list_physical_devices('GPU'))
    
    # Load data for evaluation
    train_X, train_y, train_w, valid_X, valid_y, valid_w, test_X, test_y, test_w = load_tox21_data()
    
    # Try to load a previously saved model and threshold
    final_model, best_threshold = load_latest_model_and_threshold()

    if final_model is not None:
        print("\nLoaded existing model and threshold — skipping training.")
        # proceed to evaluate on test set using best_threshold
        
        # Combine train+val labels and predictions for threshold tuning
        combined_X = np.vstack((train_X, valid_X))
        combined_y = np.concatenate((train_y, valid_y))
        combined_y_pred = final_model.predict(combined_X)

        # Find optimal threshold from combined train+val
        best_threshold, _ = find_best_threshold(combined_y, combined_y_pred, metric='f1', verbose=True)

        test_y_pred = final_model.predict(test_X)
        test_y_pred_binary = (test_y_pred > best_threshold).astype(int).flatten()

        test_accuracy = accuracy_score(test_y, test_y_pred_binary)
        test_auc = roc_auc_score(test_y, test_y_pred)
        test_precision = precision_score(test_y, test_y_pred_binary, zero_division=0)
        test_recall = recall_score(test_y, test_y_pred_binary, zero_division=0)
        test_f1 = f1_score(test_y, test_y_pred_binary, zero_division=0)

        print("\nFinal Model Performance on Test Set:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"AUC-ROC: {test_auc:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall: {test_recall:.4f}")
        print(f"F1 Score: {test_f1:.4f}")

    else:
        # Continue with hyperparameter tuning and training
        results_df, avg_results, best_config = hyperparameter_tuning()
        
        print("\nBest configuration found:")
        print(best_config)
        
        print("\nTraining final model with best configuration...")
        np.random.seed(42)
        tf.random.set_seed(42)
        
        combined_X = np.vstack((train_X, valid_X))
        combined_y = np.concatenate((train_y, valid_y))
        
        d = train_X.shape[1]
        final_model = tf.keras.Sequential()
        final_model.add(tf.keras.layers.Input(shape=(d,)))

        for _ in range(best_config['n_layers']):
            final_model.add(tf.keras.layers.Dense(best_config['hidden_units'], activation='relu'))
            final_model.add(tf.keras.layers.Dropout(best_config['dropout_rate']))

        final_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        class_weights = {0: 1.0, 1: best_config['class_weight_value']} if best_config['weight_positives'] else None

        final_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=best_config['learning_rate']),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy', tf.keras.metrics.AUC(), 
                    tf.keras.metrics.Precision(), 
                    tf.keras.metrics.Recall()]
        )

        final_model.fit(
            combined_X, combined_y,
            epochs=best_config['n_epochs'],
            batch_size=best_config['batch_size'],
            class_weight=class_weights,
            verbose=1
        )

        # Predict + optimize threshold
        combined_y_pred = final_model.predict(combined_X)
        best_threshold, best_f1 = find_best_threshold(combined_y, combined_y_pred, metric='f1', verbose=True)

        test_y_pred = final_model.predict(test_X)
        test_y_pred_binary = (test_y_pred > best_threshold).astype(int).flatten()

        test_accuracy = accuracy_score(test_y, test_y_pred_binary)
        test_auc = roc_auc_score(test_y, test_y_pred)
        test_precision = precision_score(test_y, test_y_pred_binary, zero_division=0)
        test_recall = recall_score(test_y, test_y_pred_binary, zero_division=0)
        test_f1 = f1_score(test_y, test_y_pred_binary, zero_division=0)

        print("\nFinal Model Performance on Test Set:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"AUC-ROC: {test_auc:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall: {test_recall:.4f}")
        print(f"F1 Score: {test_f1:.4f}")

        # Save model
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        model_path = f"tox21_model_{timestamp}.h5"
        final_model.save(model_path)
        print(f"Final model saved to: {model_path}")