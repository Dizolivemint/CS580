import torch
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

def train_model(
    model, 
    optimizer, 
    criterion, 
    train_loader, 
    valid_loader, 
    epochs=10, 
    device=None,
    early_stopping=True,
    patience=2,
    verbose=1
):
    """
    Train a PyTorch model.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        criterion: Loss function
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        epochs: Number of epochs to train for
        device: Device to train on ('cuda' or 'cpu')
        early_stopping: Whether to use early stopping
        patience: Patience for early stopping
        verbose: Print training progress
        
    Returns:
        model: Trained model
        history: Training history
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'valid_loss': [],
        'train_acc': [],
        'valid_acc': []
    }
    
    # Early stopping variables
    best_valid_loss = float('inf')
    best_model_state = None
    no_improve_count = 0
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1, verbose=verbose > 0)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track loss and accuracy
            train_loss += loss.item() * images.size(0)
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels.unsqueeze(1)).sum().item()
            train_total += labels.size(0)
        
        # Calculate average loss and accuracy
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        valid_loss = 0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1))
                
                # Track loss and accuracy
                valid_loss += loss.item() * images.size(0)
                predicted = (outputs > 0.5).float()
                valid_correct += (predicted == labels.unsqueeze(1)).sum().item()
                valid_total += labels.size(0)
        
        # Calculate average loss and accuracy
        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_acc = valid_correct / valid_total
        
        # Update learning rate
        scheduler.step(valid_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)
        
        # Print progress
        if verbose > 0:
            print(f'Epoch {epoch+1}/{epochs} | ' +
                  f'Train Loss: {train_loss:.4f} | ' +
                  f'Train Acc: {train_acc:.4f} | ' +
                  f'Valid Loss: {valid_loss:.4f} | ' +
                  f'Valid Acc: {valid_acc:.4f} | ' +
                  f'Time: {time.time() - start_time:.2f}s')
        
        # Early stopping check
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = model.state_dict().copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if early_stopping and no_improve_count >= patience:
            if verbose > 0:
                print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model, history

def predict(model, data_loader, device=None):
    """
    Make predictions with a trained model.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for prediction
        device: Device to use for prediction
        
    Returns:
        y_pred: Predicted probabilities
        y_true: True labels
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            y_pred.extend(outputs.cpu().numpy())
            y_true.extend(labels.numpy())
    
    return np.array(y_pred).flatten(), np.array(y_true)

def save_model(model, path):
    """
    Save a PyTorch model.
    
    Args:
        model: PyTorch model
        path: Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")