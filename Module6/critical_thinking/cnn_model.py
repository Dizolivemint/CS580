import torch
import torch.nn as nn
import torch.nn.functional as F

def build_cnn_model(img_shape, conv_filters=24, conv2_filters=None, dense_units=128, learning_rate=1e-4):
    """
    Build a CNN model for binary classification using PyTorch.
    
    Args:
        img_shape: Tuple of (height, width, channels)
        conv_filters: Number of filters for first conv layer
        conv2_filters: Number of filters for second conv layer (None = no second layer)
        dense_units: Number of units in dense layers
        learning_rate: Learning rate for optimizer
        
    Returns:
        model: A PyTorch model
    """
    # PyTorch expects channels first: (channels, height, width)
    channels, height, width = img_shape[0], img_shape[1], img_shape[2]
    
    class CatDogCNN(nn.Module):
        def __init__(self):
            super(CatDogCNN, self).__init__()
            # First conv layer
            self.conv1 = nn.Conv2d(channels, conv_filters, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Calculate size after first conv + pool
            conv1_size = height // 2, width // 2
            
            # Optional second conv layer
            self.has_conv2 = conv2_filters is not None
            if self.has_conv2:
                self.conv2 = nn.Conv2d(conv_filters, conv2_filters, kernel_size=3, padding=1)
                conv2_size = conv1_size[0] // 2, conv1_size[1] // 2
                flatten_size = conv2_filters * conv2_size[0] * conv2_size[1]
            else:
                flatten_size = conv_filters * conv1_size[0] * conv1_size[1]
            
            # Fully connected layers
            self.fc1 = nn.Linear(flatten_size, dense_units)
            self.fc2 = nn.Linear(dense_units, dense_units)
            self.fc3 = nn.Linear(dense_units, 1)
        
        def forward(self, x):
            # First conv block
            x = self.pool(F.relu(self.conv1(x)))
            
            # Optional second conv block
            if self.has_conv2:
                x = self.pool(F.relu(self.conv2(x)))
            
            # Flatten
            x = x.view(x.size(0), -1)
            
            # Fully connected layers
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
            
            return x
    
    # Create model, optimizer, and loss function
    model = CatDogCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    return model, optimizer, criterion