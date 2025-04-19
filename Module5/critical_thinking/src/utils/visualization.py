# src/utils/visualization.py
import matplotlib.pyplot as plt
import os

def save_loss_plot(g_losses, d_losses, output_dir='output'):
    """
    Save the loss history plot to the specified output directory
    
    Args:
        g_losses (list): Generator loss history
        d_losses (list): Discriminator loss history
        output_dir (str): Directory to save the plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and save the plot
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.savefig(f'{output_dir}/loss_history.png')
    plt.close()