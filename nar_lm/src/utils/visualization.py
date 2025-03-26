# src/utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import io
from PIL import Image

def plot_training_history(train_losses, val_losses, save_path=None):
    """Plot training and validation loss history"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if save_path:
        plt.savefig(save_path)
        
    return plt.gcf()

def plot_attention_matrix(attention, tokens=None, title="Attention Matrix", save_path=None):
    """Plot attention matrix as a heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(attention, cmap='viridis', ax=ax)
    
    # Add token labels if provided
    if tokens:
        ax.set_xticks(np.arange(len(tokens)) + 0.5)
        ax.set_yticks(np.arange(len(tokens)) + 0.5)
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens)
    
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path)
        
    return fig

def plot_latent_space_2d(latent_vectors, labels=None, method='pca', title="Latent Space Visualization", save_path=None):
    """Visualize latent space in 2D using PCA or t-SNE"""
    # Reshape if needed
    if len(latent_vectors.shape) > 2:
        # Average across sequence length for each sample
        latent_vectors = latent_vectors.mean(dim=1) if isinstance(latent_vectors, torch.Tensor) else np.mean(latent_vectors, axis=1)
    
    # Convert to numpy if tensor
    if isinstance(latent_vectors, torch.Tensor):
        latent_vectors = latent_vectors.detach().cpu().numpy()
    
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, perplexity=min(30, latent_vectors.shape[0]-1))
    else:
        raise ValueError("Method must be either 'pca' or 'tsne'")
    
    vectors_2d = reducer.fit_transform(latent_vectors)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot with or without labels
    if labels is not None:
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], label=label)
        ax.legend()
    else:
        ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
    
    ax.set_title(f"{title} ({method.upper()})")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, linestyle='--', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        
    return fig

def plot_confidence_distribution(confidence_scores, title="Token Confidence Distribution", save_path=None):
    """Plot histogram of confidence scores"""
    # Convert to numpy if tensor
    if isinstance(confidence_scores, torch.Tensor):
        confidence_scores = confidence_scores.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(confidence_scores, bins=20, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add vertical line for threshold if greater than 0.5
    if np.mean(confidence_scores) > 0.5:
        ax.axvline(x=0.9, color='r', linestyle='--', label="Threshold")
        ax.legend()
    
    if save_path:
        plt.savefig(save_path)
        
    return fig

def plot_generation_progress(refinement_results, title="Generation Refinement Progress", save_path=None):
    """Plot metrics across refinement steps"""
    steps = [result["step"] for result in refinement_results]
    confidences = [result["confidence"] for result in refinement_results]
    
    # Check if changed_tokens exists in all results
    has_changed_tokens = all(["changed_tokens" in result for result in refinement_results[1:]])
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot confidence
    ax1.set_xlabel("Refinement Step")
    ax1.set_ylabel("Average Confidence", color="blue")
    ax1.plot(steps, confidences, 'o-', color="blue", label="Confidence")
    ax1.tick_params(axis='y', labelcolor="blue")
    
    # Plot changed tokens if available
    if has_changed_tokens:
        changed_tokens = [0] + [result["changed_tokens"] for result in refinement_results[1:]]
        
        ax2 = ax1.twinx()
        ax2.set_ylabel("Tokens Changed", color="red")
        ax2.plot(steps, changed_tokens, 'o-', color="red", label="Tokens Changed")
        ax2.tick_params(axis='y', labelcolor="red")
    
    fig.tight_layout()
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        
    return fig

def fig_to_image(fig):
    """Convert matplotlib figure to PIL Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    image = Image.open(buf)
    return image

def plot_side_by_side_comparison(original_text, generated_text, title="Text Comparison", save_path=None):
    """Create side-by-side comparison of original and generated text"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display texts in text boxes
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    ax1.text(0.5, 0.5, original_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='center', horizontalalignment='center', bbox=props)
    ax1.set_title("Original Text")
    ax1.axis('off')
    
    ax2.text(0.5, 0.5, generated_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='center', horizontalalignment='center', bbox=props)
    ax2.set_title("Generated Text")
    ax2.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    return fig

def plot_token_confidence_matrix(token_confidence, tokens, title="Token Confidence Matrix", save_path=None):
    """Create a visualization of token confidence levels"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert tensor to numpy if needed
    if isinstance(token_confidence, torch.Tensor):
        token_confidence = token_confidence.detach().cpu().numpy()
    
    # Plot as horizontal bar chart
    y_pos = np.arange(len(tokens))
    ax.barh(y_pos, token_confidence, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Confidence Score')
    ax.set_title(title)
    
    # Add a vertical line at threshold 0.9
    ax.axvline(x=0.9, color='r', linestyle='--', label="Threshold")
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
        
    return fig