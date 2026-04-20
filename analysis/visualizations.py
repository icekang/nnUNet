import matplotlib.pyplot as plt
import numpy as np

def plot_texture_variant_embeddings(results, figsize=(10, 14)):
    """
    Plot embeddings for texture variants with color gradient showing intensity.
    
    Args:
        results: Dictionary from compute_embeddings_for_texture_variants
        figsize: Figure size tuple
    """
    models = list(results.keys())
    n_models = len(models)
    
    # Create figure with subplots (one per model, stacked vertically)
    fig, axes = plt.subplots(n_models, 1, figsize=(figsize[0], figsize[1]))
    if n_models == 1:
        axes = [axes]
    
    # Color map: from blue (reduce) through gray (original) to red (enhance)
    cmap_reduce = plt.cm.Blues_r(np.linspace(0.4, 0.8, 3))  # Blue gradient for reduction
    cmap_enhance = plt.cm.Reds(np.linspace(0.4, 0.8, 3))    # Red gradient for enhancement
    
    for ax_idx, model_name in enumerate(models):
        ax = axes[ax_idx]
        model_data = results[model_name]
        coords = model_data['coords']
        labels = model_data['labels']
        factors = model_data['factors']
        
        # Assign colors based on factor
        colors_list = []
        for factor, label in zip(factors, labels):
            if factor < 1.0:
                # Reduction: use blue gradient
                idx = int((1.0 - factor) * 2)  # 0 to 2
                idx = min(idx, 2)
                colors_list.append(cmap_reduce[idx])
            elif factor == 1.0:
                # Original: gray
                colors_list.append([0.5, 0.5, 0.5, 1.0])
            else:
                # Enhancement: use red gradient
                idx = int((factor - 1.0) * 1.5)  # 0 to 2
                idx = min(idx, 2)
                colors_list.append(cmap_enhance[idx])
        
        # Plot scatter
        for i, (x, y) in enumerate(coords):
            ax.scatter(x, y, s=150, c=[colors_list[i]], edgecolors='black', linewidth=2, zorder=3)
            # Add label next to point
            ax.annotate(labels[i], (x, y), xytext=(5, 5), textcoords='offset points', 
                       fontsize=9, ha='left', bbox=dict(boxstyle='round,pad=0.3', 
                       facecolor=colors_list[i], alpha=0.7, edgecolor='none'))
        
        ax.set_xlabel('PCA 1', fontsize=11, fontweight='bold')
        ax.set_ylabel('PCA 2', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    
    plt.suptitle('Texture Modification Impact on Embeddings\n(Blue=Reduce, Gray=Original, Red=Enhance)', 
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_texture_variant_trajectories(results, figsize=(10, 14)):
    """
    Plot embeddings with connecting lines showing trajectory from reduction to enhancement.
    
    Args:
        results: Dictionary from compute_embeddings_for_texture_variants
        figsize: Figure size tuple
    """
    models = list(results.keys())
    n_models = len(models)
    
    fig, axes = plt.subplots(n_models, 1, figsize=(figsize[0], figsize[1]))
    if n_models == 1:
        axes = [axes]
    
    # Color gradient from blue to red
    n_points = None
    labels_list = None
    
    for ax_idx, model_name in enumerate(models):
        ax = axes[ax_idx]
        model_data = results[model_name]
        coords = model_data['coords']
        labels = model_data['labels']
        factors = model_data['factors']
        
        if n_points is None:
            n_points = len(coords)
            labels_list = labels
        
        # Create color gradient
        cmap = plt.cm.RdYlBu_r
        colors_normalized = np.linspace(0, 1, n_points)
        colors_list = [cmap(c) for c in colors_normalized]
        
        # Plot line trajectory
        ax.plot(coords[:, 0], coords[:, 1], 'k--', alpha=0.3, linewidth=1.5, zorder=1)
        
        # Plot scatter with gradient colors
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors_normalized, cmap='RdYlBu_r',
                           s=200, edgecolors='black', linewidth=2, zorder=3)
        
        # Add labels and arrows
        for i in range(len(coords)):
            ax.annotate(f'{i+1}', (coords[i, 0], coords[i, 1]), 
                       fontsize=8, ha='center', va='center', color='white', fontweight='bold')
        
        ax.set_xlabel('PCA 1', fontsize=11, fontweight='bold')
        ax.set_ylabel('PCA 2', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Reduce ← Texture Intensity → Enhance', fontsize=10)
    
    plt.suptitle('Embedding Trajectory: Texture Modification Path\n(Shows how embeddings move in feature space)', 
                fontsize=13, fontweight='bold')
    
    # Create horizontal legend below the figure (compact)
    cmap = plt.cm.RdYlBu_r
    colors_normalized = np.linspace(0, 1, n_points)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(c), 
                         markeredgecolor='black', markersize=6, label=f'{i+1}. {label}')
               for i, (c, label) in enumerate(zip(colors_normalized, labels_list))]
    # Use multiple rows if too many labels to fit in one row
    ncol = min(n_points, max(3, n_points // 2))  # Wrap to 2 rows if many labels
    fig.legend(handles=handles, loc='lower center', ncol=ncol, 
              bbox_to_anchor=(0.5, -0.01), fontsize=7, frameon=True, 
              framealpha=0.9, edgecolor='gray', handletextpad=0.4, columnspacing=0.8)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.show()


def plot_texture_variant_embeddings_v2(results, figsize=(10, 14)):
    """
    Plot embeddings for texture variants with SHARED x and y axis scales across all models.
    This allows visual comparison of embedding magnitudes between models.
    
    Args:
        results: Dictionary from compute_embeddings_for_texture_variants
        figsize: Figure size tuple
    """
    models = list(results.keys())
    n_models = len(models)
    
    # First pass: compute global axis limits across all models
    all_x = []
    all_y = []
    for model_name in models:
        model_data = results[model_name]
        coords = model_data['coords']
        all_x.extend(coords[:, 0])
        all_y.extend(coords[:, 1])
    
    # Compute limits with some padding
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    x_lim = (x_min - x_pad, x_max + x_pad)
    y_lim = (y_min - y_pad, y_max + y_pad)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_models, 1, figsize=(figsize[0], figsize[1]))
    if n_models == 1:
        axes = [axes]
    
    # Color map: from blue (reduce) through gray (original) to red (enhance)
    cmap_reduce = plt.cm.Blues_r(np.linspace(0.4, 0.8, 3))
    cmap_enhance = plt.cm.Reds(np.linspace(0.4, 0.8, 3))
    
    for ax_idx, model_name in enumerate(models):
        ax = axes[ax_idx]
        model_data = results[model_name]
        coords = model_data['coords']
        labels = model_data['labels']
        factors = model_data['factors']
        
        # Assign colors based on factor
        colors_list = []
        for factor, label in zip(factors, labels):
            if factor < 1.0:
                idx = int((1.0 - factor) * 2)
                idx = min(idx, 2)
                colors_list.append(cmap_reduce[idx])
            elif factor == 1.0:
                colors_list.append([0.5, 0.5, 0.5, 1.0])
            else:
                idx = int((factor - 1.0) * 1.5)
                idx = min(idx, 2)
                colors_list.append(cmap_enhance[idx])
        
        # Plot scatter
        for i, (x, y) in enumerate(coords):
            ax.scatter(x, y, s=150, c=[colors_list[i]], edgecolors='black', linewidth=2, zorder=3)
            ax.annotate(labels[i], (x, y), xytext=(5, 5), textcoords='offset points', 
                       fontsize=9, ha='left', bbox=dict(boxstyle='round,pad=0.3', 
                       facecolor=colors_list[i], alpha=0.7, edgecolor='none'))
        
        ax.set_xlabel('PCA 1', fontsize=11, fontweight='bold')
        ax.set_ylabel('PCA 2', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        # Apply shared axis limits
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
    
    plt.suptitle('Texture Modification Impact on Embeddings\n(Blue=Reduce, Gray=Original, Red=Enhance)\nShared axis scale for comparison', 
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
