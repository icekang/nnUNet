import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description="Calculate L2 displacement of model embeddings from baseline.")
    parser.add_argument('--input', type=str, default='intensity_variant_results_101-019_patch1.pth',
                        help="Path to the .pth file containing the model embeddings.")
    parser.add_argument('--output-dir', type=str, default='/home/naravich/nnunet_texture_ablations',
                        help="Directory to save the generated plots.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        # Try finding it in the project root if it's not found in the current directory
        project_path = Path('/nfs/erelab001/shared/Computational_Group/Naravich/nnUNet') / args.input
        if project_path.exists():
            input_path = project_path
        else:
            print(f"Error: File {args.input} not found.")
            sys.exit(1)

    print(f"Loading embeddings from {input_path}...")
    try:
        results = torch.load(input_path, map_only=True) if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'safe_globals') else torch.load(input_path)
    except Exception as e:
        # Fallback to normal torch load if weights_only warnings or other issues arise
        results = torch.load(input_path)

    # Determine type of perturbation from filename
    perturbation_type = "Intensity" if "intensity" in input_path.name.lower() else "Texture"
    patch_name = input_path.stem.split('_')[-1] if 'patch' in input_path.stem else "full"
    print(f"Perturbation Type: {perturbation_type} | Patch: {patch_name}")

    # Set premium style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

    colors = {
        'LaW': '#ff7f0e',
        'Genesis': '#d62728',
        'CLIP': '#e377c2',
        'No Pretrain': '#17becf',
        'No_Pretrain': '#17becf'
    }
    markers = {
        'LaW': 'o',
        'Genesis': 's',
        'CLIP': '^',
        'No Pretrain': 'd',
        'No_Pretrain': 'd'
    }

    # Aggregate results for comparison
    comparison_data = []

    for model_name, model_data in results.items():
        embeddings = model_data['embeddings']
        factors = np.array(model_data['factors'])
        labels = model_data['labels']

        # Find original (baseline) index where factor is 1.0 or label is 'Original'
        orig_indices = np.where(np.isclose(factors, 1.0))[0]
        if len(orig_indices) == 0:
            # Fallback to search in labels
            orig_indices = [i for i, l in enumerate(labels) if 'original' in l.lower()]
        
        if len(orig_indices) == 0:
            print(f"Warning: Could not find baseline (factor=1.0 / 'Original') for {model_name}. Skipping.")
            continue
        
        orig_idx = orig_indices[0]
        baseline_emb = embeddings[orig_idx]
        baseline_norm = np.linalg.norm(baseline_emb)

        print(f"\nModel: {model_name}")
        print(f"  Baseline embedding index: {orig_idx} (Label: '{labels[orig_idx]}')")
        print(f"  Baseline L2 Norm: {baseline_norm:.4f}")

        # Compute displacements
        for i in range(len(factors)):
            emb = embeddings[i]
            diff = emb - baseline_emb
            abs_disp = np.linalg.norm(diff)
            rel_disp = abs_disp / baseline_norm if baseline_norm > 0 else 0.0

            comparison_data.append({
                'Model': model_name,
                'Factor': factors[i],
                'Label': labels[i],
                'Abs_Displacement': abs_disp,
                'Rel_Displacement': rel_disp * 100  # Convert to percentage
            })

    df = pd.DataFrame(comparison_data)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Absolute L2 Displacement (Left Plot)
    ax_abs = axes[0]
    # 2. Relative L2 Displacement (Right Plot)
    ax_rel = axes[1]

    for model_name in df['Model'].unique():
        model_df = df[df['Model'] == model_name].sort_values('Factor')
        color = colors.get(model_name, '#7f7f7f')
        marker = markers.get(model_name, 'o')

        # Decide x scale
        # If intensity variants, factors range from 1e-10 to 1000, so a log scale is best
        if perturbation_type == "Intensity":
            x_vals = model_df['Factor']
            ax_abs.set_xscale('log')
            ax_rel.set_xscale('log')
        else:
            x_vals = model_df['Factor']

        ax_abs.plot(
            x_vals, model_df['Abs_Displacement'],
            label=model_name, color=color, marker=marker,
            linewidth=2, markersize=7, alpha=0.9
        )
        
        ax_rel.plot(
            x_vals, model_df['Rel_Displacement'],
            label=model_name, color=color, marker=marker,
            linewidth=2, markersize=7, alpha=0.9
        )

    # Styling Absolute plot
    ax_abs.set_title("Absolute L2 Displacement from Baseline", fontsize=13, fontweight='bold', pad=12)
    ax_abs.set_xlabel(f"{perturbation_type} Multiplier", fontsize=11, fontweight='bold')
    ax_abs.set_ylabel("L2 Distance in Embedding Space", fontsize=11, fontweight='bold')
    ax_abs.grid(True, linestyle='--', alpha=0.5)
    ax_abs.spines['top'].set_visible(False)
    ax_abs.spines['right'].set_visible(False)

    # Styling Relative plot
    ax_rel.set_title("Relative L2 Displacement from Baseline (%)", fontsize=13, fontweight='bold', pad=12)
    ax_rel.set_xlabel(f"{perturbation_type} Multiplier", fontsize=11, fontweight='bold')
    ax_rel.set_ylabel("Relative Drift (% of Baseline Norm)", fontsize=11, fontweight='bold')
    ax_rel.grid(True, linestyle='--', alpha=0.5)
    ax_rel.spines['top'].set_visible(False)
    ax_rel.spines['right'].set_visible(False)

    # Add legends
    ax_abs.legend(fontsize=10, frameon=True)
    ax_rel.legend(fontsize=10, frameon=True)

    plt.suptitle(f"Embedding Space Robustness Analysis ({perturbation_type} - {patch_name})\n(Lower displacement = higher feature representations stability/invariance)",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    # Save comparison plot
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"l2_displacement_{input_path.stem}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to {save_path}")

    # Output markdown tables
    print(f"\n--- ABSOLUTE L2 DISPLACEMENT TABLE ({perturbation_type} - {patch_name}) ---")
    # Select a subset of factors for clean printing
    unique_factors = sorted(df['Factor'].unique())
    # Choose 6 representative factors to display in the console table
    if len(unique_factors) > 6:
        # Include baseline 1.0 always, plus some extremes
        display_factors = [f for f in unique_factors if np.isclose(f, 0.25) or np.isclose(f, 0.5) or np.isclose(f, 1.0) or np.isclose(f, 2.0) or np.isclose(f, 5.0) or np.isclose(f, 100.0) or np.isclose(f, 0.1)]
        if 1.0 not in display_factors:
            display_factors.append(1.0)
        display_factors = sorted(list(set(display_factors)))
    else:
        display_factors = unique_factors

    headers = " | ".join([f"Factor {f}" for f in display_factors])
    print(f"| Model | {headers} |")
    print(f"| :--- | " + " | ".join([":---:" for _ in display_factors]) + " |")
    
    for model_name in df['Model'].unique():
        row = f"| **{model_name}**"
        for f in display_factors:
            val = df[(df['Model'] == model_name) & (np.isclose(df['Factor'], f))]
            if len(val) > 0:
                row += f" | {val['Abs_Displacement'].values[0]:.4f}"
            else:
                row += " | -"
        row += " |"
        print(row)

    print(f"\n--- RELATIVE L2 DISPLACEMENT (%) TABLE ({perturbation_type} - {patch_name}) ---")
    print(f"| Model | {headers} |")
    print(f"| :--- | " + " | ".join([":---:" for _ in display_factors]) + " |")
    
    for model_name in df['Model'].unique():
        row = f"| **{model_name}**"
        for f in display_factors:
            val = df[(df['Model'] == model_name) & (np.isclose(df['Factor'], f))]
            if len(val) > 0:
                row += f" | {val['Rel_Displacement'].values[0]:.2f}%"
            else:
                row += " | -"
        row += " |"
        print(row)

if __name__ == "__main__":
    main()
