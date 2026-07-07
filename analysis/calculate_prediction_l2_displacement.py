import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Set project paths
PROJECT_ROOT = Path('/nfs/erelab001/shared/Computational_Group/Naravich/nnUNet')
output_dir = Path("/home/naravich/nnunet_texture_ablations")
scratch_dir = PROJECT_ROOT / "scratch"
scratch_dir.mkdir(exist_ok=True)

# Set design styles for premium look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

models = ['No_Pretrain', 'CLIP', 'Genesis', 'LaW']
texture_factors = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
intensity_factors = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

rows = []

print("Scanning output directories for prediction files...")
for model_name in models:
    model_dir = output_dir / model_name
    if not model_dir.exists():
        continue
        
    for fold_dir in model_dir.glob("fold_*"):
        try:
            fold = int(fold_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
            
        predictions_dir = fold_dir / "predictions"
        if not predictions_dir.exists():
            continue
            
        for pth_file in predictions_dir.glob("*.pth"):
            print(f"Loading predictions for Model: {model_name}, Fold: {fold}, Case: {pth_file.stem}...")
            try:
                data = torch.load(pth_file)
            except Exception as e:
                print(f"Error loading {pth_file}: {e}")
                continue
                
            case_id = data.get('case_id', pth_file.stem)
            
            # Extract predictions
            tex_preds = data.get('texture_predictions', [])
            int_preds = data.get('intensity_predictions', [])
            
            # We must have predictions for all factors
            if len(tex_preds) != len(texture_factors) or len(int_preds) != len(intensity_factors):
                print(f"  Warning: Expected {len(texture_factors)} texture and {len(intensity_factors)} intensity predictions. Got {len(tex_preds)} and {len(int_preds)}. Skipping.")
                continue
                
            # Baseline predictions are at factor 1.0
            # For texture, index 4 corresponds to factor 1.0
            baseline_tex = tex_preds[4].astype(np.float32)
            # For intensity, index 3 corresponds to factor 1.0
            baseline_int = int_preds[3].astype(np.float32)
            
            norm_baseline_tex = np.linalg.norm(baseline_tex)
            norm_baseline_int = np.linalg.norm(baseline_int)
            
            # Compute L2 displacement for texture
            for i, factor in enumerate(texture_factors):
                pred = tex_preds[i].astype(np.float32)
                diff = pred - baseline_tex
                abs_disp = np.linalg.norm(diff)
                rel_disp = abs_disp / norm_baseline_tex if norm_baseline_tex > 0 else (0.0 if abs_disp == 0 else 1.0)
                
                rows.append({
                    'Model': model_name,
                    'Fold': fold,
                    'Case': case_id,
                    'Type': 'Texture',
                    'Factor': factor,
                    'Abs_Displacement': abs_disp,
                    'Rel_Displacement': rel_disp * 100
                })
                
            # Compute L2 displacement for intensity
            for i, factor in enumerate(intensity_factors):
                pred = int_preds[i].astype(np.float32)
                diff = pred - baseline_int
                abs_disp = np.linalg.norm(diff)
                rel_disp = abs_disp / norm_baseline_int if norm_baseline_int > 0 else (0.0 if abs_disp == 0 else 1.0)
                
                rows.append({
                    'Model': model_name,
                    'Fold': fold,
                    'Case': case_id,
                    'Type': 'Intensity',
                    'Factor': factor,
                    'Abs_Displacement': abs_disp,
                    'Rel_Displacement': rel_disp * 100
                })

if not rows:
    print("Error: No prediction files processed. Exiting.")
    sys.exit(1)

df = pd.DataFrame(rows)
df.to_csv(scratch_dir / 'summarized_prediction_l2_displacement.csv', index=False)
print(f"Saved raw prediction displacements to {scratch_dir / 'summarized_prediction_l2_displacement.csv'}")

# Aggregate metrics
avg_list = []
grouped = df.groupby(['Model', 'Type', 'Factor'])
for (model, t_type, factor), group in grouped:
    mean_abs = group['Abs_Displacement'].mean()
    std_abs = group['Abs_Displacement'].std()
    mean_rel = group['Rel_Displacement'].mean()
    std_rel = group['Rel_Displacement'].std()
    n_cases = len(group)
    avg_list.append({
        'Model': model,
        'Type': t_type,
        'Factor': factor,
        'Abs_Mean': mean_abs,
        'Abs_Std': std_abs,
        'Rel_Mean': mean_rel,
        'Rel_Std': std_rel,
        'N_Cases': n_cases
    })
df_avg = pd.DataFrame(avg_list)
df_avg.to_csv(scratch_dir / 'summarized_prediction_l2_displacement_avg.csv', index=False)

# Plot Relative L2 Displacements
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
colors = {
    'LaW': '#ff7f0e',
    'Genesis': '#d62728',
    'CLIP': '#e377c2',
    'No_Pretrain': '#17becf'
}
markers = {
    'LaW': 'o',
    'Genesis': 's',
    'CLIP': '^',
    'No_Pretrain': 'd'
}

# Left Plot: Texture
ax_tex = axes[0]
data_tex = df_avg[df_avg['Type'] == 'Texture']
for model in models:
    m_data = data_tex[data_tex['Model'] == model].sort_values('Factor')
    if m_data.empty:
        continue
    ax_tex.plot(m_data['Factor'], m_data['Rel_Mean'], label=model, color=colors[model], marker=markers[model], linewidth=2.5, markersize=8)
    sem = m_data['Rel_Std'] / np.sqrt(m_data['N_Cases'])
    ax_tex.fill_between(m_data['Factor'], m_data['Rel_Mean'] - sem, m_data['Rel_Mean'] + sem, color=colors[model], alpha=0.1)

ax_tex.set_title("Prediction Rel. L2 Displacement vs. Texture Multiplier", fontsize=12, fontweight='bold')
ax_tex.set_xlabel("Texture Multiplier", fontsize=11, fontweight='bold')
ax_tex.set_ylabel("Relative Prediction Drift (%)", fontsize=11, fontweight='bold')
ax_tex.set_ylim(-5, 105)
ax_tex.set_xticks(texture_factors)
ax_tex.grid(True, linestyle='--', alpha=0.5)

# Right Plot: Intensity
ax_int = axes[1]
data_int = df_avg[df_avg['Type'] == 'Intensity']
for model in models:
    m_data = data_int[data_int['Model'] == model].sort_values('Factor')
    if m_data.empty:
        continue
    ax_int.plot(m_data['Factor'], m_data['Rel_Mean'], label=model, color=colors[model], marker=markers[model], linewidth=2.5, markersize=8)
    sem = m_data['Rel_Std'] / np.sqrt(m_data['N_Cases'])
    ax_int.fill_between(m_data['Factor'], m_data['Rel_Mean'] - sem, m_data['Rel_Mean'] + sem, color=colors[model], alpha=0.1)

ax_int.set_title("Prediction Rel. L2 Displacement vs. Intensity Multiplier", fontsize=12, fontweight='bold')
ax_int.set_xlabel("Intensity Multiplier", fontsize=11, fontweight='bold')
ax_int.set_xticks(intensity_factors)
ax_int.grid(True, linestyle='--', alpha=0.5)

handles, labels = ax_tex.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.05))
plt.suptitle("Prediction Invariance Analysis: Relative L2 Displacement of Segmentations\n(Average over all 5 folds / 8 cases; Shaded area indicates standard error; 0% = prediction identical to baseline)", fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()

save_path = output_dir / "prediction_l2_displacement.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved prediction displacements plot to {save_path}")

# Print Tables
print("\n--- TEXTURE PREDICTION RELATIVE L2 DISPLACEMENT TABLE ---")
print("| Model | Factor 0.25 | Factor 0.50 | Factor 1.0 (Base) | Factor 1.50 | Factor 2.0 |")
print("| :--- | :---: | :---: | :---: | :---: | :---: |")
for model in models:
    row = f"| **{model}**"
    for f in [0.25, 0.50, 1.0, 1.50, 2.0]:
        val = data_tex[(data_tex['Model'] == model) & (np.isclose(data_tex['Factor'], f))]
        if len(val) > 0:
            mean = val['Rel_Mean'].values[0]
            std = val['Rel_Std'].values[0]
            std_str = f"{std:.2f}%" if not np.isnan(std) else "0.00%"
            row += f" | {mean:.2f}% ± {std_str}"
        else:
            row += " | -"
    row += " |"
    print(row)

print("\n--- INTENSITY PREDICTION RELATIVE L2 DISPLACEMENT TABLE ---")
print("| Model | Factor 0.25 | Factor 0.50 | Factor 1.0 (Base) | Factor 1.50 | Factor 2.0 |")
print("| :--- | :---: | :---: | :---: | :---: | :---: |")
for model in models:
    row = f"| **{model}**"
    for f in [0.25, 0.50, 1.0, 1.50, 2.0]:
        val = data_int[(data_int['Model'] == model) & (np.isclose(data_int['Factor'], f))]
        if len(val) > 0:
            mean = val['Rel_Mean'].values[0]
            std = val['Rel_Std'].values[0]
            std_str = f"{std:.2f}%" if not np.isnan(std) else "0.00%"
            row += f" | {mean:.2f}% ± {std_str}"
        else:
            row += " | -"
    row += " |"
    print(row)
