import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Set design styles for premium look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# Set project paths
PROJECT_ROOT = Path('/nfs/erelab001/shared/Computational_Group/Naravich/nnUNet')
output_dir = Path("/home/naravich/nnunet_texture_ablations")
scratch_dir = PROJECT_ROOT / "scratch"
scratch_dir.mkdir(exist_ok=True)

# Load all cases data from the new folder structure
models = ['No_Pretrain', 'CLIP', 'Genesis', 'LaW']
rows = []

print("Scanning output directories for evaluation results...")
for model_name in models:
    model_dir = output_dir / model_name
    if not model_dir.exists():
        print(f"Directory {model_dir} does not exist. Skipping.")
        continue
        
    for fold_dir in model_dir.glob("fold_*"):
        try:
            fold = int(fold_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
            
        dice_scores_dir = fold_dir / "dice_scores"
        if not dice_scores_dir.exists():
            continue
            
        for json_file in dice_scores_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
                continue
                
            case_id = data.get('case_id', json_file.stem)
            
            # Texture factors & scores
            tex_factors = data.get('texture_factors', [])
            tex_scores = data.get('texture_dice_scores', [])
            for factor, score in zip(tex_factors, tex_scores):
                rows.append({
                    'Model': model_name,
                    'Fold': fold,
                    'Case': case_id,
                    'Type': 'Texture',
                    'Factor': factor,
                    'Dice': score
                })
                
            # Intensity factors & scores
            int_factors = data.get('intensity_factors', [])
            int_scores = data.get('intensity_dice_scores', [])
            for factor, score in zip(int_factors, int_scores):
                rows.append({
                    'Model': model_name,
                    'Fold': fold,
                    'Case': case_id,
                    'Type': 'Intensity',
                    'Factor': factor,
                    'Dice': score
                })

if not rows:
    print("Warning: No JSON files found in evaluation directories. Creating empty DataFrame.")
    df = pd.DataFrame(columns=['Model', 'Fold', 'Case', 'Type', 'Factor', 'Dice'])
else:
    df = pd.DataFrame(rows)
    print(f"Successfully aggregated {len(df)} records.")

# Save raw summarized data
raw_csv_path = scratch_dir / 'summarized_texture_ablations.csv'
df.to_csv(raw_csv_path, index=False)
print(f"Saved aggregated results to {raw_csv_path}")

if df.empty:
    print("No data available to compute metrics or plot. Exiting.")
    sys.exit(0)

# Find the baseline (factor 1.0) Dice score for each Model, Fold, and Case
baselines = df[np.isclose(df['Factor'], 1.0)][['Model', 'Fold', 'Case', 'Type', 'Dice']].rename(columns={'Dice': 'Baseline_Dice'})

# Extract baseline from Type='Texture' at Factor=1.0 for each Model, Fold, and Case
baselines_unique = baselines[baselines['Type'] == 'Texture'][['Model', 'Fold', 'Case', 'Baseline_Dice']]

# Merge baseline back into main df matching Model, Fold, and Case
df_with_baseline = pd.merge(df, baselines_unique, on=['Model', 'Fold', 'Case'], how='left')

# Calculate percentage decrease: ((Baseline - Dice) / Baseline) * 100
df_with_baseline['Pct_Decrease'] = ((df_with_baseline['Baseline_Dice'] - df_with_baseline['Dice']) / df_with_baseline['Baseline_Dice']) * 100

# Compute the average and std of Pct_Decrease over all cases
avg_list = []
grouped = df_with_baseline.groupby(['Model', 'Type', 'Factor'])
for (model, t_type, factor), group in grouped:
    mean_pct_dec = group['Pct_Decrease'].mean()
    std_pct_dec = group['Pct_Decrease'].std()
    mean_dice = group['Dice'].mean()
    n_cases = len(group)  # Dynamic number of cases for standard error calculation
    avg_list.append({
        'Model': model,
        'Type': t_type,
        'Factor': factor,
        'Dice_Mean': mean_dice,
        'Pct_Decrease_Mean': mean_pct_dec,
        'Pct_Decrease_Std': std_pct_dec,
        'N_Cases': n_cases
    })

df_avg = pd.DataFrame(avg_list)
pct_dec_csv_path = scratch_dir / 'summarized_texture_ablations_pct_decrease.csv'
df_avg.to_csv(pct_dec_csv_path, index=False)
print(f"Saved percentage decrease results to {pct_dec_csv_path}")

# Unique configurations
colors = {
    'LaW': '#ff7f0e',        # Orange
    'Genesis': '#d62728',    # Red
    'CLIP': '#e377c2',       # Pink/Purple
    'No_Pretrain': '#17becf'  # Cyan
}
markers = {
    'LaW': 'o',
    'Genesis': 's',
    'CLIP': '^',
    'No_Pretrain': 'd'
}

# Create a 1x2 grid plot (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# 1. Texture Ablations (Left Plot)
ax_tex = axes[0]
data_tex = df_avg[df_avg['Type'] == 'Texture']

for model in models:
    model_data = data_tex[data_tex['Model'] == model].sort_values('Factor')
    if model_data.empty:
        continue
    ax_tex.plot(
        model_data['Factor'], 
        model_data['Pct_Decrease_Mean'], 
        label=model, 
        color=colors[model], 
        marker=markers[model], 
        linewidth=2.5, 
        markersize=8,
        alpha=0.9
    )
    # Add shaded standard error region dynamically
    sem = model_data['Pct_Decrease_Std'] / np.sqrt(model_data['N_Cases'])
    ax_tex.fill_between(
        model_data['Factor'],
        model_data['Pct_Decrease_Mean'] - sem,
        model_data['Pct_Decrease_Mean'] + sem,
        color=colors[model],
        alpha=0.1
    )
    
ax_tex.set_title("Mean % Decrease vs. Texture Multiplier", fontsize=13, fontweight='bold', pad=12)
ax_tex.set_xlabel("Texture Multiplier", fontsize=11, fontweight='bold')
ax_tex.set_ylabel("Mean Percentage Decrease (%)", fontsize=11, fontweight='bold')
ax_tex.set_ylim(-5, 105)
ax_tex.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
ax_tex.grid(True, linestyle='--', alpha=0.5)
ax_tex.spines['top'].set_visible(False)
ax_tex.spines['right'].set_visible(False)

# 2. Intensity Ablations (Right Plot)
ax_int = axes[1]
data_int = df_avg[df_avg['Type'] == 'Intensity']

for model in models:
    model_data = data_int[data_int['Model'] == model].sort_values('Factor')
    if model_data.empty:
        continue
    ax_int.plot(
        model_data['Factor'], 
        model_data['Pct_Decrease_Mean'], 
        label=model, 
        color=colors[model], 
        marker=markers[model], 
        linewidth=2.5, 
        markersize=8,
        alpha=0.9
    )
    # Add shaded standard error region dynamically
    sem = model_data['Pct_Decrease_Std'] / np.sqrt(model_data['N_Cases'])
    ax_int.fill_between(
        model_data['Factor'],
        model_data['Pct_Decrease_Mean'] - sem,
        model_data['Pct_Decrease_Mean'] + sem,
        color=colors[model],
        alpha=0.1
    )
    
ax_int.set_title("Mean % Decrease vs. Intensity Multiplier", fontsize=13, fontweight='bold', pad=12)
ax_int.set_xlabel("Intensity Multiplier", fontsize=11, fontweight='bold')
ax_int.set_ylim(-5, 105)
ax_int.set_xticks([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
ax_int.grid(True, linestyle='--', alpha=0.5)
ax_int.spines['top'].set_visible(False)
ax_int.spines['right'].set_visible(False)

# Collect legends and place nicely
handles, labels = ax_tex.get_legend_handles_labels()
fig.legend(
    handles, 
    labels, 
    loc='lower center', 
    ncol=4, 
    fontsize=12, 
    frameon=True, 
    facecolor='#F8F9FA', 
    edgecolor='#E5E5E5',
    bbox_to_anchor=(0.5, -0.05)
)

plt.suptitle("Model Performance Degradation: Mean Percentage Decrease from Baseline (1.0)\n(Shaded area indicates standard error of the mean; 0% = no change, 100% = complete failure)", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

# Save comparison plot
save_path = output_dir / "texture_ablations_pct_decrease.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to {save_path}")

# Output markdown tables
print("\n--- TEXTURE % DECREASE AVERAGE TABLE ---")
print("| Model | Factor 0.25 | Factor 0.50 | Factor 1.0 (Base) | Factor 1.50 | Factor 2.0 |")
print("| :--- | :---: | :---: | :---: | :---: | :---: |")
for model in models:
    row = f"| **{model}**"
    for f in [0.25, 0.50, 1.0, 1.50, 2.0]:
        val = data_tex[(data_tex['Model'] == model) & (np.isclose(data_tex['Factor'], f))]
        if len(val) > 0:
            mean = val['Pct_Decrease_Mean'].values[0]
            std = val['Pct_Decrease_Std'].values[0]
            std_str = f"{std:.2f}%" if not np.isnan(std) else "0.00%"
            row += f" | {mean:.2f}% ± {std_str}"
        else:
            row += " | -"
    row += " |"
    print(row)

print("\n--- INTENSITY % DECREASE AVERAGE TABLE ---")
print("| Model | Factor 0.25 | Factor 0.50 | Factor 1.0 (Base) | Factor 1.50 | Factor 2.0 |")
print("| :--- | :---: | :---: | :---: | :---: | :---: |")
for model in models:
    row = f"| **{model}**"
    for f in [0.25, 0.50, 1.0, 1.50, 2.0]:
        val = data_int[(data_int['Model'] == model) & (np.isclose(data_int['Factor'], f))]
        if len(val) > 0:
            mean = val['Pct_Decrease_Mean'].values[0]
            std = val['Pct_Decrease_Std'].values[0]
            std_str = f"{std:.2f}%" if not np.isnan(std) else "0.00%"
            row += f" | {mean:.2f}% ± {std_str}"
        else:
            row += " | -"
    row += " |"
    print(row)
