#!/bin/bash

#SBATCH --job-name=scVI_700k
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:a4000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=48:00:00

# --- LOGGING PATHS ---
# Standard output (the progress and prints)
#SBATCH --output=/home/melrie/human_intestinal/logs/scVI_700k_%j.log

# Error log (tracebacks and warnings)
#SBATCH --error=/home/melrie/human_intestinal/logs/scVI_700k_%j.err

# data path, export so python can access it
export DATA_PATH="/home/melrie/human_intestinal/data/adata_clean.h5ad"

# Activate the conda environment
source /home/melrie/miniforge3/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"

mamba activate hi


# 2. Run python code
# We use << 'EOF' to prevent Bash from trying to interpret Python variables (like adata.obs)
python - << 'EOF'
import scanpy as sc
import scvi
import torch
import os
import matplotlib.pyplot as plt
import time
import pandas as pd

# Hardware Setup
torch.set_float32_matmul_precision('high')

# We pull the path from the Bash variable
input_file = os.environ.get('DATA_PATH')
job_id = os.environ.get('SLURM_JOB_ID', 'standalone')

config = {
    "batch_size": 1024,
    "max_epochs": 1000,
    "early_stopping": True,
    "batch_keys": 'sampleID' #sampleID,assay,study,donor
}

# Data Loading
print("Loading data...")
adata = sc.read_h5ad(input_file)
adata.layers['counts'] = adata.raw.X.copy()


# 2. Dynamic Logging Header
print("="*40)
print(f"RUN START: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'N/A')}")
print(f"Input file: {os.path.basename(input_file)}")
print(f"Cells/Genes: {adata.n_obs} x {adata.n_vars}")
print("-" * 20)
print("HYPERPARAMETERS:")
for key, value in config.items():
    print(f"  {key}: {value}")
print(f"  GPU Type: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("="*40)

# 3. Setup using the config
scvi.model.SCVI.setup_anndata(
    adata, 
    layer='counts', 
    batch_key=config["batch_keys"]
)

model = scvi.model.SCVI(adata) 


# 4. Train using the config
model.train(
    max_epochs=config["max_epochs"],
    batch_size=config["batch_size"],
    early_stopping=config["early_stopping"],
    enable_progress_bar=True, #testing if it works, can be changed to True for better monitoring
    accelerator='gpu', 
    devices=1,
    datasplitter_kwargs={
        'num_workers': 4,
        'pin_memory': True,
        'persistent_workers': True
    }
)


# Calculate iterations
n_obs = model.adata.n_obs
batch_size = config["batch_size"]
# Total iterations per epoch
steps_per_epoch = n_obs // batch_size + (1 if n_obs % batch_size > 0 else 0)

# Get actual epochs run (in case early stopping kicked in)
actual_epochs = len(model.history["elbo_train"])

total_iterations = actual_epochs * steps_per_epoch

print(f"--- Training Summary ---")
print(f"Final Epochs: {actual_epochs}")
print(f"Steps per Epoch: {steps_per_epoch}")
print(f"Total Iterations: {total_iterations}")

# --- SAVING OUTPUTS ---
job_id = os.environ.get('SLURM_JOB_ID', 'standalone')
save_path = "/home/melrie/human_intestinal/outputs"

os.makedirs(os.path.join(save_path, 'models'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'figures'), exist_ok=True)

# Save Model
model_dir = os.path.join(save_path, f'models/scvi_model_{job_id}')
model.save(model_dir, overwrite=True)

# Extract Latents and write to a small file
latent_rep = model.get_latent_representation()
latent_df = pd.DataFrame(latent_rep, index=adata.obs_names)
latent_df.to_pickle(os.path.join(save_path, f'models/latents_{job_id}.pkl'))

# Plotting Metrics
history = model.history
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics = ['elbo', 'reconstruction_loss', 'kl_local']
for i, m in enumerate(metrics):
    history[f'{m}_train'].plot(ax=axes[i], label='train')
    if f'{m}_validation' in history:
        history[f'{m}_validation'].plot(ax=axes[i], label='validation')
    axes[i].set_title(m.replace('_', ' ').upper())
    axes[i].legend()

plt.tight_layout()
plt.savefig(f"{save_path}/figures/model_metrics_job_{job_id}.png", dpi=300)
plt.close()

print(f'Integration complete. Files saved with Job ID: {job_id}')
EOF