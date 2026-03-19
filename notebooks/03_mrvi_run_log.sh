#!/bin/bash

#SBATCH --job-name=mrVI_700k
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:a4000:1     # Matches your slurm.conf/gres.conf 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=48:00:00

# --- LOGGING PATHS ---
# Standard output (the progress and prints)
#SBATCH --output=/home/melrie/human_intestinal/logs/mrVI_700k_%j.log

# Error log (tracebacks and warnings)
#SBATCH --error=/home/melrie/human_intestinal/logs/mrVI_700k_%j.err

# Input data path, export so python can access it
export DATA_PATH="/home/melrie/human_intestinal/data/adata_clean.h5ad"

# Activate the conda environment
source /home/melrie/miniforge3/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"

mamba activate hi


# 2. Run the Python code
# We use << 'EOF' to prevent Bash from trying to interpret Python variables (like adata.obs)
python - << 'EOF'
import scanpy as sc
import scvi
import torch
import os
import matplotlib.pyplot as plt
import time
from scvi.external import MRVI
import pandas as pd

# Hardware Setup
torch.set_float32_matmul_precision('high')

# 1. Configuration
# We pull the path from the Bash variable
input_file = os.environ.get('DATA_PATH')
job_id = os.environ.get('SLURM_JOB_ID', 'standalone')

config = {
    "input_path": input_file,
    "batch_size": 1024,
    "max_epochs": 1000,
    "sample_key": 'donor_disease_category',
    "early_stopping": True,
    "batch_key": 'sampleID'
}

# 2. Data Loading
print("Loading data...")
adata = sc.read_h5ad(config["input_path"])
adata.layers['counts'] = adata.raw.X.copy()


# 2. Dynamic Logging Header
print("="*40)
print(f"RUN START: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'N/A')}")
print(f"Input file: {os.path.basename(config['input_path'])}")
print(f"Cells/Genes: {adata.n_obs} x {adata.n_vars}")
print("-" * 20)
print("HYPERPARAMETERS:")
for key, value in config.items():
    print(f"  {key}: {value}")
print(f"  GPU Type: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("="*40)

sample_key = config["sample_key"]  # target covariate
batch_key=config["batch_key"]  # nuisance variable identifier
MRVI.setup_anndata(adata, layer="counts", sample_key=sample_key, batch_key=batch_key, backend="torch")
model = MRVI(adata)
 

#start = time.time()

# 4. Train using the config
model.train(
    max_epochs=config["max_epochs"],
    batch_size=config["batch_size"],
    early_stopping=config["early_stopping"],
    enable_progress_bar=True, #testing if it works, can be changed to True for better monitoring
    accelerator='gpu', 
    devices=1,
    datasplitter_kwargs={
        'num_workers': 8,
        'pin_memory': True,
        'persistent_workers': True
    }
)

#end = time.time()
#print(f'Training completed in {(end - start)/60:.2f} minutes.')

# --- SAVING OUTPUTS ---
job_id = os.environ.get('SLURM_JOB_ID', 'standalone')
save_path = "/home/melrie/human_intestinal/outputs"

os.makedirs(os.path.join(save_path, 'models'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'figures'), exist_ok=True)

# Save Model
model_dir = os.path.join(save_path, f'models/mrvi_model_{job_id}')
model.save(model_dir, overwrite=True)

# 3. VERIFY (The 'Safety' way)
# If you don't see these 3 files, the save failed!
import os
print(f"Files saved: {os.listdir(model_dir)}")

# --- EXTRACT AND SAVE LATENTS ---
print("Extracting latents...")
# Extract u (Sample-corrected)
adata.obsm["X_mrVI_u"] = model.get_latent_representation()

# Extract z (Encoder output)
adata.obsm["X_mrVI_z"] = model.get_latent_representation(give_z=True)

# Cleaned up path
final_adata_path = f'/home/melrie/human_intestinal/data/adata_mrvi_{job_id}.h5ad'

print(f"Saving AnnData to {final_adata_path}...")
adata.write_h5ad(final_adata_path)

print(f"Latents saved into AnnData at: {final_adata_path}")

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

print(adata)

print(f'Integration complete. Files saved with Job ID: {job_id}')
EOF

echo "----------------------------------------------------"
echo "JOB FINISHED SUCCESSFULLY"
echo "Date: $(date)"
echo "Check outputs at:"
echo "Model Directory: /home/melrie/human_intestinal/outputs/models/mrvi_model_${SLURM_JOB_ID}"
echo "Integrated data:   /home/melrie/human_intestinal/data/adata_mrvi_${SLURM_JOB_ID}.h5ad"
echo "Metrics Plot:    /home/melrie/human_intestinal/outputs/figures/model_metrics_job_${SLURM_JOB_ID}.png"
echo "----------------------------------------------------"