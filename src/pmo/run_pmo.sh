#!/bin/bash
#SBATCH --job-name=pmo-seed0
#SBATCH --error=log/job_pmo_seed0_error.txt
#SBATCH --output=log/job_pmo_seed0_output.txt
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100l:1
#SBATCH --partition=unkillable

module load python/3.10
module load cuda/12.4.1/cudnn  # Match your PyTorch version

# Activate venv (created *after* loading the above python module)
source ~/scratch/envs/rxn121/bin/activate


oracle_array=(
    "qed"
    "gsk3b"
    "drd2"
    "jnk3"
    "celecoxib_rediscovery"
    "troglitazone_rediscovery"
    "thiothixene_rediscovery"
    "albuterol_similarity"
    "mestranol_similarity"
    "isomers_c7h8n2o2"
    "isomers_c9h10n2o2pf2cl"
    "median1"
    "median2"
    "osimertinib_mpo"
    "fexofenadine_mpo"
    "ranolazine_mpo"
    "perindopril_mpo"
    "amlodipine_mpo"
    "sitagliptin_mpo"
    "zaleplon_mpo"
    "valsartan_smarts"
    "deco_hop"
    "scaffold_hop"
)


for oralce in "${oracle_array[@]}"
do
python run.py smiles_mle --oracles $oralce --wandb online --run_name quant_mle_ga --config_default hparams_mle_ga25.yaml --seed 0
done

