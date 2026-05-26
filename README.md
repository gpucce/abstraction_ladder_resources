# Brownie IS-A cake, but sugar is not: Can encoders annotate and clean children’s (mis)built semantic hierarchies?

Code to replicate the experiment in the paper:

First create a conda environment:

```bash
conda create -p conda_venv python=3.11
conda activate ./conda_venv
```

To replicate all experiments it should be enough to run the following:

 - Prepare data:
```bash
    cd src/primary_school/training_datasets
    python convert_to_hf.py
```
 - Run experiments:
```bash
# back in the main directory run the various experiments
bash run_hydra_primary_school.sh
bash run_hydra_primary_school_alberto.sh
bash run_hydra_primary_school_modern_bert.sh
bash run_hydra_primary_school_roberta.sh
bash run_hydra_primary_school_xlm_roberta.sh
python -m src.primary_school.tables_and_figures
```

Afterwards the `src/primary_school/paper_plots/plotting.ipynb` should create all the figures in the paper, you might need to create the `plots` and `tables` folder by hand in the `src/primary_school/paper_plots` directory.
