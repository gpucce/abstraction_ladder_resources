# Brownie IS-A cake, but sugar is not: Can encoders annotate and clean children’s (mis)built semantic hierarchies?

Code to replicate the experiment in the paper:

First create a conda environment:

`conda create -p conda_venv python=3.11`

To replicate all experiments it should be enough to run the following


```bash
conda activate conda_venv/
python -m src.primary_school.prepare_data
bash run_hydra.sh
python -m src.primary_school.tables_and_figures
```

Afterwards the `plotting.ipynb` should create all the figures in the paper, you might need to create the `plots` and `tables` folder by hand 

```bash
mkdir plots
mkdir tables
```