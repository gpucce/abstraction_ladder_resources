CUDA_VISIBLE_DEVICES=2 python -m src.primary_school.experiments -m hydra.job.chdir=True model=xlm_roberta model.learning_rate=3.e-5 args.seed=44
