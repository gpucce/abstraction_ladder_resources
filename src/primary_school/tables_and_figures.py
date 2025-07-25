import os
import yaml

import pandas as pd
import transformers
import torch
from datasets import load_from_disk

from tqdm.auto import tqdm

from src.utils import tokenize_and_align_labels, exist_df

def get_result(rows, model, tokenizer, do_print=False):
    data_collator = transformers.default_data_collator
    model_inputs = data_collator(
        [{
            "input_ids": torch.tensor(rows["input_ids"][idx]),
            "attention_mask": torch.tensor(rows["attention_mask"][idx])
        }
         for idx in range(len(rows["input_ids"]))])

    all_tokens = []
    all_labels = []

    with torch.no_grad():
        logits = model(**{i: j.to("cuda") for i, j in model_inputs.items()}).logits.cpu()

    for idx, logit in enumerate(logits):
        preds = logit.argmax(-1).tolist()[1:]
        tokens = rows["full_list"][idx]

        refs = model_inputs["input_ids"][idx][model_inputs["attention_mask"][idx].bool()].tolist()
        check = []
        for tok in tokens:
            check += tokenizer(tok, add_special_tokens=False)["input_ids"]
        assert refs[1:-1] == check, f"Tokenization mismatch: {refs[1:-1]} != {check}"

        count = 0
        row_tokens = []
        row_labels = []
        for tok in tokens:
            tokenized = tokenizer(tok, add_special_tokens=False)["input_ids"]
            label = model.config.id2label[preds[count]]
            count += len(tokenized)
            row_tokens.append(tok)
            row_labels.append(label)
            if do_print:
                print("Token:", tok, "| Label:", label)

        all_tokens.append(row_tokens)
        all_labels.append(row_labels)

    rows["tokens"] = all_tokens
    rows["labels"] = all_labels

    return rows

def get_metrics(ds):
    total = 0
    metrics = {"n": 0, "p": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0, "List Acc": 0, "List Acc @1": 0, "List Acc @2": 0}
    for idx in tqdm(range(len(ds))):
        tokens = ds[idx]["tokens"]
        labels = ds[idx]["labels"]
        results = list(zip(tokens, labels))
        full_list = ds[idx]["full_list"]
        clean_results = ds[idx]["clean_list"]
        total += len(full_list)
        metrics["p"] += len(clean_results)
        metrics["n"] += len(full_list) - len(clean_results)
        metrics["tp"] += sum(1 for r in results if r[1] != "O" and r[0] in clean_results)
        metrics["tn"] += sum(1 for r in results if r[1] == "O" and r[0] not in clean_results)
        metrics["fn"] += sum(1 for r in results if r[1] == "O" and r[0] in clean_results)
        metrics["fp"] += sum(1 for r in results if r[1] != "O" and r[0] not in clean_results)
        matches = [i[0] for i in results if i[1] != "O"]
        if matches == clean_results:
            metrics["List Acc"] += 1
        if sum(1 for i in matches if i not in clean_results) + sum(1 for i in clean_results if i not in matches) <= 1:
            metrics["List Acc @1"] += 1
        if sum(1 for i in matches if i not in clean_results) + sum(1 for i in clean_results if i not in matches) <= 2:
            metrics["List Acc @2"] += 1

    assert total == (metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"])
    assert total == (metrics["p"] + metrics["n"])
    assert metrics["p"] == (metrics["tp"] + metrics["fn"])
    assert metrics["n"] == (metrics["tn"] + metrics["fp"])

    metrics["precision"] = metrics["tp"] / (metrics["tp"] + metrics["fp"])
    metrics["recall"] = metrics["tp"] / (metrics["tp"] + metrics["fn"])
    metrics["f1"] = (2 * metrics["tp"]) / (2 * metrics["tp"] + metrics["fp"] + metrics["fn"])
    metrics["accuracy"] = (metrics["tp"] + metrics["tn"]) / (metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"])
    metrics["list accuracy"] = metrics["List Acc"] / len(ds)
    metrics["list accuracy @1"] = metrics["List Acc @1"] / len(ds)
    metrics["list accuracy @2"] = metrics["List Acc @2"] / len(ds)
    metrics["total lists"] = len(ds)

    return metrics


def run_eval(experiment_name, dataset_name, run_path, file_dir_path):
    if exist_df(os.path.join(file_dir_path, f"{experiment_name}_results.ds")):
        ds = load_from_disk(os.path.join(file_dir_path, f"{experiment_name}_results.ds"))
    else:
        model_name_or_path = None
        runs = os.listdir(run_path)
        for run in runs:
            with open(os.path.join(run_path, run, ".hydra", "config.yaml")) as f:
                config = yaml.safe_load(f)
            if config["args"]["dataset_name"].split('/')[-1] == dataset_name.split('/')[-1]:
                model_name_or_path = os.path.join(run_path, run)
                break

        assert model_name_or_path is not None, f"Model not found for dataset {dataset_name}"

        model = transformers.AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,trust_remote_code=True,)
        model.to("cuda")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True,)

        ds = load_from_disk(dataset_name)
        ref_ds = load_from_disk(dataset_name)
        if "validation" in ds:
            ds = ds["validation"]
            ref_ds = ref_ds["validation"]

        ds = ds.map(lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id=model.config.label2id), batched=True, desc="Tokenizing and aligning")
        ds = ds.map(lambda x: get_result(x, model, tokenizer), batched=True, batch_size=10, desc="Getting results")
        ds.save_to_disk(os.path.join(file_dir_path, f"{experiment_name}_results.ds"))

    return get_metrics(ds)

def main():
    import sys

    rerun = len(sys.argv) > 1 and sys.argv[1] == "rerun"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    run_path = "./multirun/2025-07-25/14-11-04"

    file_dir_path = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(file_dir_path, exist_ok=True)
    if not exist_df(os.path.join(file_dir_path, "primary_school_results.csv")) or rerun:
        experiments_data_map = {
            "all_with_spec": "./src/primary_school/training_datasets/training_data_all_with_spec.ds",
            "t1_vs_t2_with_spec": "./src/primary_school/training_datasets/training_t1_test_t2_with_spec.ds",
            "t2_vs_t1_with_spec": "./src/primary_school/training_datasets/training_t2_test_t1_with_spec.ds",
            "all_without_spec": "./src/primary_school/training_datasets/training_data_all_no_spec.ds",
            "t1_vs_t2_without_spec": "./src/primary_school/training_datasets/training_t1_test_t2_no_spec.ds",
            "t2_vs_t1_without_spec": "./src/primary_school/training_datasets/training_t2_test_t1_no_spec.ds",
        }

        all_metrics = {}
        for experiment_name, dataset_name in experiments_data_map.items():
            print(f"Running evaluation for {experiment_name} with dataset {dataset_name}")
            all_metrics[experiment_name] = run_eval(experiment_name, dataset_name, run_path, file_dir_path)
            print(all_metrics[experiment_name])

        df = pd.DataFrame.from_dict(all_metrics, orient="index")
        df.to_csv(os.path.join(file_dir_path, "primary_school_results.csv"))
    else:
        df = pd.read_csv(os.path.join(file_dir_path, "primary_school_results.csv"), index_col=0)

    # Rename for better LaTeX readability
    df = df.reset_index().rename({"index": "Setting"}, axis=1)
    df = df.rename(columns={
        "name": "Setting",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
        "accuracy": "Accuracy",
        "list accuracy": "List Acc",
        "list accuracy @1": "List Acc@1",
        "list accuracy @2": "List Acc@2"
    })

    # Select only the desired columns
    latex_df = df[["Setting", "Precision", "Recall", "F1-score", "Accuracy", "List Acc", "List Acc@1", "List Acc@2"]]

    # Round metrics for better formatting
    latex_df = latex_df.round(3)
    latex_df["Setting"] = latex_df["Setting"].str.replace("_", " ").str.title()

    # Transpose the DataFrame for LaTeX formatting
    # latex_df = latex_df.T

    # Generate LaTeX table
    latex_code = latex_df.to_latex(
        index=False, float_format="%.3f", column_format='lcccccccc', escape=False)

    with open(os.path.join(file_dir_path, "primary_school_metrics.tex"), "w") as f:
        f.write(latex_code)


if __name__ == "__main__":
    main()
