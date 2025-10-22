import os
import yaml
import spacy
import pandas as pd
import copy
import transformers
import torch
from datasets import load_from_disk

from tqdm.auto import tqdm
from nltk.corpus import wordnet as wn

from src.utils import tokenize_and_align_labels, exist_df

def is_direct_hyponym_ita(w1: str, w2: str, pos: str | None = 'n'):
    """
    Return True if any synset of w1 (Italian) is a *direct* hyponym of
    any synset of w2 (Italian). Only one edge down the hierarchy.
    """
    s1s = wn.synsets(w1, lang='ita', pos=pos)
    s2s = wn.synsets(w2, lang='ita', pos=pos)

    for s1 in s1s:
        for hyper in s1.hypernyms():  # direct parents only
            if hyper in s2s:
                return True, (s1, hyper)
    return False, None

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
    all_matches = []
    all_non_matches = []

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

        row_matches = []
        row_non_matches = []
        assert len(row_tokens) == len(row_labels)
        for idx, (tok, label) in enumerate(zip(row_tokens, row_labels)):
            if label != "O":
                row_matches.append(tok)
            else:
                row_non_matches.append(tok)

        all_tokens.append(row_tokens)
        all_labels.append(row_labels)
        all_matches.append(row_matches)
        all_non_matches.append(row_non_matches)

    rows["tokens"] = all_tokens
    rows["labels"] = all_labels
    rows["matches"] = all_matches
    rows["non_matches"] = all_non_matches

    return rows

def get_adj_and_wordnet_results(rows):
    nlp = spacy.load("it_core_news_sm")

    all_adj_labels = []
    all_wordnet_labels = []
    all_adj_matches = []
    all_adj_non_matches = []
    all_wordnet_matches = []
    all_wordnet_non_matches = []
    for idx in range(len(rows["tokens"])):
        row_tokens = rows["tokens"][idx]
        row_labels = rows["labels"][idx]
        row_matches = rows["matches"][idx]

        row_adj_labels = copy.deepcopy(row_labels)
        row_wordnet_labels = copy.deepcopy(row_labels)

        row_matches_adj = []
        row_non_matches_adj = []
        row_matches_wordnet = []
        row_non_matches_wordnet = []
        assert len(row_tokens) == len(row_labels)
        for idx, (tok, label) in enumerate(zip(row_tokens, row_labels)):
            if label != "O":
                row_matches_adj.append(tok)
                row_matches_wordnet.append(tok)
            else:
                row_non_matches_adj.append(tok)
                row_non_matches_wordnet.append(tok)
                if idx == len(row_tokens) - 1: # add last adjective
                    doc = nlp(tok)
                    if any(token.pos_ == "ADJ" for token in doc):
                        row_matches_adj.append(row_non_matches_adj.pop())
                        row_adj_labels[idx] = "B-TERM"

                if len(row_matches_wordnet) > 0:
                    doc = is_direct_hyponym_ita(row_matches[-1], tok)
                    if doc[0]: # keep when direct hyponym even if not recognized
                        row_matches_wordnet.append(row_non_matches_wordnet.pop())
                        row_wordnet_labels[idx] = "B-TERM"


        all_adj_labels.append(row_adj_labels)
        all_wordnet_labels.append(row_wordnet_labels)

        all_adj_matches.append(row_matches_adj)
        all_adj_non_matches.append(row_non_matches_adj)

        all_wordnet_matches.append(row_matches_wordnet)
        all_wordnet_non_matches.append(row_non_matches_wordnet)

    rows["adj_matches"] = all_adj_matches
    rows["adj_non_matches"] = all_adj_non_matches

    rows["wordnet_matches"] = all_wordnet_matches
    rows["wordnet_non_matches"] = all_wordnet_non_matches

    rows["adj_labels"] = all_adj_labels
    rows["wordnet_labels"] = all_wordnet_labels

    return rows

def _update_metrics(ds, idx, metrics, prefix=""):
    if prefix:
        prefix = prefix + "_"
    tokens = ds[idx]["tokens"]
    labels = ds[idx][f"{prefix}labels"]
    results = list(zip(tokens, labels))
    full_list = ds[idx]["full_list"]
    clean_results = ds[idx]["clean_list"]
    metrics[f"{prefix}p"] += len(clean_results)
    metrics[f"{prefix}n"] += len(full_list) - len(clean_results)
    metrics[f"{prefix}tp"] += sum(1 for r in results if r[1] != "O" and r[0] in clean_results)
    metrics[f"{prefix}tn"] += sum(1 for r in results if r[1] == "O" and r[0] not in clean_results)
    metrics[f"{prefix}fn"] += sum(1 for r in results if r[1] == "O" and r[0] in clean_results)
    metrics[f"{prefix}fp"] += sum(1 for r in results if r[1] != "O" and r[0] not in clean_results)
    matches = [i[0] for i in results if i[1] != "O"]
    if matches == clean_results:
        metrics[f"{prefix}List Acc"] += 1
    if sum(1 for i in matches if i not in clean_results) + sum(1 for i in clean_results if i not in matches) <= 1:
        metrics[f"{prefix}List Acc @1"] += 1
    if sum(1 for i in matches if i not in clean_results) + sum(1 for i in clean_results if i not in matches) <= 2:
        metrics[f"{prefix}List Acc @2"] += 1
    metrics[f"{prefix}total"] += len(full_list)
    return metrics

def _aggregate_metrics(metrics, ds, prefix=""):
    if prefix:
        prefix = prefix + "_"
    metrics[f"{prefix}precision"] = metrics[f"{prefix}tp"] / (metrics[f"{prefix}tp"] + metrics[f"{prefix}fp"])
    metrics[f"{prefix}recall"] = metrics[f"{prefix}tp"] / (metrics[f"{prefix}tp"] + metrics[f"{prefix}fn"])
    metrics[f"{prefix}f1"] = (2 * metrics[f"{prefix}tp"]) / (2 * metrics[f"{prefix}tp"] + metrics[f"{prefix}fp"] + metrics[f"{prefix}fn"])
    metrics[f"{prefix}accuracy"] = (metrics[f"{prefix}tp"] + metrics["tn"]) / (metrics["tp"] + metrics[f"{prefix}tn"] + metrics[f"{prefix}fp"] + metrics[f"{prefix}fn"])
    metrics[f"{prefix}list accuracy"] = metrics[f"{prefix}List Acc"] / len(ds)
    metrics[f"{prefix}list accuracy @1"] = metrics[f"{prefix}List Acc @1"] / len(ds)
    metrics[f"{prefix}list accuracy @2"] = metrics[f"{prefix}List Acc @2"] / len(ds)
    metrics[f"{prefix}total lists"] = len(ds)
    return metrics

def get_metrics(ds):
    metrics = {
        "n": 0, "p": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0, "List Acc": 0, "List Acc @1": 0, "List Acc @2": 0, "total": 0,
        "adj_n": 0, "adj_p": 0, "adj_tp": 0, "adj_tn": 0, "adj_fp": 0, "adj_fn": 0, "adj_List Acc": 0, "adj_List Acc @1": 0, "adj_List Acc @2": 0, "adj_total": 0,
        "wordnet_n": 0, "wordnet_p": 0, "wordnet_tp": 0, "wordnet_tn": 0, "wordnet_fp": 0, "wordnet_fn": 0, "wordnet_List Acc": 0, "wordnet_List Acc @1": 0, "wordnet_List Acc @2": 0, "wordnet_total": 0,
    }
    for idx in tqdm(range(len(ds))):
        metrics = _update_metrics(ds, idx, metrics)
        metrics = _update_metrics(ds, idx, metrics, prefix="adj")
        metrics = _update_metrics(ds, idx, metrics, prefix="wordnet")

    assert metrics["total"] == (metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"])
    assert metrics["total"] == (metrics["p"] + metrics["n"])
    assert metrics["p"] == (metrics["tp"] + metrics["fn"])
    assert metrics["n"] == (metrics["tn"] + metrics["fp"])

    metrics = _aggregate_metrics(metrics, ds)
    metrics = _aggregate_metrics(metrics, ds, prefix="adj")
    metrics = _aggregate_metrics(metrics, ds, prefix="wordnet")

    return metrics

def run_eval(experiment_name, dataset_name, run_path, file_dir_path, rerun=False, rerun_adj=False):
    if exist_df(os.path.join(file_dir_path, f"{experiment_name}_results.ds")) and not rerun:
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
        ds = ds.map(lambda x: get_adj_and_wordnet_results(x), batched=True, batch_size=10, desc="Getting adj and wornet results")
        ds.save_to_disk(os.path.join(file_dir_path, f"{experiment_name}_results.ds"))

    if exist_df(os.path.join(file_dir_path, f"{experiment_name}_results.ds")) and rerun_adj:
        ds = load_from_disk(os.path.join(file_dir_path, f"{experiment_name}_results.ds"))
        ds = ds.map(lambda x: get_adj_and_wordnet_results(x), batched=True, batch_size=10, desc="Getting adj and wornet results")
        ds.save_to_disk(os.path.join(file_dir_path, f"{experiment_name}_results_with_adj .ds"))

    print("Eval Done")
    all_metrics = {"Full_df": get_metrics(ds)}
    word_metrics = {}
    for word in set(ds["word"]):
        word_metrics[word.strip()] = get_metrics(ds.filter(lambda x: x["word"].strip() == word.strip()))
    return all_metrics, word_metrics

def main():
    import sys

    rerun = len(sys.argv) > 1 and sys.argv[1] == "rerun"
    rerun_adj = len(sys.argv) > 1 and sys.argv[1] == "rerun_adj"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    run_path = "./multirun/2025-10-21/10-52-06/"

    file_dir_path = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(file_dir_path, exist_ok=True)
    if not exist_df(os.path.join(file_dir_path, "primary_school_results.csv")) or rerun or rerun_adj:
        experiments_data_map = {
            "all_with_spec": "./src/primary_school/training_datasets/training_data_all_with_spec.ds",
            "t1_vs_t2_with_spec": "./src/primary_school/training_datasets/training_t1_test_t2_with_spec.ds",
            "t2_vs_t1_with_spec": "./src/primary_school/training_datasets/training_t2_test_t1_with_spec.ds",
            "all_without_spec": "./src/primary_school/training_datasets/training_data_all_no_spec.ds",
            "t1_vs_t2_without_spec": "./src/primary_school/training_datasets/training_t1_test_t2_no_spec.ds",
            "t2_vs_t1_without_spec": "./src/primary_school/training_datasets/training_t2_test_t1_no_spec.ds",
        }
        all_metrics = {}
        all_word_metrics = {}
        for experiment_name, dataset_name in experiments_data_map.items():
            print(f"Running evaluation for {experiment_name} with dataset {dataset_name}")
            full_df_metrics, word_metrics = run_eval(experiment_name, dataset_name, run_path, file_dir_path, rerun=rerun, rerun_adj=rerun_adj)
            all_metrics[experiment_name] = full_df_metrics["Full_df"]
            all_word_metrics[experiment_name] = word_metrics
            print(all_metrics[experiment_name])
        df = pd.DataFrame.from_dict(all_metrics, orient="index")
        df.to_csv(os.path.join(file_dir_path, "primary_school_results.csv"))
        words_df = pd.DataFrame.from_dict({exp: {word: metrics for word, metrics in word_metrics.items()} for exp, word_metrics in all_word_metrics.items()}, orient="index")
        words_df.to_csv(os.path.join(file_dir_path, "primary_school_word_results.csv"))
    else:
        df = pd.read_csv(os.path.join(file_dir_path, "primary_school_results.csv"), index_col=0)
        words_df = pd.read_csv(os.path.join(file_dir_path, "primary_school_word_results.csv"), index_col=0)

    # Rename for better LaTeX readability
    df = df.reset_index().rename({"index": "Setting"}, axis=1)
    def prepare_for_latex(df, prefix="", output_file_name=None):

        if prefix:
            prefix = prefix + "_"
        if output_file_name is None:
            output_file_name = f"{prefix}primary_school_metrics.tex"

        df = df.rename(columns={
            # f"name": f"{prefix}Setting",
            f"{prefix}precision": f"{prefix}Precision",
            f"{prefix}recall": f"{prefix}Recall",
            f"{prefix}f1": f"{prefix}F1-score",
            f"{prefix}accuracy": f"{prefix}Accuracy",
            f"{prefix}list accuracy": f"{prefix}List Accu",
            f"{prefix}list accuracy @1": f"{prefix}List Acc@1",
            f"{prefix}list accuracy @2": f"{prefix}List Acc@2"
        })

        stacked_df = df[["Setting", f"{prefix}Precision", f"{prefix}Recall", f"{prefix}F1-score", f"{prefix}Accuracy", f"{prefix}List Accu", f"{prefix}List Acc@1", f"{prefix}List Acc@2"]]
        stacked_df = stacked_df.rename(columns={
            # f"name": f"{prefix}Setting",
            f"{prefix}Precision": f"Precision",
            f"{prefix}Recall": f"Recall",
            f"{prefix}F1-score": f"F1-score",
            f"{prefix}Accuracy": f"Accuracy",
            f"{prefix}List Accu": f"List Accu",
            f"{prefix}List Acc@1": f"List Acc@1",
            f"{prefix}List Acc@2": f"List Acc@2"
        })

        # stacked_df.Setting = stacked_df.Setting.apply(lambda x: f"{prefix}{x}")
        stacked_df["Experiment"] = prefix[:-1] if prefix else "base"
        stacked_df[f"Setting"] = stacked_df["Setting"].str.replace("_", " ").str.title()

        # Select only the desired columns
        latex_df = df[[f"Setting", f"{prefix}Precision", f"{prefix}Recall", f"{prefix}F1-score", f"{prefix}Accuracy", f"{prefix}List Accu", f"{prefix}List Acc@1", f"{prefix}List Acc@2"]]

        # Round metrics for better formatting
        latex_df = latex_df.round(3)
        latex_df[f"Setting"] = latex_df["Setting"].str.replace("_", " ").str.title()

        # Transpose the DataFrame for LaTeX formatting
        # latex_df = latex_df.T

        # Generate LaTeX table
        latex_code = latex_df.to_latex(
            index=False, float_format="%.3f", column_format='lcccccccc', escape=False)

        with open(os.path.join(file_dir_path, output_file_name), "w") as f:
            f.write(latex_code)

        return stacked_df

    base_df = prepare_for_latex(df)
    adj_df = prepare_for_latex(df, prefix="adj")
    wordnet_df = prepare_for_latex(df, prefix="wordnet")
    latex_df = pd.concat([base_df, adj_df, wordnet_df], axis=0).round(3)

    latex_df.to_csv(os.path.join(file_dir_path, "primary_school_all_metrics.csv"), index=False)

    latex_code = latex_df.set_index(["Experiment", "Setting"]).to_latex(
        float_format="%.3f", column_format='lcccccccc', escape=False)

    with open(os.path.join(file_dir_path, f"all_primary_school_metrics.tex"), "w") as f:
        f.write(latex_code)

    _index = words_df.index
    words_df = words_df.reset_index(drop=True).map(eval)
    for word in words_df.columns:

        word_df = pd.DataFrame.from_dict(words_df[word].to_dict(), orient='index')
        word_df.index = _index
        word_df = word_df.reset_index().rename({"index": "Setting"}, axis=1)
        prepare_for_latex(word_df, output_file_name=f"{word}_primary_school_metrics.tex")



if __name__ == "__main__":
    main()
