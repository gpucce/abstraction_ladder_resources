import pandas as pd
import datasets

preposizioni_italiane = [
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    "della", "dello", "dell'", "delle", "degli", "del", "ai", "agli", "alla", "alle",
    "col", "coi", "colle", "coll'", "dai", "dagli", "dalla", "dalle", "nei", "negli",
    "nella", "nelle", "sui", "sugli", "sulla", "sulle",
]

def add_word_to_specification(x):
    """Change ["cappello", "di stoffa"] into ["cappello", "cappello di stoffa"]."""

    mapping = ""
    starts_with_prep = [any(j.startswith(prep + " ") for prep in preposizioni_italiane) for j in x]
    if not any(starts_with_prep[1:]):
        return x, mapping

    findall = [i for i, v in enumerate(starts_with_prep) if v and not (starts_with_prep[i-1] if i > 0 else True)]
    for idx, starts in enumerate(starts_with_prep):
        if starts:
            indices = [i for i in findall if i <= idx]
            if indices:
                found = max(indices)
                mapping += x[idx] + "|||" + x[found - 1] + " " + x[idx] + "___"
                x[idx] = x[found - 1] + " " + x[idx]

    return x, mapping

def polish_list(x):
    x = x.strip()
    if x.endswith(","):
        x = x[:-1]
    if x.startswith(","):
        x = x[1:]
    return x

def create_ds_from_df(df, add_word_spec=False):
    ds = datasets.Dataset.from_pandas(
        df.reset_index().loc[:, ["index","hypernym", "hyponym", "word", "words_excluded"]])
    ds = ds.map(lambda x: {
        "hypernym": x["hypernym"] if x["hypernym"] is not None else "",
        "hyponym": x["hyponym"] if x["hyponym"] is not None else "",
    })

    # clean specific entry
    ds = ds.map(
        lambda x: {"full_list": x["hypernym"] + ", " + x["word"] + ", " + x["hyponym"]})

    ds = ds.map(lambda x: {"full_list": polish_list(x["full_list"])})
    ds = ds.map(lambda x: {"full_list": x["full_list"].split(', ')})

    ds = ds.map(lambda x:
        {"clean_list": [j for j in x["full_list"]
        if x["words_excluded"] is None
        or j not in x["words_excluded"].split(', ')
        or j == x["word"]]})

    for full, clean in zip(ds["full_list"], ds["clean_list"]):
        assert all([j in full for j in clean]), f"Cleaned list {clean} is not a subset of full list {full}"

    ds = ds.map(lambda x: {"label": ["B-" if j in x["clean_list"] else "O" for j in x["full_list"]]})


    def map_addition(x):
        new_list, map = add_word_to_specification(x["full_list"])
        return {"full_list": new_list, "word_spec_map": map}

    if add_word_spec:
        ds = ds.map(lambda x: map_addition(x))

        def convert_map_to_dict(x):
            word_spec_map = {}
            for item in x["word_spec_map"].split("___"):
                if item:
                    key, value = item.split("|||")
                    word_spec_map[key] = value
            return word_spec_map

        ds = ds.map(lambda x: {
            "clean_list": [convert_map_to_dict(x).get(j, j) for j in x["clean_list"]],
        })
        for idx, (full, clean) in enumerate(zip(ds["full_list"], ds["clean_list"])):
            assert all([j in full for j in clean]), f"Cleaned list {clean} is not a subset of full list {full}"

    return ds

if __name__ == "__main__":

    import os
    os.makedirs('./training_datasets', exist_ok=True)

    df_excluded = pd.read_excel('./data_excluded_words.xlsx')
    df_t1 = pd.read_excel('./data_excluded_words.xlsx', sheet_name="Time1")
    df_t2 = pd.read_excel('./data_excluded_words.xlsx', sheet_name="Time2")
    df_t1 = df_t1.loc[df_t1.n_words_excluded != " "]

    ds_t1 = create_ds_from_df(df_t1)
    ds_t2 = create_ds_from_df(df_t2)

    datasets.DatasetDict({"train": ds_t1, "validation": ds_t2}).save_to_disk('./training_datasets/training_t1_test_t2_no_spec.ds')
    datasets.DatasetDict({"train": ds_t2, "validation": ds_t1}).save_to_disk('./training_datasets/training_t2_test_t1_no_spec.ds')

    ds_t1_spec = create_ds_from_df(df_t1, add_word_spec=True)
    ds_t2_spec = create_ds_from_df(df_t2, add_word_spec=True)

    datasets.DatasetDict({"train": ds_t1_spec, "validation": ds_t2_spec}).save_to_disk('./training_datasets/training_t1_test_t2_with_spec.ds')
    datasets.DatasetDict({"train": ds_t2_spec, "validation": ds_t1_spec}).save_to_disk('./training_datasets/training_t2_test_t1_with_spec.ds')

    all_no_spec = datasets.concatenate_datasets([ds_t1, ds_t2]).train_test_split(test_size=0.2)
    all_no_spec["validation"] = all_no_spec["test"]
    del all_no_spec["test"]
    all_no_spec.save_to_disk('./training_datasets/training_data_all_no_spec.ds')

    all_spec = datasets.concatenate_datasets([ds_t1_spec, ds_t2_spec]).train_test_split(test_size=0.2)
    all_spec["validation"] = all_spec["test"]
    del all_spec["test"]
    all_spec.save_to_disk('./training_datasets/training_data_all_with_spec.ds')