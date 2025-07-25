import pandas as pd
import ast

def try_parse_list(s):
    """Try to parse a string representation of a list into an actual list."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []

def correct_refusi(word, map_refusi):
    return map_refusi.get(word, word)

def main():
    black_lists = pd.read_excel("./Liste word ladders.xlsx")
    refusi = pd.read_excel("./refusi.xlsx")

    df = pd.read_excel("./WORDLADDERS_laddercheck_20240301.xlsx")
    df.ladder = df.ladder.apply(try_parse_list)

    map_refusi = {}
    for i in refusi.loc[:, ["refuso", "corretto"]].to_dict(orient="records"):
        map_refusi[i["refuso"]] = i["corretto"]


    df.ladder = df.ladder.apply(lambda x: [correct_refusi(w, map_refusi) for w in x])
    to_keep = (
        df.ladder.apply(lambda x: len(x) > 0) &
        df.ladder.apply(lambda x: all(len(i) > 2 for i in x))
    )
    df["to_keep"] = to_keep
    df = df[df.to_keep].drop(columns=["to_keep"])
    df.to_csv('./word_ladders_cleaned.csv', index=False, sep="\t")

if __name__ == "__main__":
    main()