import pandas as pd
import matplotlib.pyplot as plt

"""
Examples:

angle_var_results.csv:
seq,length,pdb_id,angle_var,R
['Q' 'Y' 'S' 'N' 'Q' 'N' 'S' 'F' 'V'],9,7rvg,0.5127360694990891,0.7198248994240479
['G' 'S' 'T' 'V' 'Y' 'A' 'P' 'F' 'T'],9,7n2i,0.49683946696237247,0.6408247225566654
['Q' 'Y' 'N' 'N' 'Q' 'N' 'N' 'F' 'V'],9,6axz,0.5131976895827237,0.7332156881153652
['Q' 'Y' 'N' 'N' 'E' 'N' 'N' 'F' 'V'],9,7n2l,0.5222647759106602,0.7128248299915049
['L' 'A' 'A' 'A' 'L' 'A' 'Q' 'A' 'L'],9,7tm2,0.49724636300907243,0.7022592777720461
['L' 'A' 'A' 'S' 'L' 'A' 'C' 'A' 'L'],9,8sy4,0.549316729121856,0.8013482022947698
['D' 'I' 'R' 'L' 'A' 'K' 'T' 'L' 'V'],9,1lvq,0.48282069375578435,0.6238167651959154
"""

def main(csv_path: str = "angle_var_results.csv"):
    df = pd.read_csv(csv_path)
    required = {"angle_var", "R"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}; found: {sorted(df.columns)}")

    df["angle_var"] = pd.to_numeric(df["angle_var"], errors="coerce")
    df["R"] = pd.to_numeric(df["R"], errors="coerce")
    df = df.dropna(subset=["angle_var", "R"])

    plt.rcParams.update({"font.family": "Arial", "font.size": 12})
    plt.style.use("ggplot")
    plt.figure(figsize=(6, 5))
    plt.scatter(df["angle_var"], df["R"], s=50, alpha=0.75, edgecolor="k", linewidth=0.6)
    plt.xlabel("Var")
    plt.ylabel("R")
    plt.tight_layout()
    plt.savefig("0angle_var_vs_R.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()

