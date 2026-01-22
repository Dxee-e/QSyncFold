import pathlib
import numpy as np
import pandas as pd


def read_a3m(path: pathlib.Path):
    seqs = []
    with path.open() as f:
        current = []
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("#"):
                continue  # skip header/meta lines like "#9\t1"
            if line.startswith(">"):
                if current:
                    seqs.append("".join(current))
                    current = []
            else:
                current.append(line.strip())
        if current:
            seqs.append("".join(current))
    # strip insertions (lowercase) to keep aligned length
    seqs = [s.replace(".", "").replace("*", "") for s in seqs]
    seqs = ["".join([c for c in s if not c.islower()]) for s in seqs]
    return seqs


def seq_identity(a: str, b: str):
    a_arr, b_arr = np.frombuffer(a.encode(), dtype="S1"), np.frombuffer(b.encode(), dtype="S1")
    both = (a_arr != b"-") & (b_arr != b"-")
    denom = both.sum()
    if denom == 0:
        return 0.0
    matches = (a_arr[both] == b_arr[both]).sum()
    return matches / denom


def neff(seqs, thresh=0.8):
    if len(seqs) == 0:
        return 0.0
    n = len(seqs)
    weights = np.ones(n, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if seq_identity(seqs[i], seqs[j]) >= thresh:
                weights[i] += 1.0
    return float((1.0 / weights).sum())


def compute_stats(seqs):
    if not seqs:
        return {"depth": 0, "mean_coverage": 0.0, "mean_gap_frac": 0.0, "neff": 0.0}
    aln_len = len(seqs[0])
    for s in seqs:
        if len(s) != aln_len:
            raise ValueError("Alignment length mismatch in A3M.")
    arr = np.array([list(s) for s in seqs])
    non_gap = (arr != "-")
    coverage = non_gap.sum(axis=0) / non_gap.shape[0]
    mean_cov = float(coverage.mean())
    mean_gap = float(1.0 - mean_cov)
    return {
        "depth": len(seqs),
        "mean_coverage": mean_cov,
        "mean_gap_frac": mean_gap,
        "neff": neff(seqs),
    }


def collect_a3m_files(outputs_dir: pathlib.Path):
    files = list(outputs_dir.glob("*.a3m"))
    files += list(outputs_dir.glob("*_env/*.a3m"))
    return files


def main(outputs_dir="outputs", save_path="outputs/msa_stats.csv"):
    outputs_dir = pathlib.Path(outputs_dir)
    records = []
    for a3m_path in collect_a3m_files(outputs_dir):
        seqs = read_a3m(a3m_path)
        stats = compute_stats(seqs)
        records.append({
            "file": str(a3m_path.relative_to(outputs_dir)),
            **stats,
        })
    df = pd.DataFrame(records).sort_values("file")
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(df)
    return df


if __name__ == "__main__":
    main()


