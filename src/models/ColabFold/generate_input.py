"""generate input csv for localcolabfold
csv consist of 2 columns: id, sequence"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

data_dir = "../../../data/"
idx = pd.read_csv(os.path.join(data_dir, "dataset_index.csv"))
idx = idx[idx["dataset_type"] == "test"]
data: Dict[str, List[str]] = {
    "id": [],
    "sequence": [],
}
for i, row in tqdm(idx.iterrows(), total=len(idx)):
    item = np.load(os.path.join(data_dir, "FormatData", f"{row['pdb_id']}.npz"))
    data["id"].append(row["pdb_id"])
    data["sequence"].append("".join(item["sequence"]))


df = pd.DataFrame(data)
df = df.sort_values(by="sequence", key=lambda x: x.str.len())
df.to_csv("inputs.csv", index=None)
