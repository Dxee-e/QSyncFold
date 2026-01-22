import os.path

import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Dict

from src.metrics.rmsd import rmsd
from src.metrics.gdt_ts import gdt_ts
from src.metrics.lddt import lddt
from src.metrics.tmscore import tmsocre
from src.metrics.fix_coords import fix_coords_by_svd, fix_coords_by_xyz


def get_VQE_result(pdb_ids: List[str]) -> Dict[str, np.ndarray]:
    data_dir = os.path.join("../models/VQE/result/")
    results: Dict[str, np.ndarray] = {}
    for pdb_id in pdb_ids:
        try:
            results[pdb_id] = np.load(os.path.join(data_dir, f"{pdb_id}.npz"))[
                                  "target"
                              ][np.newaxis, :, :]
        except FileNotFoundError:
            pass
    return results


def get_QAOA_result(pdb_ids: List[str]) -> Dict[str, np.ndarray]:
    data_dir = os.path.join("../models/QAOA/result/")
    results: Dict[str, np.ndarray] = {}
    for pdb_id in pdb_ids:
        try:
            results[pdb_id] = np.load(os.path.join(data_dir, f"{pdb_id}.npz"))[
                                  "target"
                              ][np.newaxis, :, :]
        except FileNotFoundError:
            pass
    return results


def get_colabfold_result(pdb_ids: List[str]) -> Dict[str, np.ndarray]:
    data = np.load("../models/ColabFold/format_result_nothing.npz")
    results: Dict[str, np.ndarray] = {}
    for pdb_id in pdb_ids:
        try:
            results[pdb_id] = data[pdb_id]
        except KeyError:
            pass
    return results


def get_QNN_result(pdb_ids: List[str]) -> Dict[str, np.ndarray]:
    data = np.load('../models/QNN/results.npz')
    results: Dict[str, np.ndarray] = {}
    for pdb_id in pdb_ids:
        try:
            results[pdb_id] = data[pdb_id]
        except KeyError:
            pass
    return results


def run(
        dataset_length: Tuple[int, int],
        model_result_callables: List[Tuple[str, Callable]],
        metrics: List[Callable],
        superimposer_method: str,
) -> None:
    results = {
        model_callable[0]: {
            "pdb_id": [],
            "count": [],
            "length": [],
            **{metric.__name__: [] for metric in metrics},
        }
        for model_callable in model_result_callables
    }

    # load index
    dataset_index = pd.read_csv("../../data/dataset_index.csv")
    dataset_index = dataset_index[dataset_index["dataset_type"] == "test"]
    dataset_index = dataset_index[(dataset_index["length"]>=dataset_length[0]) & (dataset_index["length"]<=dataset_length[1])]

    # load reference structure
    references: Dict[str, np.ndarray] = {}
    for i, row in dataset_index.iterrows():
        pdb_id = row["pdb_id"]
        references[pdb_id] = np.load(f"../../data/FormatData/{pdb_id}.npz")["coord"]

    # load model predict structure
    targets: Dict[
        str, Dict[str, np.ndarray]
    ] = {}  # {model_name: {pdb_id: [structure, ...]}}
    for model_name, model_result_callable in model_result_callables:
        targets[model_name] = model_result_callable(dataset_index["pdb_id"].tolist())

    # transformer - superimposer
    if superimposer_method == "None":
        pass
    elif superimposer_method == "fix_coords_by_svd":
        targets = {
            model_name: {
                pdb_id: np.stack(
                    [
                        fix_coords_by_svd(references[pdb_id], structure)
                        for structure in structures
                    ],
                    axis=0,
                )
                for pdb_id, structures in model_result.items()
            }
            for model_name, model_result in targets.items()
        }
    elif superimposer_method == "fix_coords_by_xyz":
        targets = {
            model_name: {
                pdb_id: np.stack(
                    [fix_coords_by_xyz(structure) for structure in structures]
                )
                for pdb_id, structures in model_result.items()
            }
            for model_name, model_result in targets.items()
        }
        references = {
            pdb_id: fix_coords_by_xyz(structure)
            for pdb_id, structure in references.items()
        }
    else:
        raise ValueError("Invalid superimposer method")

    # calculate metrics
    for model_name, model_result in targets.items():
        for pdb_id, predict_structures in model_result.items():
            for count in range(predict_structures.shape[0]):
                results[model_name]["pdb_id"].append(pdb_id)
                results[model_name]["count"].append(count)
                results[model_name]["length"].append(len(predict_structures[0, :, 0]))
                for metric_callable in metrics:
                    metric_result = metric_callable(
                        predict_structures[count, :, :], references[pdb_id]
                    )
                    results[model_name][metric_callable.__name__].append(metric_result)

    for model_name in results.keys():
        df = pd.DataFrame(results[model_name])
        if not os.path.exists(f'./{superimposer_method}/'):
            os.mkdir(f'./{superimposer_method}/')
        df.to_csv(f"./{superimposer_method}/performance{model_name}.csv", index=False)


if __name__ == '__main__':
    metrics = [
        rmsd,
        gdt_ts,
        lddt,
        tmsocre,
    ]
    model_result_callables = [
        ('VQE', get_VQE_result),
        ('QAOA', get_QAOA_result),
        ('ColabFold', get_colabfold_result),
        ('QNN', get_QNN_result)
    ]
    run((5, 9), model_result_callables, metrics, 'fix_coords_by_xyz')
    run((5, 9), model_result_callables, metrics, 'fix_coords_by_svd')
