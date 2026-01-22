import pandas as pd
from Bio.PDB import PDBParser
from typing import List
import os
import numpy as np

def get_CA_coords(result_dir_name:str):
    file_names = os.listdir(result_dir_name)
    file_names = [i for i in file_names if i.endswith(".pdb")]
    inputs_csv = pd.read_csv('inputs.csv')
    inputs_csv.set_index('id', inplace=True)
    result = {k: [] for k in set(i.split('_')[0] for i in file_names)}
    for file_name in file_names:
        pdb_id = file_name.split('_')[0]
        parser = PDBParser()
        structure = parser.get_structure(pdb_id, os.path.join(result_dir_name, file_name))
        coord = []
        for model_i, model in enumerate(structure.get_models()):
            for chain_i, chain in enumerate(model.get_chains()):
                for residue in chain.get_residues():
                    for atom in residue.get_atoms():
                        if atom.get_name() == "CA":
                            coord.append(atom.get_coord())
        protein_sequence = inputs_csv.loc[pdb_id]['sequence']
        # replace all 'X' to average coord
        for i in range(len(protein_sequence)):
            if protein_sequence[i] == 'X':
                if i==0:
                    coord.insert(0, [0,0,0])
                elif i==len(protein_sequence)-1:
                    coord.append(coord[-1])
                else:
                    coord.insert(i, np.mean([coord[i-1], coord[i]], axis=0).tolist())
        if len(coord) !=len(protein_sequence):
            print(f'Error: {pdb_id} - {file_name}')
        coords = np.array(coord)
        result[pdb_id].append(coords)
    np.savez(f"format_{result_dir_name}.npz", **result)
    print("> ok~")

if __name__=="__main__":
    get_CA_coords("result_nothing")