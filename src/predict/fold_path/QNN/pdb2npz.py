from Bio.PDB import PDBParser
import numpy as np

# read pdb file and extract alpha carbon coordinates
file_name = 'AF-P04637-F1-model_v4.pdb'

pdb_parser = PDBParser()

st = pdb_parser.get_structure('st', file_name)
print(len(st))
model = st[0]
for chain in model:
    coord = []
    residues = chain.get_residues()
    for residue in residues:
        for atom in residue:
            if atom.get_name() == 'CA':
                coord.append(atom.coord)
    coord = np.stack(coord, axis=0)
print(coord.shape)

np.save('ref_coord.npy', coord)