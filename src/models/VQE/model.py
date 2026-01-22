"""# Model Define
VQE
loss function: CVaR
ansatz: Real Amplitudes Ansatz
optimizer: COBYLA """

import random

import numpy as np
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.utils import algorithm_globals
from qiskit_research.protein_folding.interactions.miyazawa_jernigan_interaction import (
    MiyazawaJerniganInteraction,
)
from qiskit_research.protein_folding.penalty_parameters import PenaltyParameters
from qiskit_research.protein_folding.peptide.peptide import Peptide
from qiskit_research.protein_folding.protein_folding_problem import (
    ProteinFoldingProblem,
)

random.seed(55555)
np.random.seed(55555)
algorithm_globals.random_seed = 55555


def solve_problem(sequence: str):
    # random_interaction = RandomInteraction()
    mj_interaction = MiyazawaJerniganInteraction()
    penalty_back = 10
    penalty_chiral = 10
    penalty_1 = 10
    penalty_terms = PenaltyParameters(penalty_chiral, penalty_back, penalty_1)
    main_chain = sequence
    main_chain = main_chain.replace("X", "G")
    peptide = Peptide(main_chain, [""] * len(main_chain))
    protein_folding_problem = ProteinFoldingProblem(
        peptide, mj_interaction, penalty_terms
    )
    qubit_op = protein_folding_problem.qubit_op()
    optimizer = COBYLA(maxiter=50)
    ansatz = RealAmplitudes(reps=1)
    vqe = SamplingVQE(
        Sampler(),
        ansatz=ansatz,
        optimizer=optimizer,
        aggregation=0.1,
    )
    raw_result = vqe.compute_minimum_eigenvalue(qubit_op)
    result = protein_folding_problem.interpret(raw_result=raw_result)
    target = result.protein_shape_file_gen.get_xyz_data()[:, 1:]
    target = target.astype(np.float32)
    return target
