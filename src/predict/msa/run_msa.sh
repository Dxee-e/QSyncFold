#!/bin/bash

RANDOMSEED=0

export PATH="./localcolabfold/.pixi/envs/default/bin:${PATH}"

colabfold_batch inputs.fasta ./outputs --msa-only