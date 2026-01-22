import os

import numpy as np
import pandas as pd
import torch
from src.utils.SVDSuperimposer import l2_loss_with_superimposer
from tqdm import tqdm

from config import model_config
from dataloader import DataLoader
from model import Model
import argparse

def run(val_kfold: int):
    model_config.init(val_kfold)
    net = Model(model_config)
    optim = torch.optim.Adam(net.parameters(), lr=model_config.learning_rate)
    train_fold = [i for i in range(5) if i != val_kfold]
    train_loader = DataLoader(
        min_length=model_config.dataset_min_length,
        max_length=model_config.dataset_max_length,
        dataset_type="train",
        select_fold=train_fold,
    )
    val_loader = DataLoader(
        min_length=model_config.dataset_min_length,
        max_length=model_config.dataset_max_length,
        dataset_type="train",
        select_fold=val_kfold,
    )

    df = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }
    for epoch in tqdm(range(model_config.epoch), desc="epoch"):
        net.train()
        train_loader.shuffle()
        train_loss = []
        for i, (pdb_id, length, sequence, structure) in enumerate(train_loader):
            out_abs = net(sequence)
            loss = l2_loss_with_superimposer(out_abs, structure, svd_grad=True)
            loss.backward()
            optim.step()
            optim.zero_grad()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)

        val_loss = []
        net.eval()
        for i, (pdb_id, length, sequence, structure) in enumerate(val_loader):
            out_abs = net(sequence)
            loss = l2_loss_with_superimposer(out_abs, structure, svd_grad=False)
            val_loss.append(loss.item())
        val_loss = np.mean(val_loss)

        print(
            f"epoch {epoch} loss - train {train_loss} - val {val_loss}"
        )
        df["epoch"].append(epoch)
        df["train_loss"].append(train_loss)
        df["val_loss"].append(val_loss)

        # save model
        if not os.path.exists("./saved_model"):
            os.makedirs("./saved_model")
        torch.save(net.state_dict(), f"./saved_model/epoch{epoch}_val{val_kfold}.pth")

    df = pd.DataFrame(df)
    df.to_csv(f"result_val{val_kfold}.csv", index=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--valkfold', type=int, default=None)
    args = parser.parse_args()
    run(args.valkfold)