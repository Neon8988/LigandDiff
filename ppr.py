import argparse
from pathlib import Path
import os
import numpy as np
import torch
from src import const
from src.lightning import DDPM


parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=Path)
parser.add_argument('--model', type=Path)
parser.add_argument('--dataset', type=Path)
parser.add_argument('--batch_size', type=int, default=64)


def main(outdir,model,dataset,batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    ddpm = DDPM.load_from_checkpoint(model, map_location=device).eval().to(device)
    dataset=torch.load(dataset,map_location=device)
    batch_size=batch_size if batch_size is not None else ddpm.batch_size
    with torch.no_grad():
        ddpm.sample_and_analyze(dataset, batch_size=batch_size,outdir=outdir,animation=False)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.outdir,args.model,args.dataset,args.batch_size)




