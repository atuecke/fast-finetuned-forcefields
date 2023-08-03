import argparse
import subprocess
import json
import os
from pathlib import Path
from ase.data import atomic_numbers
from hashlib import sha512
from ase.io import read, write
from shutil import rmtree
from fff.fff_mace_utils.fffmaceutils import run_preprocess, run_train

#Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to configuration file', type=str, default="../configs/config.json")
parser.add_argument('--method', help='Path to configuration file', type=str, default="wb97x_dz")
args = parser.parse_args()

#Parse the configuration file
with open(args.config, 'r') as f:
    config = json.load(f)

#Change the reference energies name
config["reference_energies_path"] = f"../data/{args.method}-reference_energies.json"
with open(args.config, 'w') as f:
        json.dump(config, f, indent=4)

config_path = run_preprocess(config_path=args.config)
run_train = run_train(config_path=config_path)