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
parser.add_argument('--finetune', help='Whether or not to finetune', type=bool, default=True)
parser.add_argument('--configs', help='Path to configuration folder', type=str, default="../configs/run_list")
parser.add_argument('--method', help='Path to configuration file', type=str, default="wb97x_dz")
args = parser.parse_args()

finetune = True
configs = "../configs/run_list"
method = "wb97x_dz"

for config_name in os.listdir(configs):
    config_name = f"{configs}/{config_name}"
    if(str(config_name).endswith(".json")):

        #Parse the configuration file
        with open(config_name, 'r') as f:
            config = json.load(f)

        #Change the reference energies name
        config["reference_energies_path"] = f"../data/{method}-reference_energies.json"
        with open(config_name, 'w') as f:
                json.dump(config, f, indent=4)  

        #Initial training
        config_path = run_preprocess(config_path=config_name)
        run_train(config_path=config_path)


        #Finetuning
        if(finetune == True):
            #Parse the configuration file
            with open(config_name, 'r') as f:
                config = json.load(f)

            #Change the reference energies to the new ones
            config["reference_energies_path"] = f"../data/{method}-new_reference_energies.json"
            with open(config_name, 'w') as f:
                    json.dump(config, f, indent=4)

            #Run preprocessing and training
            run_preprocess(config_path=config_path, finetune=True)
            run_train(config_path=config_path, finetune=True)