import argparse
import subprocess
import json
from pathlib import Path
from ase.data import atomic_numbers
from hashlib import sha512
from ase.io import read, write
from shutil import rmtree
import h5py
from ase import Atoms, units
import numpy as np
import shutil

def create_train_command(config: dict):
    """Create a list of arguments to be used in the subprocess from a dictionary

    Args:
        config: The dictionary that should be converted to a list
    """
    command = []
    for arg, value in config.items():
        if value == "yes":
            #If the value is "yes" then just type the argument and not the value. Used for arguments that are of type "store_true"
            command.extend([f'--{arg}'])
        else:
            command.extend([f'--{arg}', value])
    return command

def get_reference_energies_string(ref_energies_path: Path):
    """Parses and converts the reference energies json file into a string to be used by MACE
    
    Args:
        ref_energies_path: The path to the reference energies json file
    
    Return: A string version of the dict
    """
    ref_energies = json.loads(ref_energies_path.read_text())
    ref_energies_string = "{" + ", ".join(f"{symbol}:{value}" for symbol, value in ref_energies.items()) + "}"
    return ref_energies_string

def run_preprocess(config_path: str, finetune: bool = False):
    """Runs the "preprocess_data.py" script from MACE using arguments from the config json file

    Args:
        config_path: The path to the config json file
        finetune: Whether or not this function is used for finetuning or not
    
    Return: Path to a copy of the config json file
    """
    preprocess_dir = "../../../mace/scripts/preprocess_data.py"

    #Parse the configuration file
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if finetune: config["h5_prefix"] = "finetune_"+config["h5_prefix"]

    #Get reference energies string
    ref_energies_string = get_reference_energies_string(Path(config["reference_energies_path"]))
    
    #Generate an 8-number identifier for the model you are training from the config file
    seed_hash = str(int(sha512(json.dumps(config).encode()).hexdigest(), 16))[-8:] if finetune == False else config["seed_hash"]
    seed_hash = str(int(seed_hash)) #Takes away any starting 0's in the seed, so if the seed was "00468233" then it would change it to "468233", because that is what the MACE code does.
    name = f"{config['all_train']['name']}_run-{seed_hash}"

    #Create/set all of the required data files
    method_dir = Path(f"{config['new_data_location']}/{name}")
    if finetune == False:
        #Creates a new directory for this run and makes the h5 files for training
        print(f"Creating xyz dataset for {name}")
        if method_dir.exists():
            rmtree(method_dir)
        method_dir.mkdir()
        xyz_file_path = method_dir/'train.xyz'
        h5_to_xyz(h5_file_path=config["initial_dataset"], xyz_file_path=xyz_file_path, percentage=config["train_size"])
    else:
        xyz_file_path = config['finetune_dataset']


    #Create argument list and add any arguments that might change between runs
    command = [
        "python",
        preprocess_dir,
        "--E0s", ref_energies_string,
        "--seed", seed_hash,
        "--h5_prefix", f"{method_dir}/{config['h5_prefix']}",
        "--train_file", xyz_file_path
    ]

    if(finetune == False):
        command.extend(create_train_command(config["train_preprocess"]))
    else:
        command.extend(create_train_command(config["finetune_preprocess"]))

    print(f"Running preprocessing for '{config['all_train']['model']}' model named '{name}' using config from '{config_path}' with the parameters: \n {command}")
    
    subprocess.run(command, check=True)

    #Save this config to a directory for use later
    config_save_path: str
    if finetune == False:
        config["seed_hash"] = seed_hash
        config_save_path = f'{config["configs_dir"]}/{name}.json'
        with open(config_save_path, 'w') as f:
            json.dump(config, f, indent=4)
    else:
        config_save_path = config_path
    
    return config_save_path

def run_train(config_path: str, finetune: bool = False):
    """Runs the "run_train.py" script from MACE using the arguments from the config json file

    Args:
        config_path: The path to the config json file
        finetune: Whether or not this function is used for finetuning or not
    """

    run_train_file = "../../../mace/scripts/run_train.py"

    #Parse the configuration file
    with open(config_path, 'r') as f:
        config = json.load(f)

    #Get reference energies string
    ref_energies_string = get_reference_energies_string(Path(config["reference_energies_path"]))

    #The seed hash has already been saved from preprocessing
    seed_hash = config["seed_hash"]
    name = f"{config['all_train']['name']}_run-{seed_hash}"

    method_dir = Path(f"{config['new_data_location']}/{name}")

    #Change the h5 prefix make a copy the intial training model for evaluation later
    if finetune:
        config["h5_prefix"] = "finetune_"+config["h5_prefix"]
        shutil.copy2(f"{config['all_train']['checkpoints_dir']}/{name}.model", f"{config['all_train']['checkpoints_dir']}/{name}_initial.model")

    #Create argument list and add any arguments that might change between runs
    command = [
        "python",
        run_train_file,
        "--E0s", ref_energies_string,
        "--seed", seed_hash,
        "--train_file", f"{method_dir}/{config['h5_prefix']}train.h5",
        "--valid_file", f"{method_dir}/{config['h5_prefix']}valid.h5",
        "--statistics_file", f"{method_dir}/{config['h5_prefix']}statistics.json"
    ]

    #Add every argument in the config file
    command.extend(create_train_command(config["all_train"]))
    if(finetune == False):
        command.extend(create_train_command(config["train"]))
    else:
        command.extend(create_train_command(config["finetune_train"]))
    if("wandb" in config["all_train"]):
        command.extend(add_wandb(config=config))
        if finetune:
            command.extend(["--wandb_name", f"{name}-finetune"])
        else:
            command.extend(["--wandb_name", name])

    print(f"Running training for '{config['all_train']['model']}' model named '{name}' using config from '{config_path}' with the parameters: \n {command}")

    #Run the command
    subprocess.run(command, check=True)


def add_wandb(config: dict):
    """Adds the list of hyper parameters to log for weights and biases

    Args:
        config: The dictionary for the config json file

    Returns: The wandb_log_hypers command
    """
    command = []
    if("wandb_log_hypers" in config):
        command = ["--wandb_log_hypers"]
        command.extend([hyper for hyper in config["wandb_log_hypers"]])
    return command
        

# Function to convert a page of the HDF5 file to an Atoms object
def page_to_atoms(page, energy_method: str, forces_method: str, percentage=None):
    """Convert an h5 page to a list of molecules

    Args:
        page: An h5 page
        energy_method: The index of the total energy in the h5 page
        forces_method: The index of the forces for each atom in the h5 page
        percentage: The percentage of molecules from each type to save. For example, 20 means 20%. None means no limit
    
    Returns: An array of Atoms objects
    """
    # Get the energies at the desired level
    all_energies = page[energy_method]
    
    # Get the forces if they are available
    all_forces = page[forces_method] if forces_method in page else None

    # Calculate the number of molecules to yield based on the percentage
    num_molecules = int(len(all_energies) * percentage / 100) if percentage is not None else None

    # Generate configurations
    for i, coords in enumerate(page['coordinates']):
        # Skip if energy not done
        if np.isnan(all_energies[i]):
            continue

        # Make the atoms object
        atoms = Atoms(numbers=page['atomic_numbers'], positions=coords)

        # Attach energy and forces to the Atoms object
        atoms.info['energy'] = all_energies[i] * units.Ha
        atoms.arrays['forces'] = np.zeros_like(coords) if all_forces is None else all_forces[i] * units.Ha

        yield atoms

        # Stop if we've reached the desired number of molecules
        if num_molecules is not None and i+1 >= num_molecules:
            break


def h5_to_xyz(h5_file_path, xyz_file_path, method="wb97x_dz", percentage=None):
    """Convert h5 file to xyz file

    Args:
        h5_file_path: File path for h5 file that you want to convert
        xyz_file_path: Path where you want to save the converted xyz file
        method: The method of the h5 file that you want to save
        num_molecules: The number of molecules from each type to save. 1 would be the lowest, saving one molecule from each of the types of molecules in the h5 file. None means no limit
    """
    with h5py.File(h5_file_path, 'r') as original_data:
        # Loop over each composition
        for composition, page in original_data.items():
            for atoms in page_to_atoms(page, f'{method}.energy', f'{method}.forces', percentage):
                write(xyz_file_path, atoms, format='extxyz', append=True)  # Use ase's write function to write to xyz file