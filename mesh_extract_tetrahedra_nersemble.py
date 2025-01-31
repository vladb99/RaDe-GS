from argparse import ArgumentParser
from tqdm import tqdm
import os

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Script parameters")
    parser.add_argument("--timesteps_folder", type=str)
    parser.add_argument("--models_folder", type=str)
    args = parser.parse_args()

    timesteps_folders = sorted(os.listdir(args.timesteps_folder))

    for timestep_folder in tqdm(timesteps_folders):
        os.system("python mesh_extract_tetrahedra.py -s {} -m {} -r 2 -w".format(os.path.join(args.timesteps_folder, timestep_folder), os.path.join(args.models_folder, timestep_folder)))