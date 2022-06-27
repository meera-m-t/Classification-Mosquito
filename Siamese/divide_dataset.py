
import glob
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import splitfolders
import os


def split_data(path, output_path):
    output = output_path
    splitfolders.ratio(f'{path}', output=output, seed=1337, ratio=(.8, 0.2))

def main():
    parser = argparse.ArgumentParser(description='split dataset')
    parser.add_argument(
       '-p','--path_dataset', required=False, default='/home/meera/myjob/Computer-Vision---METHODS/Dataset/processed_dataset/', type=str, help='path of dataset')
    parser.add_argument(
       '-p_ws','--path_output', required=False, default='/home/meera/myjob/Computer-Vision---METHODS/Siamese/Dataset/', type=str, help='path of output')    
    args = parser.parse_args()
    path_dataset = args.path_dataset
    path_output = args.path_output 
    split_data( path_dataset, path_output)

if __name__ == "__main__":
    main()