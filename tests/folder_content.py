# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:42:10 2026

@author: clher
"""

from pathlib import Path
import pandas as pd


def list_folder_contents(target_path):
    # Convert the string path into a Path object
    path = Path(target_path)

    # Check if the path actually exists and is a directory
    if not path.exists():
        return f"Error: The path '{target_path}' does not exist."
    if not path.is_dir():
        return f"Error: '{target_path}' is not a folder."

    print(f"Contents of: {path.resolve()}\n")
    print(f"{'Type':<10} | {'Name'}")
    print("-" * 30)

    # Iterate through every item in the folder
    for item in path.iterdir():
        # Determine if it's a file or a folder for better clarity
        item_type = "Folder" if item.is_dir() else "File"
        print(f"{item_type:<10} | {item.name}")


# --- Implementation ---
# Note: Use a 'raw' string (r"path") for Windows paths to handle backslashes correctly.

# Path(__file__).resolve().parent
my_windows_path = r"C:\Users\Public\Documents"
list_folder_contents(my_windows_path)

# %%
target_path = my_windows_path
path = Path(target_path)

dirlist = []
filelist = []


for item in path.iterdir():
    # Determine if it's a file or a folder for better clarity
    dirlist.append(item.name) if item.is_dir() else filelist.append(item.name)
    print(item.name)


foldercontent = {'folders': dirlist, 'files': filelist}

fcdf = pd.DataFrame([foldercontent])
