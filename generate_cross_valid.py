#!/usr/bin/env python
# coding: utf-8

# # Define Globals

# In[1]:


import os
from pathlib import Path
import numpy as np
import shutil


# In[2]:


# Original database path
database_path = "./"

# Splited validation set rate (only apply to atom_11 and atom_12)
database_split_rate = 0.02

# Splited validation sets
database_split_number = 5


# In[3]:


def makedirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


# In[4]:


# Make nessary directories

# makedirs(Path(database_destination_path) / "raw" / "dev")
for i in range(database_split_number):
    for j in ["11", "12"]:
        makedirs(Path(database_path) / "raw" / "valid_{:02d}".format(i) / "sdf" / ("atom_" + j))


# # Copy file by probability

# In[5]:


for j in ["11", "12"]:
    np.random.seed(0)
    copy_file_idx = np.array([int(i[:-4]) for i in os.listdir(Path(database_path) / "raw-orig" / "valid" / "sdf" / ("atom_" + j))])
    np.random.shuffle(copy_file_idx)
    copy_batch = int(copy_file_idx.size * database_split_rate) + 1
    for i in range(database_split_number):
        for k in copy_file_idx[copy_batch * i : copy_batch * (i + 1)]:
            shutil.copy(
                Path(database_path) / "raw-orig" / "valid" / "sdf" / ("atom_" + j) / (str(k) + ".sdf"),
                Path(database_path) / "raw" / "valid_{:02d}".format(i) / "sdf" / ("atom_" + j) / (str(k) + ".sdf"),
            )


# # Dev and Test set

# In[8]:


database_crop_rate = 0.01

for j in ["9", "10", "11", "12"]:
    makedirs(Path(database_path) / "raw" / "dev" / "sdf" / ("atom_" + j))


# In[10]:


for j in ["9", "10", "11", "12"]:
    np.random.seed(0)
    copy_file_idx = np.array([int(i[:-4]) for i in os.listdir(Path(database_path) / "raw-orig" / "dev" / "sdf" / ("atom_" + j))])
    np.random.shuffle(copy_file_idx)
    copy_batch = int(copy_file_idx.size * database_crop_rate) + 1
    for k in copy_file_idx[:copy_batch]:
        shutil.copy(
            Path(database_path) / "raw-orig" / "dev" / "sdf" / ("atom_" + j) / (str(k) + ".sdf"),
            Path(database_path) / "raw" / "dev" / "sdf" / ("atom_" + j) / (str(k) + ".sdf"),
        )


# In[11]:


database_crop_rate = 0.01

for j in ["11", "12"]:
    makedirs(Path(database_path) / "raw" / "test" / "sdf" / ("atom_" + j))


# In[12]:


for j in ["11", "12"]:
    np.random.seed(0)
    copy_file_idx = np.array([int(i[:-4]) for i in os.listdir(Path(database_path) / "raw-orig" / "test" / "sdf" / ("atom_" + j))])
    np.random.shuffle(copy_file_idx)
    copy_batch = int(copy_file_idx.size * database_crop_rate) + 1
    for k in copy_file_idx[:copy_batch]:
        shutil.copy(
            Path(database_path) / "raw-orig" / "test" / "sdf" / ("atom_" + j) / (str(k) + ".sdf"),
            Path(database_path) / "raw" / "test" / "sdf" / ("atom_" + j) / (str(k) + ".sdf"),
        )

