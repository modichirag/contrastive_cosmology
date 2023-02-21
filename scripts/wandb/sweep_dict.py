import numpy as np
import sys, os


datapath = f'/mnt/ceph/users/cmodi/contrastive/analysis/'

sims = ['quijote', 'fastpm']

# for sim in sims:
#     print(f"configs for simulation : {sim}")
#     os.system(f"ls {datapath}/{sim}/*/*/*/*")
#     print()

for sim in sims:
    print(f"configs for simulation : {sim}")
    os.system(f"find  {datapath}/{sim}/*/*/*/*/*/sweep*.yaml -maxdepth 1 -type f ")
    print()


