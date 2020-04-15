from pathlib import Path
import os

directory_in_str = '../Data/H5/FlyingChairs2_del/'

num_deleted = 0
pathlist = Path(directory_in_str).glob('*')
for path in pathlist:
    # print(path)
    os.remove(path)
    num_deleted += 1

print("Num files deleted: ",num_deleted)
