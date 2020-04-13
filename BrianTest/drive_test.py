from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

g_login = GoogleAuth()
g_login.LocalWebserverAuth()
drive = GoogleDrive(g_login)

# with open("../Data/H5/512by512Kitti/mini_batch_0","r") as file:
#     #do something here with file
file_drive = drive.CreateFile({'title':os.path.basename(file.name) })  
file_drive.SetContentFile("../Data/H5/512by512Kitti/mini_batch_0") 
file_drive.Upload()