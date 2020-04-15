from pathlib import Path
from PIL import Image
import os

directory_in_str = '../Data/FlyingChairs_release/data/'

pathlist = Path(directory_in_str).glob('*.ppm')
count_del = 0
count_files = 0

flo_end = '_flow.flo'
img1_end = '_img1.ppm'
img2_end = '_img2.ppm'


for path in pathlist:
    # because path is object not string
    path_in_str = str(path)
    #  print(path_in_str)
    dir = Path(os.path.dirname(path))
    s = os.path.basename(path)
    s = s.replace(flo_end, '')
    s = s.replace(img1_end, '')
    s = s.replace(img2_end, '')
    im0 = s + img1_end
    im1 = s + img2_end
    flo = s + flo_end
    try:
        Image.open(path)
        # print("opened")
        exist = os.path.exists(dir/im0) and os.path.exists(dir/im1) and os.path.exists(dir/flo)
        if(not exist):
            print('Missing File, deleting')
            if os.path.exists(dir/im0): 
                os.remove(dir/im0)
                print(dir/im0)
            if os.path.exists(dir/im1): 
                os.remove(dir/im1)
                print(dir/im1)
            if os.path.exists(dir/flo): 
                os.remove(dir/flo)
                print(dir/flo)
            count_del = count_del + 1
            continue
        else:
            flo_path = dir/flo
            f = open(flo_path, 'rb')

            header = f.read(4)
            if header.decode("utf-8") != 'PIEH':
                print('Flow file header does not contain PIEH, deleting')
                if os.path.exists(dir/im0): 
                    os.remove(dir/im0)
                    print(dir/im0)
                if os.path.exists(dir/im1): 
                    os.remove(dir/im1)
                    print(dir/im1)
                if os.path.exists(dir/flo): 
                    os.remove(dir/flo)
                    print(dir/flo)
                count_del = count_del + 1
                continue
        count_files = count_files + 1
        continue
    except Exception as inst:
        count_del = count_del + 1
        print(inst)
        print('\nBroken File, deleting...')
        if os.path.exists(dir/im0): 
            os.remove(dir/im0)
            print(dir/im0)
        if os.path.exists(dir/im1): 
            os.remove(dir/im1)
            print(dir/im1)
        if os.path.exists(dir/flo): 
            os.remove(dir/flo)
            print(dir/flo)
        print('\n')
    

# count_del_flo = 0
# pathlist_flo = Path(directory_in_str).glob('*.flo')
# for path in pathlist_flo:
#     print(path)
#     f = open(path, 'rb')

#     header = f.read(4)
#     if header.decode("utf-8") != 'PIEH':
#         print(path)
#         print('Flow file header does not contain PIEH, deleting')
#         dir = Path(os.path.dirname(path))
#         s = os.path.basename(path)
#         s = s.replace(flo_end, '')
#         im0 = s + img1_end
#         im1 = s + img2_end
#         os.remove(path)
#         os.remove(dir/im0)
#         os.remove(dir/im1)
#         count_del_flo = count_del_flo + 1



print("deleted ", count_del)
print("num files ", count_files)
print("num files ", count_files // 2)