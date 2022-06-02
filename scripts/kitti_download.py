import wget
import os
import zipfile

root_dir = '/home/user/data/Dataset'
output_dir = os.path.join(root_dir, "kitti_raw")

with open(os.path.join(root_dir, "kitti_3d", "mapping", "train_mapping.txt")) as _f:
    lines = _f.readlines()
split = [line.rstrip("\n")[11:-16] for line in lines]
split = set(split)


for date in split:
    print("\n{}".format(date))
    file_list = os.listdir(output_dir)

    if "{}_sync.zip".format(date) in file_list:
        continue
    url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{}/{}_sync.zip".format(date, date)
    file_path = os.path.join(output_dir, "{}_sync.zip".format(date))
    wget.download(url, file_path)

    file = zipfile.ZipFile(file_path)
    file.extractall(output_dir)
