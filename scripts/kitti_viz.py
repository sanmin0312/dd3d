import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def main():

    first = "2022-08-04"
    second = "16-24-12"
    third = "Results_20000_TTA rev"


    # data_dir = os.path.join(os.getcwd(), "outputs", date, model_name, "Result-tta", "inference", "final-tta", "kitti_3d_val")
    data_dir = os.path.join(os.getcwd(), "outputs", first, second, "inference", "final-tta", "kitti_3d_val")

    vizfile = os.path.join(data_dir, "visualization.npz")

    vizfile = np.load(vizfile)

    lst = vizfile.files

    for f in lst:
        index = f[:6]
        save_dir = os.path.join(data_dir, "visualization", index)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        image = vizfile[f]
        # image = cv2.imread(image)

        image_filename = f[7:]
        plt.imsave('{}/{}.jpg'.format(save_dir, image_filename), image)




if __name__ == '__main__':
    main()