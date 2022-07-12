import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def main():

    date = "2022-04-16"
    model_name = "DD3D_dla34_KITTI"



    # data_dir = os.path.join(os.getcwd(), "outputs", date, model_name, "Result-tta", "inference", "final-tta", "kitti_3d_val")
    data_dir = os.path.join(os.getcwd(), "outputs", date, model_name, "inference", "final-tta", "kitti_3d_val")

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