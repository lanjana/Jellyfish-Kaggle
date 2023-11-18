import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os


def read_imgs(folder_path):

    output_vector, inp_vector = [], []

    for lable in os.listdir(path=folder_path):
        for img in os.listdir(path=folder_path + "/" + lable):
            img = cv2.imread(folder_path + "/" + lable + "/" + img)
            img = img/255.0

            inp_vector.append(img)
            output_vector.append(lable)

    unique_out = []
    for lable in output_vector:
        if lable not in unique_out:
            unique_out.append(lable)

    output_vector_final = []

    for lbl in output_vector:
        s = [0] * len(unique_out)
        s[unique_out.index(lbl)] = 1
        output_vector_final.append(s)

    return np.array(inp_vector), np.array(output_vector_final)


inp_vector, output_vector = read_imgs("./Datasets")


print(inp_vector.shape)
