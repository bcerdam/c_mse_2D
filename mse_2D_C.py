import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from PIL import Image
import time

def plot_arrays(data_list, title='', xlabel='', ylabel='', legends=None, save_path=None):
    fig, ax = plt.subplots()
    plt.grid('black')
    ax.set_facecolor('lightgrey')
    for i, data_tuple in enumerate(data_list):
        name, data = data_tuple
        if legends is not None and len(legends) == len(data_list):
            label = legends[i]
        else:
            label = name
        plt.plot(data, color=f'C{i}', marker='v', markersize=5, label=label, markeredgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def calculate_std(csv_path):
    unique_values = set()

    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for val in row:
                unique_values.add(float(val))

    std = np.std(list(unique_values))

    return std

def image_to_array(image_path):
    img = Image.open(image_path).convert('L')
    arr = np.array(img)
    return arr

def run_c_program(csv_path, scales, rows, cols, m, r):
    command = ['./mse_2D', csv_path, str(scales), str(rows), str(cols), str(m), str(r)]
    result = subprocess.run(command, stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').strip()
    n_values = list(output.split())
    n_values = [float(x) for x in n_values]
    return n_values

def mse_2D(folder_path, scales, m, r):
    mse_values = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        image_array = image_to_array(file_path)
        rows, cols = image_array.shape

        np.savetxt("output.csv", image_array, delimiter=",", fmt="%d")

        # Descomentar para medir tiempo
        print('Working on: ', filename)
        start_time = time.time()

        mse_values.append([filename, run_c_program('output.csv', scales, rows, cols, m, r*calculate_std('output.csv'))])

        end_time = time.time()
        execution_time = end_time - start_time
        print(execution_time)

    return mse_values

# Ejemplo:

# '/home/bcm/Desktop/Repo/mse_c/datos/2D/500x500' => Hay que pasarle el path de la carpeta en donde estan las imagenes
# 20 => escalas
# 2 => m
# 0.25 => r

# Tambien imprime el tiempo en segundos que se demora.

# En el ejemplo de abajo son imagenes chicas de 100x100, por eso hay que ponerle m=1 y r=0.5, pero con imagenes mas grandes la idea es ocupar m=2 y r=0.25

# Importante, tienes que cambiar el path de las imagenes al de tu computador y tienes que compilar de nuevo
# Para compilar ocupa este comando: gcc mse_2D.c -o mse_2D -lm (En la terminal de la carpeta del repo)

# v = mse_2D('/home/bcm/Desktop/Repo/c_mse_2D/datos/2D/100x100', 20, 1, 0.5)
# plot_arrays(v)
