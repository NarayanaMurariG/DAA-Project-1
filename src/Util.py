import os
import shutil
import numpy
import matplotlib.pyplot as plt

def clear_state_files():
    currentDirectory = os.getcwd()
    path = os.path.join(currentDirectory, 'state_files')
    isExist = os.path.exists(path)
    if isExist:
        shutil.rmtree(path)

    os.mkdir(path)

    intermediate_images_path = os.path.join(currentDirectory, 'image_files')
    isExist = os.path.exists(intermediate_images_path)
    if isExist:
        shutil.rmtree(intermediate_images_path)

    os.mkdir(intermediate_images_path)

def save_graph(graph, x, y):
    currentDirectory = os.getcwd()
    identifier = '('+str(x)+','+str(y)+')'
    path = os.path.join(currentDirectory,'state_files',identifier)
    numpy.savetxt(path+'.csv', graph, delimiter=",")

def save_image(image,x,y):
    plt.imshow(image)
    currentDirectory = os.getcwd()
    identifier = '(' + str(x) + ',' + str(y) + ').png'
    path = os.path.join(currentDirectory, 'state_files', identifier)
    plt.savefig(path)

def save_plots(image,str):
    currentDirectory = os.getcwd()
    identifier = str+'.png'
    path = os.path.join(currentDirectory, 'image_files', identifier)
    plt.imshow(image)
    plt.savefig(path)
    pass