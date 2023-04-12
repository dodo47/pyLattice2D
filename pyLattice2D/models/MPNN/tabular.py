import numpy as np
import matplotlib.image as mpimg
from pyLattice2D.utils.record import RecordAndSave
import os
from tqdm import tqdm

def load_image_data(path, shapes):
    images = load_images(path, shapes)
    stiffness, poisson, _ =  load_stiffness_and_density(path, shapes)
    return images, stiffness, poisson

def load_images(path, shapes):
    images = []
    for shape in shapes:
        num_samples = int(len(os.listdir('{}/images/{}'.format(path, shape)))/2)
        for i in tqdm(range(num_samples)):
            img = mpimg.imread('{}/images/{}/{}.png'.format(path, shape,i))
            images.append((img[24:-26,28:-28,2]<.85)*1.)

    return images

def load_image_metrics(path, shapes):
    img_mean, img_std = [], []
    imgi_std = []
    for shape in shapes:
        num_samples = int(len(os.listdir('{}/images/{}'.format(path, shape)))/2)
        for i in tqdm(range(num_samples)):
            img = mpimg.imread('{}/images/{}/{}.png'.format(path, shape,i))
            img = (img[24:-26,28:-28,2]<.85)*1.
            img_mean.append(np.mean(img))
            img_std.append(np.std(img))
            imgi_std.append(np.std(1-np.array(img)))

    return img_mean, img_std, imgi_std

def get_number_of_cells(path, shapes):
    num_cells = []
    for shape in shapes:
        loader = RecordAndSave(path, shape)
        loader.load()
        coords = loader['Coordinates']
        edges = loader['Edges']
        for i in range(len(coords)):
            num_cells.append(len(edges[i])-len(coords[i])+1)
    return num_cells

def get_edge_length(path, shapes):
    edge_mean = []
    edge_std = []
    edge_min = []
    edge_max = []
    for shape in shapes:
        loader = RecordAndSave(path, shape)
        loader.load()
        coords = loader['Coordinates']
        edges = loader['Edges']

        for k in tqdm(range(len(coords))):
            counts = []
            for i,j in edges[k]:
                counts.append(np.sqrt(np.sum((np.array(coords[k][i])-np.array(coords[k][j]))**2)))
            edge_mean.append(np.mean(counts))
            edge_std.append(np.std(counts))
            edge_max.append(np.max(counts))
            edge_min.append(np.min(counts))
            
    return edge_mean, edge_std, edge_min, edge_max

def load_stiffness_and_density(path, shapes):
    stiff = []
    dens = []
    pois = []
    for shape in shapes:
        loader = RecordAndSave(path, shape)
        loader.load()
        stiff += loader['Stiff']
        dens += loader['Density']
        pois += loader['Poisson']
    return stiff, pois, dens
        
def extract_tabular_data(path, shapes):
    stiff, pois, density = load_stiffness_and_density(path, shapes)
    img_mean, img_std, imgi_std = load_image_metrics(path, shapes)
    num_cells = get_number_of_cells(path, shapes)
    edge_mean, edge_std, edge_min, edge_max = get_edge_length(path, shapes)

    stiffness = []
    poisson = []
    tabular_features = []
    for i in tqdm(range(len(stiff))):
        if stiff[i] > 0 and ~np.isnan(pois[i]):
            stiffness.append(stiff[i])
            poisson.append(pois[i])
            tabular_features.append([img_mean[i],
                                    img_std[i],
                                    density[i],
                                    edge_mean[i],
                                    edge_std[i],
                                    edge_min[i],
                                    edge_max[i],
                                    (1-img_mean[i])/num_cells[i],
                                    imgi_std[i]/num_cells[i],
                                    ])
    return stiffness, poisson, tabular_features  

def load_tabular_data(original_path, tabular_path, shapes):
    stiff = []
    pois = []
    features = []
    for shape in shapes:
        loader = RecordAndSave(original_path, shape)
        loader.load()
        stiff += loader['Stiff']
        pois += loader['Poisson']
        features += np.load('{}/{}.npy'.format(tabular_path, shape)).tolist()
    return stiff, pois, features