import numpy as np
import os
import functools
import time
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm
from allensdk.core.reference_space import ReferenceSpace
from allensdk.core.reference_space_cache import ReferenceSpaceCache

def open_AllenSDK(reference_space_key='annotation/ccf_2017', resolution=25):
    # Opens every variable necessary for analysis. 
    # __reference_space_key: key name reference to open a certain annotation (see documentation at allensdk Downloading an annotation volume for other annotations)
    # __resolution: resolution of slices in microns (default is 25)
    # rspc: opens the cache from which tree and annotation will be extracted
    # annotation, meta: downloads the annotation volume to your hard drive if not already done
    # os.listdir is a command that prints the directory in which it is now installed
    # rsp: gets reference space
    # name_map: dictionary of Names to IDs

    reference_space_key = 'annotation/ccf_2017'
    resolution = 25 
    rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest='manifest.json')
    tree = rspc.get_structure_tree(structure_graph_id=1) 
    annotation, meta = rspc.get_annotation_volume()
    print(os.listdir(reference_space_key)) 
    rsp = rspc.get_reference_space()
    

    return rsp, tree



def map_generator(rsp, tree, show = 'no', structure = 'all'):
    # Creates a vertical projection of superficial IDs in structure ID map (from top to bottom)
    # __rsp: reference space from which the atlas data is taken (like Id maps and names and stuff)
    # __show: if it equals 'yes', it plots the ID_map (WARNING: some IDs are so high they overshadow every other)
    # __structure: if it equals 'all', every region is kept in id_map. Else, structure takes as value a string of the region name to keep and removes every 
    #              subregion not contained in structure. structure can be 'Cerebellum', 'Isocortex', 'Olfactory areas', and much more (see 3D viewer brain map).

    
    y_dim = rsp.annotation.transpose([0,2,1]).shape[0]
    x_dim = rsp.annotation.transpose([0,2,1]).shape[1]
    z_dim = rsp.annotation.transpose([0,2,1]).shape[2]
    id_map = np.zeros((y_dim,x_dim)) 

    for slice in tqdm(range(z_dim)):
        image = np.squeeze(rsp.annotation.take([slice], axis=1)) #Image is structure id map
        
        np.copyto(id_map, image,where=id_map==0)

    id_list = np.unique(id_map)[1:].astype(int)

    if structure != 'all':

        id_compare = tree.get_structures_by_name([structure])[0].get('id')
        remove_id = []
        keep_id = []
        for i in id_list:
            if not tree.structure_descends_from(i, id_compare):
                remove_id.append(i)
            else:
                keep_id.append(i)

        for i in remove_id:
            id_map = np.where(id_map==i, 0, id_map)

        
    name_map = tree.get_name_map()
    id_name_dict = {}
    for i in keep_id:
        id_name_dict[i] = name_map[i]


    if show == 'yes':    
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.imshow(id_map, interpolation = 'none',vmax=1300)
        plt.show()

    hardcoded_bregma = (218,228)

    return id_map, id_name_dict, hardcoded_bregma


def create_mask(id_map, i):
    if type(i) == list:
        mask = np.zeros(id_map.shape)
        for j in i:
            mask += np.where(id_map==j, 1, 0)
        return mask
    return np.where(id_map==i, 1, 0)


def create_contour(structure): #Contours in horizontal (h) and vertical (v) planes. contours are sum of both, and returns boolean (normalized to 1).
    contours_h = abs(np.diff(structure))
    contours_v = abs(np.diff(structure, axis=0))
    contour_h = np.concatenate((contours_h, np.zeros((len(contours_h),1))),axis=1)
    contour_v = np.concatenate((contours_v, np.zeros((1,len(contours_v[0])))))
    contours = contour_h + contour_v
    return np.where(contours!=0, 1, 0)

                        
                        


#rsp, tree = open_AllenSDK()
#isocortex_map, id_name_dict, bregma = map_generator(rsp, tree, structure='Isocortex')
#fig, ax = plt.subplots(figsize=(10, 10))
#sma_mask = create_mask(isocortex_map, 656)
#contour = create_contour(isocortex_map)
#plt.imshow(isocortex_map,vmax=1300, cmap='gray', alpha=0.5)
#plt.imshow(contour, cmap='binary_r')
#plt.show()

#mask_list = []
#for i in id_name_dict:
#    mask_list.append(i)

#newmask = create_mask(isocortex_map, mask_list)
#plt.imshow(newmask, cmap='binary_r')
#plt.show()

