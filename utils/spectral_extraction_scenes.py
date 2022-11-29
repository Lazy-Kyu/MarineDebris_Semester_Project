# -*- coding: utf-8 -*-
'''
Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Python Version: 3.7.10
Description: spectral_extraction.py extraction of the spectral signature, indices or texture features
             in a hdf5 table format for analysis and for the pixel-level semantic segmentation with 
             random forest classifier.
'''

import os
import sys
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from osgeo import gdal
from os.path import dirname as up

root_path = up(up(os.path.abspath(__file__)))
data_path = '/data/sushen/marinedebris/MARIDA'

# sys.path.append(os.path.join(root_path, 'utils'))
from assets_marida import s2_mapping, cat_mapping, conf_mapping, indexes_mapping, texture_mapping

rev_cat_mapping = {v:k for k,v in cat_mapping.items()}
rev_conf_mapping = {v:k for k,v in conf_mapping.items()}

def ImageToDataframe(RefImage, RefScene, cols_mapping = {}, keep_annotated = True, coordinates = True):
    # This function transform an image with the associated class and 
    # confidence tif files (_cl.tif and _conf.tif) to a dataframe

    # Read scene
    ds_scene = gdal.Open(RefScene)
    IM_scene = np.copy(ds_scene.ReadAsArray()) 

    # Read patch (Shape 11 x 256 x 256)
    ds = gdal.Open(RefImage)
    IM = np.copy(ds.ReadAsArray())

    # Read associated confidence level patch (Shape 1 x 256 x 256)
    ds_conf = gdal.Open(os.path.join(up(RefImage), '_'.join(os.path.basename(RefImage).split('.tif')[0].split('_')[:4]) + '_conf.tif'))
    IM_conf = np.copy(ds_conf.ReadAsArray())[np.newaxis, :, :]
    
    # Read associated class patch (Shape 1 x 256 x 256)
    ds_cl = gdal.Open(os.path.join(up(RefImage), '_'.join(os.path.basename(RefImage).split('.tif')[0].split('_')[:4]) + '_cl.tif'))
    IM_cl = np.copy(ds_cl.ReadAsArray())[np.newaxis, :, :]
    
    # Stack all these together and move axis (Shape 256 x 256 x 13)
    IM_T = np.moveaxis(np.concatenate([IM, IM_conf, IM_cl], axis = 0), 0, -1)
    
    if coordinates:
        # Get the coordinates in space.
        padfTransform = ds.GetGeoTransform()
        
        y_coords, x_coords = np.meshgrid(range(IM_T.shape[0]), range(IM_T.shape[1]), indexing='ij')
        
        Xp = padfTransform[0] + x_coords*padfTransform[1] + y_coords*padfTransform[2]
        Yp = padfTransform[3] + x_coords*padfTransform[4] + y_coords*padfTransform[5]
        
        # shift to the center of the pixel
        Xp -= padfTransform[5] / 2.0
        Yp -= padfTransform[1] / 2.0
        XpYp = np.dstack((Xp,Yp))
        IM_T = np.concatenate((IM_T, XpYp), axis=2)
        
    bands = IM_T.shape[-1]
    IM_VECT = IM_T.reshape([-1,bands])
    
    if keep_annotated and coordinates:
        IM_VECT = IM_VECT[IM_VECT[:,-3] > 0] # Keep only based on non zero class
    elif keep_annotated and not coordinates:
        IM_VECT = IM_VECT[IM_VECT[:,-1] > 0] # Keep only based on non zero class
        
    if cols_mapping:
        IM_df = pd.DataFrame({k:IM_VECT[:,v] for k, v in cols_mapping.items()})
    else:
        IM_df = pd.DataFrame(IM_VECT)
        
    if coordinates:
        IM_df['XCoords'] = IM_VECT[:,-2]
        IM_df['YCoords'] = IM_VECT[:,-1]
    
    IM_df.date = ds.GetMetadataItem("TIFFTAG_DATETIME")
    ds = None
    ds_conf = None
    ds_cl = None
    
    IM_df["Class"] = IM_df["Class"].apply(lambda x: rev_cat_mapping[x])
    IM_df['Confidence'] = IM_df['Confidence'].apply(lambda x: rev_conf_mapping[x])
    
    return IM_df

# function copied from : https://sciience.tumblr.com/post/101722591382/finding-the-georeferenced-intersection-between-two
def findRasterIntersect(raster1,raster2):
    # load data
    # band1 = raster1.GetRasterBand(1)
    # band2 = raster2.GetRasterBand(1)
    gt1 = raster1.GetGeoTransform()
    gt2 = raster2.GetGeoTransform()
    
    # find each image's bounding box
    # r1 has left, top, right, bottom of dataset's bounds in geospatial coordinates.
    r1 = [gt1[0], gt1[3], gt1[0] + (gt1[1] * raster1.RasterXSize), gt1[3] + (gt1[5] * raster1.RasterYSize)]
    r2 = [gt2[0], gt2[3], gt2[0] + (gt2[1] * raster2.RasterXSize), gt2[3] + (gt2[5] * raster2.RasterYSize)]
    print('\t1 bounding box: %s' % str(r1))
    print('\t2 bounding box: %s' % str(r2))
    
    # find intersection between bounding boxes
    intersection = [max(r1[0], r2[0]), min(r1[1], r2[1]), min(r1[2], r2[2]), max(r1[3], r2[3])]
    if r1 != r2:
        print('\t** different bounding boxes **')
        # check for any overlap at all...
        if (intersection[2] < intersection[0]) or (intersection[1] < intersection[3]):
            intersection = None
            print('\t***no overlap***')
            return
        else:
            print('\tintersection:',intersection)
            # Calculate offsets for ReadAsArray
            left1 = int(round((intersection[0]-r1[0])/gt1[1])) # difference divided by pixel dimension
            top1 = int(round((intersection[1]-r1[1])/gt1[5]))
            col1 = int(round((intersection[2]-r1[0])/gt1[1])) - left1 # difference minus offset left
            row1 = int(round((intersection[3]-r1[1])/gt1[5])) - top1
            
            left2 = int(round((intersection[0]-r2[0])/gt2[1])) # difference divided by pixel dimension
            top2 = int(round((intersection[1]-r2[1])/gt2[5]))
            col2 = int(round((intersection[2]-r2[0])/gt2[1])) - left2 # difference minus new left offset
            row2 = int(round((intersection[3]-r2[1])/gt2[5])) - top2
            
            #print '\tcol1:',col1,'row1:',row1,'col2:',col2,'row2:',row2
            if col1 != col2 or row1 != row2:
                print("*** MEGA ERROR *** COLS and ROWS DO NOT MATCH ***")
            # these arrays should now have the same spatial geometry though NaNs may differ
            array1 = raster1.ReadAsArray(left1,top1,col1,row1)
            array2 = raster2.ReadAsArray(left2,top2,col2,row2)

    else: # same dimensions from the get go
        col1 = raster1.RasterXSize # = col2
        row1 = raster1.RasterYSize # = row2
        array1 = raster1.ReadAsArray()
        array2 = raster2.ReadAsArray()
        
    return array1, array2, col1, row1, intersection

def main(options):
    
    # Which features?
    if options['type']=='s2':
        mapping = s2_mapping
        h5_prefix = 'dataset_scenes'
        
        # Get patches files without _cl and _conf associated files
        patches = glob(os.path.join(options['path'], 'patches', '*/*.tif'))
        
    elif options['type']=='indices':
        mapping = indexes_mapping
        h5_prefix = 'dataset_scenes_si'
        
        # Get patches files without _cl and _conf associated files
        patches = glob(os.path.join(options['path'], 'indices', '*/*.tif'))
        
    elif options['type']=='texture':
        mapping = texture_mapping
        h5_prefix = 'dataset_scenes_glcm'
        
        # Get patches files without _cl and _conf associated files
        patches = glob(os.path.join(options['path'], 'texture', '*/*.tif'))
        
    else:
        raise AssertionError("Wrong Type, select between s2, indices or texture")

    patches = [p for p in patches if ('_cl.tif' not in p) and ('_conf.tif' not in p)]

    # Read splits
    X_train = np.genfromtxt(os.path.join(options['path'], 'splits','train_X.txt'),dtype='str')
    
    X_val = np.genfromtxt(os.path.join(options['path'], 'splits','val_X.txt'),dtype='str')
    
    X_test = np.genfromtxt(os.path.join(options['path'], 'splits','test_X.txt'),dtype='str')

    # Load Marida scenes mapping csv
    # Getting rid of entries where description tile name in region column doesn't match tile name in tif file
    df_map_scenes = pd.read_csv(os.path.join(data_path,"marida_mapping.csv"))
    df_map_scenes = df_map_scenes[df_map_scenes['mod'].str.contains('SR')==False]
    df_map_scenes['tile'] = df_map_scenes['region'].apply(lambda x: x.split('_')[-1])
    df_map_scenes['tile_contained'] = df_map_scenes.apply(lambda x: x.tile in x.tifpath, axis=1)
    df_map_scenes = df_map_scenes[df_map_scenes['tile_contained']==True]
    # df_map_scenes.to_csv(os.path.join(data_path,"marida_mapping_fixed.csv"))
    
    # dataset_name = os.path.join(options['path'], h5_prefix + '_nonindex.h5')
    # hdf = pd.HDFStore(dataset_name, mode = 'w')
    
    # For each patch extract the spectral signatures and store them
    for im_name in tqdm(patches):

        # Get date_tile_image info
        img_name = '_'.join(os.path.basename(im_name).split('.tif')[0].split('_')[1:4]) 
        # im_name is the path of a tile such as /data/sushen/marinedebris/MARIDA/patches/S2_1-12-19_48MYU/S2_1-12-19_48MYU_0.tif
        # example of img_name is 1-12-19_48MYU_0
        date_tile = '_'.join(img_name.split('_')[0:2])
        scene_names = df_map_scenes.loc[df_map_scenes['region'].str.contains(date_tile)]['tifpath'].values
        if len(scene_names) == 1:
            scene_path = os.path.join(data_path, 'scenes', scene_names[0])
        else:
            raise AssertionError("Multiple .tif files correspond to date_tile information.") 
        print(scene_path)
        
        # Generate Dataframe from Image
        split = 'train'
        temp = ImageToDataframe(im_name, scene_path, mapping)
        
        # Update Satellite and Date info
        temp['Date'] = os.path.splitext(os.path.basename(im_name))[0].split('_')[1]
        temp['Tile'] = os.path.splitext(os.path.basename(im_name))[0].split('_')[2]
        temp['Image'] = os.path.splitext(os.path.basename(im_name))[0].split('_')[3]
        # Store data
        hdf.append(split, temp, format='table', data_columns=True, min_itemsize={'Class':27,
                                                                                 'Confidence':8,
                                                                                 'Date':8,
                                                                                 'Image':3,
                                                                                 'Tile':5})
    
    hdf.close()
    
    # Read the stored file and fix an indexing problem (indexes were not incremental and unique)
    hdf_old = pd.HDFStore(dataset_name, mode = 'r')

    df_marida_scenes = hdf_old['train'].copy(deep=True)
    df_marida_scenes.reset_index(drop = True, inplace = True)

    hdf_old.close()
    
    # Store the fixed table to a new dataset file
    dataset_name_fixed = os.path.join(options['path'], h5_prefix+'.h5')
    df_marida_scenes.to_hdf(dataset_name_fixed, key='train', mode='a', format='table', data_columns=True)
    
    os.remove(dataset_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--path', default=data_path, help='Path to dataset')
    parser.add_argument('--type', default='s2', type=str, help=' Select between s2, indices or texture for Spectral Signatures, Produced Indices or GLCM Textures, respectively')

    args = parser.parse_args()
    options = vars(args)  # convert to ordinary dict
    
    main(options)
