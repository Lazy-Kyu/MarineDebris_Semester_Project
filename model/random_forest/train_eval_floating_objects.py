# -*- coding: utf-8 -*-
'''
Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Python Version: 3.7.10
Description: train_eval.py includes the training and evaluation process for the
             pixel-level semantic segmentation with random forest.
'''

import os
import ast
import sys
import time
import json
import random
import logging
import rasterio
import argparse
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import display
from joblib import dump, load
from os.path import dirname as up


sys.path.append(up(os.path.abspath(__file__)))
from random_forest import rf_classifier, bands_mean

sys.path.append(os.path.join(up(up(up(os.path.abspath(__file__)))), 'utils'))
from assets import conf_mapping, cat_mapping_vec

random.seed(0)
np.random.seed(0)

root_path = up(up(up(os.path.abspath(__file__))))

logging.basicConfig(filename=os.path.join(root_path, 'logs','evaluation_rf.log'), filemode='a',level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logging.info('*'*10)


###############################################################
# Training                                                    #
###############################################################

def main(options):

    # Rewrite rf_features arrays to only use bands, real one must be imported from assets.py
    rf_features = ['nm440','nm490','nm560','nm665','nm705','nm740','nm783','nm842',
            'nm865','nm1600','nm2200']
    
    # Load Spectral Signatures
    hdf_ss = pd.HDFStore(os.path.join(options['path'], 'dataset.h5'), mode = 'r')
    df_train = hdf_ss.select('train')
    hdf_ss.close()

    print('df_train columns : ', df_train.columns)
    
    # Calculate weights for each sample on Train/ Val splits based on Confidence Level
    df_train['Weight'] = 1/df_train['Confidence'].apply(lambda x: conf_mapping[x])
    
    # Aggregate classes to Water Super class
    for agg_class in options['agg_to_water']:
        df_train.loc[df_train['Class'] == agg_class, 'Class'] = 'Marine Water'

    # Keep selected features and transform to numpy array
    X_train = df_train[rf_features].values
    # X_train = (X_train - np.mean(X_train))/np.std(X_train) ###
    y_train = df_train['Class'].values
    weight_train = df_train['Weight'].values
    
    print('Number of Input features: ', X_train.shape[1])
    print('Train: ',X_train.shape[0])
    
    print('Training X shape: ',X_train.shape)
    print('Training y shape: ',y_train.shape)

    logging.info('Number of Input features: ' + str(X_train.shape[1]))
    logging.info('Train: ' + str(X_train.shape[0]))

    # Training
    print('Started training')
    logging.info('Started training')
    
    start_time = time.time()
    rf_classifier.fit(X_train, y_train, **dict(rf__sample_weight=weight_train))
    
    print("Training finished after %s seconds" % (time.time() - start_time))
    logging.info("Training finished after %s seconds" % (time.time() - start_time))
    
    cl_path = os.path.join(up(os.path.abspath(__file__)), 'rf_classifier.joblib')
    dump(rf_classifier, cl_path)
    print("Classifier is saved at: " +str(cl_path))
    logging.info("Classifier is saved at: " +str(cl_path))


    # cl_path = os.path.join(up(os.path.abspath(__file__)), 'rf_classifier.joblib')
    # rf_classifier = load(cl_path)
    
    # Testing on floating object image
    path = os.path.join(options['path'], 'floating_objects_subset')
    
    location_list = ['accra_20181031', 'kentpointfarm_20180710', 'kolkata_20201115']
    file_name_flo_obj = location_list[0]

    flo_obj_file = os.path.join(path, file_name_flo_obj + '.tif')     # Get File path

    os.makedirs(options['gen_masks_path'], exist_ok=True)

    output_image = os.path.join(options['gen_masks_path'], os.path.basename(flo_obj_file).split('.tif')[0] + '_rf.tif')
    
    # Load the image patch and metadata
    with rasterio.open(flo_obj_file, mode ='r') as src:
        tags = src.tags().copy()
        meta = src.meta
        image = src.read()
        image = np.moveaxis(image, (0, 1, 2), (2, 0, 1))
        dtype = src.read(1).dtype
        
    # Update meta to reflect the number of layers
    meta.update(count = 1)
    
    sz1 = image.shape[0]
    sz2 = image.shape[1]

    # Remove bands 9 and 10 like MARIDA images, c.f bands at https://hatarilabs.com/ih-en/how-many-spectral-bands-have-the-sentinel-2-images
    image = np.delete(image, 9, 2)
    image = np.delete(image, 10, 2)

    # Scaling verification
    # for band in np.arange(11):
    #     train_band_slice = X_train[:, band]
    #     print('Training band {} maximum is {}'.format(band, np.max(train_band_slice)))
    #     print('Training band {} minimum is {}'.format(band, np.min(train_band_slice)))
    #     train_range = np.max(train_band_slice) - np.min(train_band_slice)
        
    #     test_band_slice = image[:, :, band]
    #     test_band_slice = np.squeeze(test_band_slice)
    #     test_range = np.max(test_band_slice) - np.min(test_band_slice)
        
    #     test_band_slice = test_band_slice*(train_range/test_range)

    #     # test_band_slice = test_band_slice - np.min(test_band_slice) + np.min(train_band_slice)

    #     train_band_mean = np.mean(train_band_slice)
    #     train_std = np.std(train_band_slice)
    #     test_band_mean = np.mean(test_band_slice)
    #     test_std = np.std(test_band_slice)

    #     # test_band_slice = test_band_mean + (test_band_slice - test_band_mean)*(train_std/test_std)
    #     test_band_slice = test_band_slice*(train_std/test_std)
    #     print('Test band {} maximum is {}'.format(band, np.max(test_band_slice)))
    #     print('Test band {} minimum is {}'.format(band, np.min(test_band_slice)))

    #     image[:, :, band] = test_band_slice

    # image = (image - np.mean(image))/np.std(image)

    train_range = np.max(X_train) - np.min(X_train)
    test_range = np.max(image) - np.min(image)
    image = image*(train_range/test_range)
    
    # train_mean = np.mean(X_train)
    # train_std = np.std(X_train)
    # test_mean = np.mean(image)
    # test_std = np.std(image)
    # image = (image - test_mean)/test_std
    # image = test_mean + (image - test_mean)*(train_std/test_std)
    # image = image*(train_std/test_std)
        
    # Preprocessing
    # Fill image nan with mean
    impute_nan = np.tile(bands_mean, (sz1,sz2,1))
    nan_mask = np.isnan(image)
    image[nan_mask] = impute_nan[nan_mask]
    
    image_features = np.reshape(image, (sz1*sz2, -1))
    print('Image features shape: ',image_features.shape)

    # Write it
    with rasterio.open(output_image, 'w', **meta) as dst:
        
        # use classifier to predict labels for the whole image

        if True:
            image_features = np.array_split(image_features, indices_or_sections = 256, axis = 0)
            predictions_list = []
            for iter in tqdm(image_features): 
                prediction = rf_classifier.predict(iter)
                # print('Prediction shape: ', prediction.shape)
                print(prediction)
                predictions_list.append(prediction)
            
            predictions_list = np.hstack(predictions_list)
            print('Prediction list shape: ', predictions_list.shape)
            predicted_labels = np.reshape(predictions_list, (sz1,sz2))
            print('Predicted labels shape: ', predicted_labels.shape)
        else:
            predictions = rf_classifier.predict(image_features)  
            predicted_labels = np.reshape(predictions, (sz1,sz2))
            print(predicted_labels)

        class_ind = cat_mapping_vec(predicted_labels).astype(dtype).copy()
        dst.write_band(1, class_ind) # In order to be in the same dtype

        dst.update_tags(**tags)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Options

    # Evaluation/Checkpointing
    parser.add_argument('--path', default=os.path.join(root_path, 'data'), help='Path to dataset')

    # Produce Predicted Masks
    parser.add_argument('--eval_set', default='test', type=str, help="Set for the evaluation 'val' or 'test' for final testing")
    parser.add_argument('--predict_masks', default= True, type=bool, help='Generate test set prediction masks?')
    parser.add_argument('--gen_masks_path', default=os.path.join(root_path, 'data', 'predicted_rf'), help='Path to where to produce store predictions')

    parser.add_argument('--agg_to_water', default='["Mixed Water", "Wakes", "Cloud Shadows", "Waves"]', type=str, help='Specify the Classes that will aggregate with Marine Water')
    
    args = parser.parse_args()
    options = vars(args)  # convert to ordinary dict
    
    # agg_to_water list fix
    agg_to_water = ast.literal_eval(options['agg_to_water'])
    if type(agg_to_water) is list:
        pass
    elif type(agg_to_water) is str:
        agg_to_water = [agg_to_water]
    else:
        raise
        
    options['agg_to_water'] = agg_to_water
    
    logging.info('parsed input parameters:')
    logging.info(json.dumps(options, indent = 2))
    main(options)
