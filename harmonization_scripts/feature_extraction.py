import sys
sys.path.append("/home/sushen/marine_debris_semester_project")
import model.random_forest.engineering_patches as eng
import numpy as np
import skimage.color
import skimage
from skimage import feature
from skimage.exposure import equalize_hist


# https://hatarilabs.com/ih-en/how-many-spectral-bands-have-the-sentinel-2-images
L2CBANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
L2ABANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
marida_bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
marida_band_idxs = np.array([L2ABANDS.index(b) for b in marida_bands])

# eng functions use Bands number based on 
# 1:nm440  2:nm490  3:nm560  4:nm665  5:nm705  6:nm740  7:nm783  8:nm842  9:nm865  10:nm1600  11:nm2200
# B1       B2       B3       B4       B5       B6       B7       B8       B8A      B11        B12      

def rgb(scene):
    return equalize_hist(scene[np.array([3,2,1])])

def extract_feature_image(img):
    indices = calculate_indices(img)
    texture = calculate_texture(img)
    return indices, texture

def calculate_indices(img):
    # Must create a buffer layer to make the layer index value match
    # im_size = img.shape
    # buffer_layer = np.zeros((1, im_size[1], im_size[2]))
    # img = np.concatenate((buffer_layer, img), axis = 0)

    NDVI = eng.ndvi(img[3], img[7])
    FAI = eng.fai(img[3], img[7], img[9])
    FDI = eng.fdi(img[5], img[7], img[9])
    SI = eng.si(img[1], img[2], img[3])
    NDWI = eng.ndwi(img[2], img[7])
    NRD = eng.nrd(img[3], img[7])
    NDMI = eng.ndmi(img[7], img[9])
    BSI = eng.bsi(img[1], img[3], img[7], img[9])

    return np.stack([NDVI, FAI, FDI, SI, NDWI, NRD, NDMI, BSI])

def calculate_texture(img, window_size = 13, max_value = 16):

    rgb_image = rgb(img)
    gray = skimage.color.rgb2gray(rgb_image.transpose(1,2,0))

    bins = np.linspace(0.00, 1.00, max_value)
    num_levels = max_value + 1

    temp_gray = np.pad(gray, (window_size - 1) // 2, mode='reflect')

    features_results = np.zeros((gray.shape[0], gray.shape[1], 6), dtype=temp_gray.dtype)

    for col in range((window_size - 1) // 2, gray.shape[0] + (window_size - 1) // 2):
        for row in range((window_size - 1) // 2, gray.shape[0] + (window_size - 1) // 2):
            temp_gray_window = temp_gray[row - (window_size - 1) // 2: row + (window_size - 1) // 2 + 1,
                               col - (window_size - 1) // 2: col + (window_size - 1) // 2 + 1]

            inds = np.digitize(temp_gray_window, bins)

            # Calculate on E, NE, N, NW as well as symmetric. So calculation on all directions and with 1 pixel offset-distance
            matrix_coocurrence = skimage.feature.graycomatrix(inds, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=num_levels,
                                              normed=True, symmetric=True)

            # Aggregate all directions
            matrix_coocurrence = matrix_coocurrence.mean(3)[:, :, :, np.newaxis]

            con, dis, homo, ener, cor, asm = eng.glcm_feature(matrix_coocurrence)
            features_results[row - (window_size - 1) // 2, col - (window_size - 1) // 2, 0] = con
            features_results[row - (window_size - 1) // 2, col - (window_size - 1) // 2, 1] = dis
            features_results[row - (window_size - 1) // 2, col - (window_size - 1) // 2, 2] = homo
            features_results[row - (window_size - 1) // 2, col - (window_size - 1) // 2, 3] = ener
            features_results[row - (window_size - 1) // 2, col - (window_size - 1) // 2, 4] = cor
            features_results[row - (window_size - 1) // 2, col - (window_size - 1) // 2, 5] = asm

    return np.stack(features_results).transpose(2,0,1)

