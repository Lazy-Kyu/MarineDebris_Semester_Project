import rasterio.windows
import geopandas as gpd
import os
import sys
import rasterio as rio
import pandas as pd
from rasterio.features import rasterize
import numpy as np
from tqdm import tqdm

from os.path import dirname as up
sys.path.append('/home/sushen/marine_debris_semester_project')
from data.utils_file import read_tif_image, pad

from feature_extraction import calculate_indices, calculate_texture
from data.utils_file import pad

def main():
    df_map_scenes = pd.read_csv('/data/sushen/marinedebris/MARIDA/marida_mapping.csv')
    df_map_scenes = df_map_scenes[df_map_scenes['mod'].str.contains('SR')==False]
    df_map_scenes['tile'] = df_map_scenes['region'].apply(lambda x: x.split('_')[-1])
    df_map_scenes['tile_contained'] = df_map_scenes.apply(lambda x: x.tile in x.tifpath, axis=1)
    df_map_scenes = df_map_scenes[df_map_scenes['tile_contained']==True]
    df_map_scenes.reset_index(drop=True, inplace=True) 
    df_map_scenes.drop([15], inplace=True) # Problematic row : 'S2_4-3-18_50LLR' data not available

    # Folder paths
    data_path = '/data/sushen/marinedebris/MARIDA'
    mask_id_path = '/data/sushen/marinedebris/project/marida_masks_id'
    mask_conf_path = '/data/sushen/marinedebris/project/marida_masks_conf'
    hdf_path = '/data/sushen/marinedebris/project/dataset_original_classes.h5'
    
    # HDF file open
    hdf = pd.HDFStore(hdf_path, mode = 'w')

    for i in tqdm(np.arange(len(df_map_scenes))):
        # Locate patches and their shapefiles from the mapping table
        scene_name = df_map_scenes.iloc[i]['tifpath']
        shp_name = df_map_scenes.iloc[i]['s2name']
        region_name = df_map_scenes.iloc[i]['region']

        # File paths
        tif_file_path = os.path.join(data_path, 'scenes', scene_name)
        shp_file_path = os.path.join(data_path, 'shapefiles', shp_name)
        mask_id_file_path = os.path.join(mask_id_path, region_name + ".tif")
        mask_conf_file_path = os.path.join(mask_conf_path, region_name + ".tif")

        # Geopandas Data Frame read
        gdf = gpd.read_file(shp_file_path)

        # Rasterio Image opening
        with rio.open(tif_file_path) as src:
            crs = src.crs
            width = src.width
            height = src.height
            transform = src.transform
            profile = src.profile

        gdf = gdf.to_crs(crs)

        # Rasterize geometry of shp into a mask with labels
        if not os.path.exists(mask_id_file_path):
            mask_id = rasterize(zip(gdf.geometry, gdf.id), all_touched=True,
                            transform=transform, out_shape=(height, width))

            profile["count"] = 1
            profile["dtype"] = "uint8"

            print(f"writing mask to {mask_id_file_path}")
            with rio.open(mask_id_file_path, "w", **profile) as dst:
                dst.write(mask_id[None])

        # Rasterize geometry of shp into a mask with conf levels
        if not os.path.exists(mask_conf_file_path):
            mask_conf = rasterize(zip(gdf.geometry, gdf.conf), all_touched=True,
                            transform=transform, out_shape=(height, width))

            profile["count"] = 1
            profile["dtype"] = "uint8"

            print(f"writing mask to {mask_conf_file_path}")
            with rio.open(mask_conf_file_path, "w", **profile) as dst:
                dst.write(mask_conf[None])

        imagesize = 16*10 # 16 pixels around centroid, 10m per pixel

        # Iterate of every geometry object and get the pixels
        for j in tqdm(np.arange(len(gdf))):
            row = gdf.iloc[j]
            minx, miny, maxx, maxy = row.geometry.centroid.buffer(imagesize // 2).bounds
            window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=transform)

            image, _ = read_tif_image(tif_file_path, window)
            image = image.astype("float")
            if image.size == 0:
                continue

            with rasterio.open(mask_id_file_path, "r") as src:
                mask_id = src.read(window=window)[0]
            if mask_id.size == 0:
                continue

            with rasterio.open(mask_conf_file_path, "r") as src:
                mask_conf = src.read(window=window)[0]
            if mask_conf.size == 0:
                continue

            # Protect images and mask with padding if an geometric object too close to the boundary
            image, mask_id = pad(image, mask_id, imagesize // 10)
            _, mask_conf = pad(image, mask_conf, imagesize // 10)
            
            # Deleting Band9 according to MARIDA paper since read_tif_image already removes B10
            image = np.delete(image, 9, axis = 0)

            # Calculate indices and textures
            indices = calculate_indices(image)
            textures = calculate_texture(image)

            # Moves axis & concatenate
            image = np.moveaxis(image, (0, 1, 2), (2, 0, 1))
            indices = np.moveaxis(indices, (0, 1, 2), (2, 0, 1))
            textures = np.moveaxis(textures, (0, 1, 2), (2, 0, 1))

            features = np.dstack((mask_id, mask_conf, image, indices, textures))
            sz1 = features.shape[0]
            sz2 = features.shape[1]

            # Reshape into a 2D vector and keep entries where target label is not 0
            features = np.reshape(features, (sz1*sz2, -1))
            features = features[features[:, 0] > 0]
            features.shape

            # Prepare temporary dataframe
            columns = ['Class','Conf',"B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", 
                "NDVI", "FAI", "FDI", "SI", "NDWI", "NRD", "NDMI", "BSI",
                "con", "dis", "homo", "ener", "cor", "asm"]
            temp_df = pd.DataFrame(features, columns = columns)

            # Append dataframe
            hdf.append('train', temp_df, format='table', data_columns=True)
    hdf.close()

if __name__ == "__main__":
    main()