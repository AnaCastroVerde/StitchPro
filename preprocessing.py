## ----------------------------------####### PREPROCESSING SCRIPT #######-------------------------------------------- ##
## - This script does preprocessing on histopathology fragments, before undergoing image sticthing and registration - ##
## ----------------------------- 1) Extracts downsampled image at a lower downsampling level ------------------------ ##
## ----------------------------- 2) Crops image to remove background outside slide ---------------------------------- ##
## ----------------------------- 3) Rotates the fragment to be aligned (optional) ----------------------------------- ##
## ----------------------------- 4) Divides the fragment into two quadrants (optional) ------------------------------ ##

## ------------------------------------------------ Import packages ------------------------------------------------- ##

from tiatoolbox.wsicore import WSIReader
from tiatoolbox.tools.tissuemask import MorphologicalMasker
from tiatoolbox.utils.misc import get_bounding_box
from pprint import pprint
import tifffile
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
import argparse
import cv2 as cv
import json
from typing import Tuple

## ------------------------------------------ Create preprocessing function ----------------------------------------- ##

def main(input_path: str,
         output_path: str,
         output_name: str,
         tiff_image: str,
         level: int,
         divide_horizontal: bool,
         divide_vertical: bool,
         show_image: bool):
    
    ## -------------------------------------- Read the original TIFF images ---------------------------------------- ##

    reader = WSIReader.open(input_path + tiff_image)
    info_dict = reader.info.as_dict()
    #pprint(info_dict)
    #original_size = info_dict['level_dimensions'][0]
    original_spacing = info_dict['mpp']
    wsi_thumb = reader.slide_thumbnail(
        resolution=level,
        units="level")
    
    reader_aux = WSIReader.open(input_path[:-1] + "_aux/" + tiff_image)

    def preds_to_image(coords: np.ndarray, probs: np.ndarray, dims: Tuple[int, int], original_dims: Tuple[int, int], tile_size: Tuple[int, int] = (256, 256), binarize: bool = True) -> np.ndarray:
        dims = np.array(dims)
        original_dims = np.array(original_dims)
        tile_size = np.array(tile_size)
        factor = original_dims / dims
        tile_size = np.int32(tile_size // factor)
        coords = np.int32(coords // factor)
        output_image = np.zeros(dims, dtype=np.uint8)
        for coord, prob in zip(coords, probs):
            if binarize is True:
                v = 255 if prob > 0.5 else 0
            else:
                v = int(prob * 255)
            output_image[
                coord[1] : coord[1] + tile_size[0] + 1,
                coord[0] : coord[0] + tile_size[1] + 1,
            ] = v
        return output_image

    def preds_json_to_image(json_file: str, dims: Tuple[int, int], binarize: bool = False) -> np.ndarray:
        with open(json_file) as o:
            data_dict = json.load(o)[0]
        ds = data_dict["downsample"]
        coords = np.array(data_dict["coordinates"]) // ds
        values = [x["probabilities"][1] for x in data_dict["values"]]
        dimensions = [x // ds for x in data_dict["dimensions"]][::-1]
        tile_size = data_dict["tile_size"]
        return preds_to_image(
            coords=coords,
            probs=values,
            dims=dims,
            original_dims=dimensions,
            tile_size=tile_size,
            binarize=binarize,
        )
    
    image_aux = preds_json_to_image(input_path[:-1] + "_aux/"+ tiff_image + ".json", dims=wsi_thumb.shape[:2], binarize=False)

    ## ------------------ Extract a downsampled region within bounds (to remove whitespace on slide) --------------- ##

    masker = MorphologicalMasker(power=1.25, min_region_size=500000)
    masks = masker.fit_transform([wsi_thumb])[0]

    if show_image == True:
        def show_side_by_side(image_1: np.ndarray, image_2: np.ndarray) -> None:
            plt.subplot(1, 2, 1)
            plt.imshow(image_1)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(image_2)
            plt.axis("off")
            plt.show()
        show_side_by_side(wsi_thumb, masks)
    
    start_x, start_y, end_x, end_y = get_bounding_box(masks)
    bounds = [start_x, start_y, end_x, end_y]
    region = reader.read_bounds(bounds, resolution=level, units="level", coord_space = "resolution")

    #region_aux = reader_aux.read_bounds(bounds, resolution=level, units="level", coord_space = "resolution")
    region_aux = image_aux[bounds[1] : bounds[3], bounds[0] : bounds[2]]
    region_aux = np.stack((region_aux, region_aux, region_aux), axis=2)

    ## ------------------------------- Extract angle to orient fragment to horizontal ---------------------------- ##
    
    binary_mask = masks.astype(np.uint8)*255
    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)
    rect = cv.minAreaRect(largest_contour)
    if rect[-1]<45:
        angle = round(rect[-1], 0)
    else:
        angle = 0

    ## ----------------------------- Small rotation angle to straighten up the fragment -------------------------- ##
    
    region_rotated = rotate(region, angle = angle, axes=(1, 0), reshape=True, mode = 'constant', cval=230.0)

    region_rotated_aux = rotate(region_aux, angle = angle, axes=(1, 0), reshape=True, mode = 'constant', cval=230.0)
    
    if show_image == True:
        fig, axs = plt.subplots(nrows=2, ncols=2)
        axs[0,0].imshow(region)
        axs[0,1].imshow(region_rotated)
        axs[1,0].imshow(region_aux)
        axs[1,1].imshow(region_rotated_aux)
        plt.show()

    new_spacing = (2**level)*original_spacing[0]

    ## ------------------------------- Divide the fragments (optional) before saving ---------------------------- ##
    M = region_rotated.shape[0]//2
    N = region_rotated.shape[1]//2

    if (divide_horizontal == True) & (divide_vertical == False):
        tiles = [region_rotated[:, y:y + N] for y in range(0, region_rotated.shape[1], N)]
        tiles_aux = [region_rotated_aux[:, y:y + N] for y in range(0, region_rotated_aux.shape[1], N)]
        fragment_1 = tiles[0]
        fragment_2 = tiles[1]
        fragment_1_aux = tiles_aux[0]
        fragment_2_aux = tiles_aux[1]
        if show_image == True:
            fig, axs = plt.subplots(nrows=1, ncols=2)
            axs[0].imshow(fragment_1, cmap="gray")
            axs[1].imshow(fragment_2, cmap="gray")
            plt.show()
        tifffile.imwrite(
            output_path + str(output_name) + "_fragment1.tiff",
            np.array(fragment_1), photometric='rgb', imagej=True, resolution=(1 / new_spacing, 1 / new_spacing),
            metadata={'spacing': new_spacing, 'unit': 'um'})
        tifffile.imwrite(
            output_path + str(output_name) + "_fragment2.tiff",
            np.array(fragment_2), photometric='rgb', imagej=True, resolution=(1 / new_spacing, 1 / new_spacing),
            metadata={'spacing': new_spacing, 'unit': 'um'})
        tifffile.imwrite(
            output_path + str(output_name) + "_aux_fragment1.tiff",
            np.array(fragment_1_aux), photometric='rgb', imagej=True, resolution=(1 / new_spacing, 1 / new_spacing),
            metadata={'spacing': new_spacing, 'unit': 'um'})
        tifffile.imwrite(
            output_path + str(output_name) + "_aux_fragment2.tiff",
            np.array(fragment_2_aux), photometric='rgb', imagej=True, resolution=(1 / new_spacing, 1 / new_spacing),
            metadata={'spacing': new_spacing, 'unit': 'um'})
    if (divide_vertical == True) & (divide_horizontal == False):
        tiles = [region_rotated[x:x + M, :] for x in range(0, region_rotated.shape[0], M)]
        tiles_aux = [region_rotated_aux[x:x + M, :] for x in range(0, region_rotated_aux.shape[0], M)]
        fragment_1 = tiles[0]
        fragment_2 = tiles[1]
        fragment_1_aux = tiles_aux[0]
        fragment_2_aux= tiles_aux[1]
        if show_image == True:
            fig, axs = plt.subplots(nrows=1, ncols=2)
            axs[0].imshow(fragment_1, cmap="gray")
            axs[1].imshow(fragment_2, cmap="gray")
            plt.show()
        tifffile.imwrite(
            output_path + str(output_name) + "_fragment1.tiff",
            np.array(fragment_1), photometric='rgb', imagej=True, resolution=(1 / new_spacing, 1 / new_spacing),
            metadata={'spacing': new_spacing, 'unit': 'um'})
        tifffile.imwrite(
            output_path + str(output_name) + "_fragment2.tiff",
            np.array(fragment_2), photometric='rgb', imagej=True, resolution=(1 / new_spacing, 1 / new_spacing),
            metadata={'spacing': new_spacing, 'unit': 'um'})
        tifffile.imwrite(
            output_path + str(output_name) + "_aux_fragment1.tiff",
            np.array(fragment_1_aux), photometric='rgb', imagej=True, resolution=(1 / new_spacing, 1 / new_spacing),
            metadata={'spacing': new_spacing, 'unit': 'um'})
        tifffile.imwrite(
            output_path + str(output_name) + "_aux_fragment2.tiff",
            np.array(fragment_2_aux), photometric='rgb', imagej=True, resolution=(1 / new_spacing, 1 / new_spacing),
            metadata={'spacing': new_spacing, 'unit': 'um'})
    if (divide_horizontal == False) & (divide_vertical == False):
        tifffile.imwrite(
            output_path + str(output_name) + ".tiff",
            np.array(region_rotated), photometric='rgb', imagej=True, resolution=(1 / new_spacing, 1 / new_spacing),
            metadata={'spacing': new_spacing, 'unit': 'um'})  # metadata={'spacing': 32, 'unit': 'um'}
        tifffile.imwrite(
            output_path + str(output_name) + "_aux.tiff",
            np.array(region_rotated_aux), photometric='rgb', imagej=True, resolution=(1 / new_spacing, 1 / new_spacing),
            metadata={'spacing': new_spacing, 'unit': 'um'})  # metadata={'spacing': 32, 'unit': 'um'}
        
## ------------------------------------------------ Parse arguments ------------------------------------------------- ##
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocessing histopathology images')

    parser.add_argument('--input_path', dest='input_path',required=True,
                        help='Path to folder containing the image.')
    parser.add_argument('--output_path', dest='output_path',required=True,
                            help='Path to store the preprocessing results.')
    parser.add_argument('--output_name', dest='output_name',required=True,
                            help='Name of the output preprocessed fragment.')
    parser.add_argument('--tiff_image', dest='tiff_image', required=True,
                        help='Histopathology image to preprocess.')
    parser.add_argument('--level', dest='level', required=True, type=int,
                        help='Resolution level to downsample.')
    parser.add_argument('--divide_horizontal', dest='divide_horizontal', action = 'store_true',
                        help='Create two fragments from one, cut in the horizontal direction.')
    parser.add_argument('--divide_vertical', dest='divide_vertical', action = 'store_true',
                        help='Create two fragments from one, cut in the vertical direction.')
    parser.add_argument('--show_image', dest = 'show_image', action = 'store_true',
                    help = 'Show intermediate images.')

    args = parser.parse_args()

    main(input_path=args.input_path,
         output_path=args.output_path,
         output_name=args.output_name,
         tiff_image=args.tiff_image,
         level=args.level,
         divide_horizontal=args.divide_horizontal,
         divide_vertical=args.divide_vertical,
         show_image=args.show_image)