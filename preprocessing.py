## -- This script does preprocessing on histopathology images, before undergoing image sticthing and registration -- ##
## -- Steps 1) Extracts downsampled image at a lower downsampling level -- ##
## -------- 2) Crops image to remove background outside slide ------------ ##

## Import packages
#from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.wsicore import WSIReader
from tiatoolbox.utils.image import make_bounds_size_positive
from tiatoolbox.tools.tissuemask import MorphologicalMasker
from tiatoolbox.utils.misc import get_bounding_box
from pprint import pprint
import tifffile
import numpy as np
from matplotlib import pyplot as plt
import imageio.v2 as imageio
from scipy.ndimage import rotate
import argparse
import cv2 as cv

def main(input_path: str,
         input_path_aux: str,
         output_path: str,
         output_name: str,
         output_name_aux: str,
         tiff_image: str,
         tiff_image_aux: str,
         #left: int,
         #right: int,
         #top: int,
         #bottom: int,
         level: int,
         #angle: int,
         divide_horizontal: bool,
         divide_vertical: bool,
         show_image: bool):
    ## Read the original TIFF image
    reader = WSIReader.open(input_path + tiff_image)
    print(reader)

    reader_aux = WSIReader.open(input_path_aux + tiff_image_aux)

    ## Obtain TIFF metadata
    info_dict = reader.info.as_dict()
    pprint(info_dict)
    original_size = info_dict['level_dimensions'][0]
    original_spacing = info_dict['mpp']

    ## Extract a downsampled region within bounds (to remove whitespace on slide)
    wsi_thumb = reader.slide_thumbnail(
    resolution=level,
    units="level")

    masker = MorphologicalMasker(power=1.25, min_region_size=80000)
    masks = masker.fit_transform([wsi_thumb])[0]

    # def show_side_by_side(image_1: np.ndarray, image_2: np.ndarray) -> None:
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(image_1)
    #     plt.axis("off")
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(image_2)
    #     plt.axis("off")
    #     plt.show()

    # show_side_by_side(wsi_thumb, masks)

    start_x, start_y, end_x, end_y = get_bounding_box(masks)

    #bounds = [left, top, info_dict['level_dimensions'][level][0]-right, info_dict['level_dimensions'][level][1]-bottom]
    bounds = [start_x, start_y, end_x, end_y]
    region = reader.read_bounds(bounds, resolution=level, units="level", coord_space = "resolution")

    region_aux = reader_aux.read_bounds(bounds, resolution=level, units="level", coord_space = "resolution")
    
    ## Extract angle to orient fragment to horizontal
    binary_mask = masks.astype(np.uint8)*255
    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)
    #rect_corners = np.array([[end_x,start_y],[start_x,start_y],[start_x,end_y],[end_x,end_y]], dtype =np.float32)
    #print("RECT corners", rect_corners)
    rect = cv.minAreaRect(largest_contour)
    angle = round(rect[-1], 0)

    ## Small rotation angle to straighten up the fragment
    region_rotated = rotate(region, angle = angle, axes=(1, 0), reshape=True, mode = 'constant', cval=230.0)

    region_rotated_aux = rotate(region_aux, angle = angle, axes=(1, 0), reshape=True, mode = 'constant', cval=230.0)

    if show_image == True:
        fig, axs = plt.subplots(nrows=2, ncols=2)
        axs[0,0].imshow(region)
        axs[0,1].imshow(region_rotated)
        axs[1,0].imshow(region_aux)
        axs[1,1].imshow(region_rotated_aux)
        plt.show()

    print(region.shape, region_aux.shape)
    print(region_rotated.shape, region_rotated_aux.shape)
    new_spacing = (2**level)*original_spacing[0]

    M = region_rotated.shape[0]//2
    N = region_rotated.shape[1]//2

    if (divide_horizontal == True) & (divide_vertical == False):
        tiles = [region_rotated[:, y:y + N] for y in range(0, region_rotated.shape[1], N)]
        fragment_1 = tiles[0]
        fragment_2 = tiles[1]
        if show_image == True:
            fig, axs = plt.subplots(nrows=1, ncols=2)
            axs[0].imshow(fragment_1, cmap="gray")
            axs[1].imshow(fragment_2, cmap="gray")
            plt.show()
        ## Write the downsampled TIFF file
        tifffile.imwrite(
            output_path + str(output_name) + "_fragment1.tiff",
            np.array(fragment_1), photometric='rgb', imagej=True, resolution=(1 / new_spacing, 1 / new_spacing),
            metadata={'spacing': new_spacing, 'unit': 'um'})
        tifffile.imwrite(
            output_path + str(output_name) + "_fragment2.tiff",
            np.array(fragment_2), photometric='rgb', imagej=True, resolution=(1 / new_spacing, 1 / new_spacing),
            metadata={'spacing': new_spacing, 'unit': 'um'})
    if (divide_vertical == True) & (divide_horizontal == False):
        tiles = [region_rotated[x:x + M, :] for x in range(0, region_rotated.shape[0], M)]
        fragment_1 = tiles[0]
        fragment_2 = tiles[1]
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
    if (divide_horizontal == False) & (divide_vertical == False):
        ## Write the downsampled TIFF file
        tifffile.imwrite(
            output_path + str(output_name) + ".tiff",
            np.array(region_rotated), photometric='rgb', imagej=True, resolution=(1 / new_spacing, 1 / new_spacing),
            metadata={'spacing': new_spacing, 'unit': 'um'})  # metadata={'spacing': 32, 'unit': 'um'}
        tifffile.imwrite(
            output_path + str(output_name_aux) + ".tiff",
            np.array(region_rotated_aux), photometric='rgb', imagej=True, resolution=(1 / new_spacing, 1 / new_spacing),
            metadata={'spacing': new_spacing, 'unit': 'um'})  # metadata={'spacing': 32, 'unit': 'um'}
        
## Parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocessing histopathology images')

    parser.add_argument('--input_path', dest='input_path',required=True,
                        help='Path to folder containing the image.')
    parser.add_argument('--input_path_aux', dest='input_path_aux',required=True,
                        help='Path to folder containing the image.')
    parser.add_argument('--output_path', dest='output_path',required=True,
                            help='Path to store the preprocessing results.')
    parser.add_argument('--output_name', dest='output_name',required=True,
                            help='Name of the output preprocessed fragment.')
    parser.add_argument('--output_name_aux', dest='output_name_aux',required=True,
                            help='Name of the auxiliary output preprocessed fragment.')
    parser.add_argument('--tiff_image', dest='tiff_image', required=True,
                        help='Histopathology image to preprocess.')
    parser.add_argument('--tiff_image_aux', dest='tiff_image_aux', required=True,
                        help='Auxiliary histopathology image to preprocess.')
    parser.add_argument('--level', dest='level', required=True, type=int,
                        help='Resolution level to downsample.')
    #parser.add_argument('--top', dest='top', required=True, type=int,
                        #help='Bounds distance to subtract on the top.')
    #parser.add_argument('--left', dest='left', required=True, type=int,
                        #help='Bounds distance to subtract on the left.')
    #parser.add_argument('--right', dest='right', required=True, type=int,
                        #help='Bounds distance to subtract on the right.')
    #parser.add_argument('--bottom', dest='bottom', required=True, type=int,
                        #help='Bounds distance to subtract on the bottom.')
    #parser.add_argument('--angle', dest='angle', required=True, type=int,
                        #help='Small angle to rotate the fragment; (-) if clockwise.')
    parser.add_argument('--divide_horizontal', dest='divide_horizontal', action = 'store_true',
                        help='Create two fragments from one, cut in the horizontal direction.')
    parser.add_argument('--divide_vertical', dest='divide_vertical', action = 'store_true',
                        help='Create two fragments from one, cut in the vertical direction.')
    parser.add_argument('--show_image', dest = 'show_image', action = 'store_true',
                    help = 'Show intermediate images.')

    args = parser.parse_args()
    #print("Preprocessing histopathology image is starting...")

    main(input_path=args.input_path,
         input_path_aux=args.input_path_aux,
         output_path=args.output_path,
         output_name=args.output_name,
         output_name_aux=args.output_name_aux,
         tiff_image=args.tiff_image,
         tiff_image_aux=args.tiff_image_aux,
         #left=args.left,
         #right=args.right,
         #top=args.top,
         #bottom=args.bottom,
         level=args.level,
         #angle=args.angle,
         divide_horizontal=args.divide_horizontal,
         divide_vertical=args.divide_vertical,
         show_image=args.show_image)