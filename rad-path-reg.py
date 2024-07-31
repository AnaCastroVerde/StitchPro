## -----------------------------------####### REGISTRATION SCRIPT #######-------------------------------------------- ##

## ------------------------------------------------ Import packages ------------------------------------------------- ##

import imageio.v2 as imageio
import itk
import numpy as np
from skimage.filters import threshold_otsu
from skimage.feature import corner_harris, corner_peaks
from skimage import color
import scipy.ndimage as ndi
from skimage import morphology
import cv2
import SimpleITK as sitk
import math
import argparse
import matplotlib.pyplot as plt
import tifffile

## ------------------------------------------------ Parse arguments ------------------------------------------------- ##

parser = argparse.ArgumentParser(
    description='2D registration script of an MRI slice \
        and a corresponding histopathology slide.')

parser.add_argument('--input_path', dest='input_path',required=True,
                    help='Path to folder containing the images.')
parser.add_argument('--output_path', dest='output_path',required=True,
                        help='Path to store the registration results.')
parser.add_argument('--fixed', dest='fixed', required=True,
                    help='Fixed image (3D image).')
parser.add_argument('--moving', dest='moving', required=True,
                    help='Moving image (2D image).')
parser.add_argument('--moving_aux', dest='moving_aux', required=True,
                    help='Auxiliary moving image (2D image).')
parser.add_argument('--fixed_mask', dest='fixed_mask', required=True,
                    help='Binary fixed mask (Segmentation drawn on the 3D fixed image).')
parser.add_argument('--index_slice', dest='index_slice', required=True,
                    help='Index slice - number of the corresponding slice on MRI (int).')
parser.add_argument('--angle', dest='angle', required=False,
                    help='Rotation angle - angle to rotate automated points in counterclockwise orientation (int).')
parser.add_argument('--preprocessing', dest='preprocessing', action = 'store_true',
                    help='Apply preprocessing (boolean).')

args = parser.parse_args()

print("2D registration script of an MRI slice and a corresponding histopathology slide.")

## --------------------------------------------- Convert 3D to 2D mask ---------------------------------------------- ##

mask_3d = itk.imread(args.input_path+args.fixed_mask, itk.F)

Dimension = mask_3d.GetImageDimension()
indx_slice = int(args.index_slice)
extractFilter = itk.ExtractImageFilter.New(mask_3d)
extractFilter.SetDirectionCollapseToSubmatrix()

# set up the extraction region [one slice]
inputRegion = mask_3d.GetBufferedRegion()
size = inputRegion.GetSize()
size[2] = 1  # we extract along z direction
start = inputRegion.GetIndex()
sliceNumber = indx_slice
start[2] = sliceNumber

RegionType = itk.ImageRegion[Dimension]
desiredRegion = RegionType()
desiredRegion.SetIndex(start)
desiredRegion.SetSize(size)

extractFilter.SetExtractionRegion(desiredRegion)
extractFilter.Update()
mask_2d = extractFilter.GetOutput()

itk.imwrite(mask_2d, args.output_path + 'mask.nii.gz')
mask_2d = sitk.ReadImage(args.output_path + 'mask.nii.gz', sitk.sitkFloat32)[:, :, 0]
sitk.WriteImage(mask_2d, args.output_path + 'mask_2d.nii.gz')

## ---------------------------------------------- Flip the 2D mask ----------------------------------------------- ##

PixelType = itk.UC
Dimension = 2
ImageType = itk.Image[PixelType, Dimension]

reader = itk.ImageFileReader[ImageType].New()
reader.SetFileName(args.output_path + "mask_2d.nii.gz")
reader.Update()
original_image = reader.GetOutput()
original_spacing = original_image.GetSpacing()
original_origin = original_image.GetOrigin()
original_direction = original_image.GetDirection()
direction_matrix = original_direction.GetVnlMatrix().as_matrix()

flipFilter = itk.FlipImageFilter[ImageType].New()
flipFilter.SetInput(original_image)
if direction_matrix(1,1)<0:
    flipAxes = (True, True) # If last element of direction matrix is negative, performs an additional vertical flip
else:
    flipAxes = (True, False)
flipFilter.SetFlipAxes(flipAxes)
flipFilter.Update()

flipped_mask = flipFilter.GetOutput()
flipped_mask.SetSpacing(original_spacing)
flipped_mask.SetOrigin([0,0])

writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName(args.output_path + "mask_2d_flipped.nii.gz")
writer.SetInput(flipped_mask)
writer.Update()

fixed_mask_2d = itk.imread(args.output_path+'mask_2d_flipped.nii.gz', itk.UC)
fixed_mask_2d_array = itk.GetArrayFromImage(fixed_mask_2d)
fixed_mask_2d = itk.GetImageFromArray(fixed_mask_2d_array)
fixed_mask_2d.SetSpacing(original_spacing)
spacing_fixed_mask = fixed_mask_2d.GetSpacing()
origin_fixed_mask = fixed_mask_2d.GetOrigin()
direction_fixed_mask = fixed_mask_2d.GetDirection()
print("Fixed mask spacing: ", spacing_fixed_mask)
print("Fixed mask origin: ", origin_fixed_mask)
print("Fixed mask direction: ", direction_fixed_mask)
itk.imwrite(fixed_mask_2d, args.output_path + 'mask_2d_flipped_correct.nii.gz')

# ## -------------------------------------------- MRI data preprocessing ---------------------------------------------- ##

if args.preprocessing == True:
    print("Preprocessing in progress..")
    ## 1) Bias field correction

    image_T2_0001 = sitk.ReadImage(args.input_path + args.fixed)
    image_T2_0001_float = sitk.Cast(image_T2_0001, sitk.sitkFloat32) #Convert image type from short to float

    bias_field_filter = sitk.N4BiasFieldCorrectionImageFilter()
    bias_field_filter.SetNumberOfControlPoints([4,4,4])
    image_T2_0001_bias_corrected = bias_field_filter.Execute(image_T2_0001_float)

    #sitk.Show(image_T2_0001_float, title ='Original MRI data', debugOn=True)
    #sitk.Show(image_T2_0001_bias_corrected, title ='Bias field corrected MRI data', debugOn=True)

    sitk.WriteImage(image_T2_0001_float, args.input_path + 'image_T2_original.nii.gz')
    sitk.WriteImage(image_T2_0001_bias_corrected, args.input_path +  'image_T2_bias.nii.gz')

    ## 2) Intensity normalization, using the Z-score method

    Zscore_filter = sitk.NormalizeImageFilter()
    image_T2_0001_norm = Zscore_filter.Execute(image_T2_0001_bias_corrected)
    
    #sitk.Show(image_T2_0001_norm, title ='Normalized MRI data', debugOn=True)
    #plt.figure()
    #plt.hist(sitk.GetArrayViewFromImage(image_T2_0001_float).flatten(), bins=100)
    #plt.show()
    #plt.figure()
    #plt.hist(sitk.GetArrayViewFromImage(image_T2_0001_norm).flatten(), bins=100)
    #plt.show()

    sitk.WriteImage(image_T2_0001_norm, args.input_path + 'image_T2_norm_dir.nii.gz')

    image_T2_0001_norm = itk.imread(args.input_path + 'image_T2_norm_dir.nii.gz', itk.F)

else:
    image_T2_0001_norm = itk.imread(args.input_path + args.fixed, itk.F)

## ------------------------------------------ Extract 2D slice from volume ------------------------------------------ ##

Dimension = image_T2_0001_norm.GetImageDimension()
indx_slice = int(args.index_slice)
extractFilter = itk.ExtractImageFilter.New(image_T2_0001_norm)
extractFilter.SetDirectionCollapseToSubmatrix()

# set up the extraction region [one slice]
inputRegion = image_T2_0001_norm.GetBufferedRegion()
size = inputRegion.GetSize()
size[2] = 1  # we extract along z direction
start = inputRegion.GetIndex()
sliceNumber = indx_slice
start[2] = sliceNumber

RegionType = itk.ImageRegion[Dimension]
desiredRegion = RegionType()
desiredRegion.SetIndex(start)
desiredRegion.SetSize(size)

extractFilter.SetExtractionRegion(desiredRegion)
extractFilter.Update()
fixed_slice = extractFilter.GetOutput()

itk.imwrite(fixed_slice, args.output_path + 'fixed_slice.nii.gz')
fixed_image = sitk.ReadImage(args.output_path + 'fixed_slice.nii.gz', sitk.sitkFloat32)[:, :, 0]
sitk.WriteImage(fixed_image, args.output_path + 'fixed_2d.nii.gz')

## ----------------------------------------------- Flip the 2D slice ----------------------------------------------- ##
PixelType = itk.F
Dimension = 2
ImageType = itk.Image[PixelType, Dimension]

reader = itk.ImageFileReader[ImageType].New()
reader.SetFileName(args.output_path + "fixed_2d.nii.gz")
reader.Update()
original_image = reader.GetOutput()
original_spacing = original_image.GetSpacing()
original_origin = original_image.GetOrigin()
original_direction = original_image.GetDirection()
direction_matrix = original_direction.GetVnlMatrix().as_matrix()

flipFilter = itk.FlipImageFilter[ImageType].New()
flipFilter.SetInput(original_image)
if direction_matrix(1,1)<0:
    flipAxes = (True, True)
else:
    flipAxes = (True, False)
flipFilter.SetFlipAxes(flipAxes)
flipFilter.Update()

flipped_image = flipFilter.GetOutput()
flipped_image.SetSpacing(original_spacing)
flipped_image.SetOrigin([0,0])

writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName(args.output_path + "flipped.nii.gz")
writer.SetInput(flipped_image)
writer.Update()

## -------------------------------------- Extract automatic control points --------------------------------------- ##

## Get image characteristics

fixed = itk.imread(args.output_path + 'flipped.nii.gz', itk.F)
spacing_f = fixed.GetSpacing()
fixed_array = itk.GetArrayFromImage(fixed)
fixed = itk.GetImageFromArray(fixed_array)
fixed.SetSpacing(spacing_f)
spacing_f = fixed.GetSpacing()
origin_f = fixed.GetOrigin()
direction_f = fixed.GetDirection()
itk.imwrite(fixed, args.output_path + 'flipped_correct.nii.gz')

moving = itk.imread(args.input_path + args.moving, itk.F)
original_spacing = (0.25, 0.25)
new_spacing = (2**6)*original_spacing[0]*(10**(-3))
moving = itk.image_from_array(moving, is_vector=False)
moving.SetSpacing(new_spacing)
moving.SetOrigin([0,0])
region_m = moving.GetLargestPossibleRegion()
size_m = region_m.GetSize()
center_m = region_m.GetIndex()
origin_m = moving.GetOrigin()
spacing_m = moving.GetSpacing()
direction_m = moving.GetDirection()
#print("Fixed size", size_f)
print("Fixed spacing", spacing_f)
print("Fixed origin", origin_f)
print("Fixed direction", direction_f)
print("Moving size", size_m)
print("Moving spacing", spacing_m)
print("Moving origin", origin_m)
print("Moving direction", direction_m)

## Read moving image

im_moving = imageio.imread(args.input_path + args.moving, format='tiff')
im_moving = color.rgb2gray(im_moving)
plt.imshow(im_moving)

## Segment moving image based on threshold

thresh_moving = threshold_otsu(im_moving)
im_thresh_moving = im_moving > thresh_moving
im_thresh_filtered_moving = ndi.median_filter(im_thresh_moving, size = 10)

#moving_mask = morphology.binary_closing(im_thresh_filtered_moving, footprint=morphology.square(20)) # 80
moving_mask = morphology.binary_opening(im_thresh_filtered_moving, footprint=morphology.disk(20))
moving_mask = morphology.binary_closing(moving_mask, footprint=morphology.disk(20))
fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].imshow(im_thresh_filtered_moving, cmap="gray")
axs[1].imshow(moving_mask, cmap="gray")
moving_mask = sitk.GetImageFromArray(moving_mask.astype(int))
sitk.WriteImage(moving_mask, args.output_path + 'moving_mask.nii.gz')

## Read fixed image

im_fixed = imageio.imread(args.output_path+'mask_2d_flipped_correct.nii.gz', format='nii')
#im_fixed_flip = cv2.flip(im_fixed, 0)

## Segment fixed image based on threshold

thresh_fixed = threshold_otsu(im_fixed)
im_thresh_fixed = im_fixed > thresh_fixed
#thresh_fixed_flip = threshold_otsu(im_fixed_flip)
#im_thresh_fixed_flip = im_fixed_flip > thresh_fixed_flip

## Corner peaks detection

coords_fixed = corner_peaks(corner_harris(im_thresh_fixed), min_distance = 5, num_peaks = 10, threshold_rel = 0.01)
#coords_fixed_flip = corner_peaks(corner_harris(im_thresh_fixed_flip), min_distance = 5, num_peaks = 10, threshold_rel = 0.01) #min_distance = 19

#fig, axs = plt.subplots(nrows=1, ncols=2)
#axs[0].imshow(im_thresh_fixed, cmap=plt.cm.gray)
#axs[0].contour(canny_edges_fixed, colors='r')
#axs[0].plot(coords_fixed[:, 1], coords_fixed[:, 0], color='cyan', marker='o',
        #linestyle='None', markersize=6)


## Obtain automated scaling factor for points

moving_mask_pts = sitk.GetImageFromArray(im_thresh_filtered_moving.astype(int))
fixed_mask_pts = sitk.GetImageFromArray(im_thresh_fixed.astype(int)) ###m_thresh_fixed_flip

# Generate label and compute the Feret diameter - longest diameter of the mask
filter_label = sitk.LabelShapeStatisticsImageFilter()
filter_label.SetComputeFeretDiameter(True)
filter_label.Execute(moving_mask_pts)
feret_moving = filter_label.GetFeretDiameter(1)
com_y, com_x = filter_label.GetCentroid(1)
filter_label.Execute(fixed_mask_pts)
feret_fixed = filter_label.GetFeretDiameter(1)
com_y_f, com_x_f = filter_label.GetCentroid(1)

scaling_factor = feret_moving/feret_fixed
#scaling_factor = scaling_factor - 0.1*scaling_factor # Remove 10% of scaling factor to adjust the moving image

# Point rotation function

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

## Map fixed peaks to moving image

moving_center = np.array([com_x, com_y], dtype=float)
fixed_center = np.array([com_x_f, com_y_f], dtype=float)
fixed_origin = abs(np.array([origin_f[0],origin_f[1]], dtype=float))
moving_origin = abs(np.array([origin_m[0],origin_m[1]], dtype=float))

p = coords_fixed
q1 = p + (moving_center-fixed_center)
q = moving_center * (1 - scaling_factor) + q1 * scaling_factor

## Rotate points counterclockwise by given angle, for better matching

rotated_points = []
angle = int(args.angle)
for point in q:
    rotated_point = rotate(moving_center, point, math.radians(angle))
    rotated_points.append(rotated_point)
rotated_points = np.array(rotated_points)

fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].imshow(im_fixed, cmap="gray")
axs[1].imshow(im_moving, cmap="gray")
axs[0].plot(coords_fixed[:, 1], coords_fixed[:, 0], color='cyan', marker='o',
        linestyle='None', markersize=6)
axs[0].plot(fixed_center[1], fixed_center[0], color='red', marker='o',
        linestyle='None', markersize=6)
axs[1].plot(moving_center[1], moving_center[0], color='red', marker='o',
        linestyle='None', markersize=6)
axs[1].plot(rotated_points[:, 1], rotated_points[:, 0], color='cyan', marker='o',
        linestyle='None', markersize=6)

#plt.show()

## Flip x with y for index coordinates
coords_fixed_flipped = np.flip(coords_fixed, axis=1)
q_flipped = np.flip(rotated_points, axis=1)
# To get physical coordinates
coords_fixed_flipped = (coords_fixed_flipped-1)*spacing_f[0]
q_flipped = (q_flipped-1)*spacing_m[0]

## Save control points coordinates

np.savetxt(args.output_path + 'auto_coords_fixed.txt', coords_fixed_flipped)
np.savetxt(args.output_path + 'auto_coords_moving.txt', q_flipped)

# Add first two lines of the text files
mov_txt = open(args.output_path + 'auto_coords_moving.txt', "r")
fline = "point\n"
sline = str(len(q_flipped))+"\n"
oline = mov_txt.readlines()
oline.insert(0, fline)
oline.insert(1, sline)
mov_txt.close()
mov_txt = open(args.output_path+'auto_coords_moving.txt', "w")
mov_txt.writelines(oline)
mov_txt.close()

fix_txt = open(args.output_path+'auto_coords_fixed.txt', "r")
fline = "point\n"
sline = str(len(coords_fixed_flipped))+"\n"
oline = fix_txt.readlines()
oline.insert(0, fline)
oline.insert(1, sline)
fix_txt.close()
fix_txt = open(args.output_path+'auto_coords_fixed.txt', "w")
fix_txt.writelines(oline)
fix_txt.close()

# ## ------------------------------------------------ Registration ------------------------------------------------- ##

## Dilate original mask to get mask with surrounding areas - improves the registration results!

PixelType = itk.UC
Dimension = 2
radius = 10

ImageType = itk.Image[PixelType, Dimension]

reader = itk.ImageFileReader[ImageType].New()
reader.SetFileName(args.output_path+'mask_2d_flipped_correct.nii.gz')

StructuringElementType = itk.FlatStructuringElement[Dimension]
structuringElement = StructuringElementType.Ball(radius)

grayscaleFilter = itk.GrayscaleDilateImageFilter[
    ImageType, ImageType, StructuringElementType
].New()
grayscaleFilter.SetInput(reader.GetOutput())
grayscaleFilter.SetKernel(structuringElement)

writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName(args.output_path+'dilated_mask_2d.nii.gz')
writer.SetInput(grayscaleFilter.GetOutput())

writer.Update()

fixed_mask_dilated = itk.imread(args.output_path+'dilated_mask_2d', itk.UC)

moving_mask = itk.imread(args.output_path+'moving_mask', itk.UC)
moving_mask = itk.image_from_array(moving_mask, is_vector=False)
moving_mask.SetSpacing(new_spacing)
spacing_mask = moving_mask.GetSpacing()
origin_mask = moving_mask.GetOrigin()
direction_mask = moving_mask.GetDirection()
print("Moving Mask spacing: ", spacing_mask)
print("Moving Mask origin: ", origin_mask)
print("Moving Mask direction: ", direction_mask)

## Define parameter map to perform registration

parameter_object = itk.ParameterObject.New()
parameter_map_affine = parameter_object.GetDefaultParameterMap('affine')

# INITIALIZATION PARAMETERS
parameter_map_affine['AutomaticTransformInitialization'] = ['true'] # Initial translation to align image centers
parameter_map_affine['AutomaticTransformInitializationMethod'] = ['GeometricalCenter'] 
# Options={GeometricalCenter, CenterOfGravity, Origins, GeometryTop} 
#Alignment based of the Geometrical Center of the images

parameter_map_affine['Registration'] = [
    'MultiMetricMultiResolutionRegistration']
original_metric = parameter_map_affine['Metric']
parameter_map_affine['Metric'] = [original_metric[0],
                                 'CorrespondingPointsEuclideanDistanceMetric']

parameter_map_affine['AutomaticScalesEstimation'] = ['true']
#parameter_map_affine['Scales'] = ['200000.0']

# IMAGE TYPES
parameter_map_affine['FixedInternalImagePixelType'] = ['float'] # Pixels are converted to float
parameter_map_affine['FixedImageDimension'] = ['2'] #['3']
parameter_map_affine['MovingInternalImagePixelType'] = ['float']
parameter_map_affine['MovingImageDimension'] = ['2'] #['3']

# SIMILARITY MEASURE
#parameter_map_affine['Metric'] = ['AdvancedMattesMutualInformation'] 
# #{MeanSquares, Demons, Correlation, ANTSNeighborhoodCorrelation, JointHistogramMutualInformation, MattesMutualInformation}
# Measures the alignment between the two images; 'AdvancedMattesMutualInformation' is a very robust similarity measure. 
# #AdvancedNormalizedCorrelation' results well with the Optimizer Gradient Descent to solve problems in mono or multi-modal studies

# OPTIMIZER
parameter_map_affine['Optimizer'] = ['AdaptiveStochasticGradientDescent'] #'StandardGradientDescent
parameter_map_affine['ASGDParameterEstimationMethod'] = ['DisplacementDistribution']
#parameter_map_affine['MaximumNumberOfIterations'] = ['500']
#parameter_map_affine['SetMaximumStepLength'] = ['1.0']

# INTERPOLATOR
parameter_map_affine['Interpolator'] = ['LinearInterpolator'] #To map the new pixel value for the transformed image, in an off-grid position
#"Options: {NearestNeighborInterpolator, LinearInterpolator or BSplineInterpolator}

# GEOMETRIC TRANSFORMATION
parameter_map_affine['Transform'] = ['SimilarityTransform'] # SimilarityTransform is an AffineTransform without shear and with uniform scaling

# IMAGE PYRAMIDS
#parameter_map_affine['Registration'] = ['MultiResolutionRegistration']
parameter_map_affine['NumberOfResolutions'] = ['4']
parameter_map_affine['FixedImagePyramid'] = ['FixedShrinkingImagePyramid'] #{'FixedSmoothingImagePyramid', 'FixedRecursiveImagePyramid', 'FixedShrinkingImagePyramid'}
parameter_map_affine['MovingImagePyramid'] = ['MovingShrinkingImagePyramid']
parameter_map_affine['FixedImagePyramidSchedule'] = ['8', '8', '4', '4', '2', '2', '1', '1']
parameter_map_affine['MovingImagePyramidSchedule'] = ['8', '8', '4', '4', '2', '2', '1', '1']

# MASKS
parameter_map_affine['ErodeMask'] = ['false'] #When set to false considers structures surrounding the edge of the mask to be meaningful

# SAMPLER
parameter_map_affine['ImageSampler'] = ['RandomCoordinate']
parameter_map_affine['NumberOfSpatialSamples'] = ['2000'] #At least 2000 (2000-5000)
parameter_map_affine['NewSamplesEveryIteration'] = ['true']
parameter_map_affine['UseRandomSampleRegion'] = ['false']
parameter_map_affine['MaximumNumberOfSamplingAttempts'] = ['5']

parameter_object.AddParameterMap(parameter_map_affine)

parameter_map_bspline = parameter_object.GetDefaultParameterMap('bspline')
parameter_map_bspline['Transform'] = ['BSplineTransform']
parameter_map_bspline['Metric'] = ['AdvancedMattesMutualInformation',
                                 'TransformBendingEnergyPenalty'] # This metric introduces a penalty for larger deformations

parameter_map_bspline['NumberOfResolutions'] = ['4']
parameter_map_bspline['FinalGridSpacingInPhysicalUnits'] = ['10.0', '10.0']#A lower spacing between grid points allows to better match smaller structures, allowing more freedom to the transformation
#parameter_map_bspline['FinalGridSpacingInVoxels'] = ['8.0', '8.0']
parameter_map_bspline['GridSpacingSchedule'] = ['6.0', '6.0', '4.0', '4.0', '2.5', '2.5', '1.0', '1.0'] # Specifies a multi-grid schedule, by defining diferent multiplication factors for all resolution levels (going from larger to smaller voxels)

parameter_object.AddParameterMap(parameter_map_bspline)

## Call registration function

result_image, result_transform_parameters = itk.elastix_registration_method(
    fixed, moving,
    fixed_point_set_file_name = args.output_path+'auto_coords_fixed.txt',
    moving_point_set_file_name = args.output_path+'auto_coords_moving.txt',
    fixed_mask = fixed_mask_2d,
    moving_mask = moving_mask,
    parameter_object=parameter_object,
    log_to_console=True)

itk.imwrite(result_image, args.output_path + 'result_image.nii.gz')

# Convert the output image to a NumPy array
result_array = itk.GetArrayFromImage(result_image)
result_array_transpose = np.flip(result_array, axis = 0)

rgb_array = np.repeat(result_array_transpose[..., np.newaxis], 3, axis=-1)

# Save the image in TIFF format
tifffile.imwrite(args.output_path + 'result_image.tif', rgb_array, photometric='rgb', 
                 resolution = (1/new_spacing,1/new_spacing), metadata={'spacing': new_spacing, 'unit': 'um'})

print("Registration finished!")

## ------------------------------------- Compute registration accuracy (Dice) -------------------------------------- ##

# Load the Images
fixed_mask = itk.imread(args.output_path + 'mask_2d_flipped_correct.nii.gz', itk.F)
result_image = itk.imread(args.output_path + 'result_image.nii.gz', itk.F)

# Binarize the result image, according to Ostu threshold
out = itk.OtsuMultipleThresholdsImageFilter(result_image)

# Smooth the image, to remove annotation points
smooth = itk.MedianImageFilter(out)

# Compute the Dice score between the two masks (fixed T2 and histopathology registered)
intersection = np.logical_and(fixed_mask, smooth)
dice = np.sum(intersection) * 2.0 / (np.sum(fixed_mask) + np.sum(smooth))
print("Dice similarity coefficient (DSC):", round(dice, 2))

# Open a file and append Dice score
file_object = open("/home/jose_almeida/prospective_uc4/DSC.txt", "a")
file_object.write(str(args.fixed)[9:12]+";")
file_object.write(str(args.moving)[0:7]+";")
file_object.write(str(args.index_slice)+";")
file_object.write(str(round(dice,2))+"\n")
file_object.close()

## ------------------------------------ Transformix for registering predictions ------------------------------------- ##

moving_aux = itk.imread(args.input_path + args.moving_aux, itk.F)
moving_aux.SetSpacing(new_spacing)

result_image_transformix = itk.transformix_filter(
    moving_aux,
    result_transform_parameters,
)

itk.imwrite(result_image_transformix, args.output_path + 'result_image_aux.nii.gz')

# Exclude predictions with low intensity values for visualization
moving_aux_np = itk.array_view_from_image(result_image_transformix)
moving_aux_np[moving_aux_np < 50] = 0
moving_aux_modified = itk.image_from_array(moving_aux_np)
moving_aux_modified.CopyInformation(result_image_transformix)

itk.imwrite(moving_aux_modified, args.output_path + 'result_image_aux_visual.nii.gz')