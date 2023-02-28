from collections import Counter
import cv2
from skimage import io, metrics
import tifffile
import numpy as np
from matplotlib import pyplot as plt

def exercise21(im_class_name, im_seg_name):
    im_class=tifffile.imread(im_class_name)
    nbc=np.unique(im_class).size
    im_seg=tifffile.imread(im_seg_name)
    im_seg_unique = np.unique(im_seg)
    nbs=im_seg_unique.size

    """Parcourir conjointement les images de segmentation et de classification:
    Pour chaque pixel, ajouter +1 à l'élément (classe, id_segment) de M 
        classe=valeur du pixel dans l'image de classification"""
    M = np.zeros((nbc,nbs))
    for x in range(im_class.shape[0]):
        for y in range(im_class.shape[1]):
            val_cla = im_class[x,y]-1
            val_seg = im_seg[x,y]
            #id_segment is index of segmentation value in our min->max ordered array of unique segmentation values 
            id_segment = np.where(im_seg_unique == val_seg)
            M[val_cla, id_segment] += 1

    """Créer un vecteur V colonne de taille (nbs) et récupérer l'identifiant de la valeur max pour chaque segment"""
    M_transpose = M.transpose()
    V = np.amax(M_transpose, axis=1)

    """Créer une image vide de taille identique à celle de segmentation
    Pour chaque pixel, affecter la valeur de la ligne V[id_segment]."""
    return_img = np.zeros((im_seg.shape[0],im_seg.shape[1]))
    for i in range(return_img.shape[0]):
        for j in range(return_img.shape[1]):
            val_seg = im_seg[i][j]
            id_segment = np.where(im_seg_unique == val_seg)
            return_img[i][j] = V[id_segment]
    return return_img



def exercise22(im_irc_name, im_seg_name):
    im_irc =tifffile.imread(im_irc_name)
    im_seg =tifffile.imread(im_seg_name)
    # Creates two identical dictionaries the keys are the unique values in the segmentation image
    unique_values_dictinary= {element: [] for element in np.unique(im_seg)}
    im_irc_seg_average = {element: [] for element in np.unique(im_seg)} 

    # Adds all I/R/C to the corresponding segment in our dictionary
    for x in range(im_irc.shape[0]):
        for y in range(im_irc.shape[1]):
            segment_value = im_seg[x][y]
            unique_values_dictinary[segment_value].append(im_irc[x,y])
    
    # Calculate average I/R/C-values for each unique segment
    for elements in unique_values_dictinary:
        i_values = []
        r_values = []
        c_values = []
        # For each set of I/R/C values for each segment, calculate the average
        for array in unique_values_dictinary[elements]:
            i_values.append(array[0])
            r_values.append(array[1])
            c_values.append(array[2])
        average_values = [int(average(i_values)),int(average(r_values)), int(average(c_values))]
        im_irc_seg_average[elements] = average_values
    return_image = np.zeros((im_seg.shape[0],im_seg.shape[1], 3))
    # Inserts all I/R/C values in the correct location
    for x in range(return_image.shape[0]):
        for y in range(return_image.shape[1]):
            segment = im_seg[x][y]
            pixel_value = im_irc_seg_average[segment]
            return_image[x][y] = pixel_value
    return return_image 

def exercise23(im_class_name, im_seg_name):
    im_class = tifffile.imread(im_class_name)
    im_seg = tifffile.imread(im_seg_name)
    output_image = im_seg
    max_seg = im_seg.max()
    min_seg = im_seg.min()
    # Dictionary where the keys are all the unique segmentation values
    unique_values_dictinary= {element: [] for element in np.unique(im_seg)}
    for x in range(im_class.shape[0]):
        for y in range(im_class.shape[1]):
            # For each unique segmentation value, add the classes of the pixels in that segment
            unique_values_dictinary[im_seg[x][y]].append(im_class[x][y])
    average_segment = unique_values_dictinary
    for segment_value in range(min_seg,max_seg+1):
        # For each unique segment, find the class that is most frequent.
        average_segment[segment_value] = most_frequent(unique_values_dictinary[segment_value])
    for x in range(output_image.shape[0]):
        for y in range(output_image.shape[1]):
            # Iterate through each pixel an replace the value with the most frequent class for each segment
            output_image[x][y] = average_segment[im_seg[x][y]]
    return output_image
            



# Method that finds the average of numbers in an array
def average(arr):
    return sum(arr) / len(arr)

# Method that finds the number that is most frequest in an array
def most_frequent(arr):
    """
    Finds the most common number in an array of numbers.

    Args:
        arr: A list of integers.

    Returns:
        The most common integer in the array.
    """
    counter = Counter(arr)
    return counter.most_common(1)[0][0]

def imageCompare_SSIM(im_name_1, im_name_2):
    # Load the two images
    image1 = io.imread(im_name_1)
    image2 = io.imread(im_name_2)
    
    image2 = image2 / 5 * 255

    # Compare the images using SSIM
    ssim_value = metrics.structural_similarity(image1,image2)

    # Print the SSIM value
    print("The SSIM value between the two images is: ", ssim_value)

def imageCompare_BinaryPixelCompare(im_name_1, im_name_2):
    # Load the two images
    img1 = io.imread(im_name_1)
    img2 = io.imread(im_name_2) 

    result = np.zeros((img1.shape[0],img1.shape[1]))

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i, j] == img2[i, j]:
                result[i][j]=1
            else:
                result[i][j]=0

    return result

def imageCompare_BinaryDifference(im_name_1, im_name_2):
    # Load the two images
    img1 = cv2.imread(im_name_1)
    img2 = cv2.imread(im_name_2)

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the two grayscale images
    diff = cv2.absdiff(gray1, gray2)

    # Threshold the difference image to create a binary image
    threshold = 30
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    return thresh

def image_normalize(image):
    return_image = image.astype(np.float64) / np.max(image)# normalize the data to 0 - 1
    return_image = 255 * return_image# Now scale by 255
    tiff_save = return_image.astype(np.uint8)
    return tiff_save


im_class_name = "./TP Segmentation/Donnees_OBIA/Exercice_2_1/IRC_Classif.tif"
im_seg_name = "./TP Segmentation/Donnees_OBIA/Exercice_2_1/IRC_Segmentation.tif"
exercise_21_save_name = "./TP Segmentation/Donnees_OBIA/Exercice_2_1/exercise_21.tif"

exercise_21_image = exercise21(im_class_name, im_seg_name)
normalize_21 = image_normalize(exercise_21_image)
tifffile.imwrite(exercise_21_save_name, normalize_21)


im_irc_name = "./TP Segmentation/Donnees_OBIA/Exercice_2_1/ImageIRC.tif"
irc_save_name = "./TP Segmentation/Donnees_OBIA/Exercice_2_2/irc_computed.tif"

exercise_22_image_1 = exercise22(im_irc_name, im_seg_name)

tiff_save = image_normalize(exercise_22_image_1) 
tifffile.imwrite(irc_save_name, tiff_save)

im_23_class = "./TP Segmentation/Donnees_OBIA/Exercice_2_3/KoLanta_classification.tif"
im_23_seg = "./TP Segmentation/Donnees_OBIA/Exercice_2_3/KoLanta_segmentation.tif"
output_name = "./TP Segmentation/Donnees_OBIA/Exercice_2_3/output.tif"

excerise_23_regularized = exercise23(im_23_class, im_23_seg)
excerise_23_regularized_normalized = image_normalize(excerise_23_regularized)

tifffile.imwrite(output_name, excerise_23_regularized_normalized)

image_1_name = "./TP Segmentation/image_compare/exercise_21.tif"
image_2_name = "./TP Segmentation/image_compare/sortie_classee_5.tif"
image_2 = image_normalize(tifffile.imread(image_2_name))
tifffile.imwrite("./TP Segmentation/image_compare/normalized.tif", image_2)

compare_array = imageCompare_BinaryDifference(image_1_name, image_2_name)
tifffile.imwrite("./TP Segmentation/Donnees_OBIA/Exercice_2_3/difference.tif")