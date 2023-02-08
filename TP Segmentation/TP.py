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

im_class_name = "./TP Segmentation/Donnees_OBIA/Exercice_2_1/IRC_Classif.tif"
im_seg_name = "./TP Segmentation/Donnees_OBIA/Exercice_2_1/IRC_Segmentation.tif"
im_irc_name = "./TP Segmentation/Donnees_OBIA/Exercice_2_1/ImageIRC.tif"

def exercise22(im_irc_name, im_seg_name):
    im_irc =tifffile.imread(im_irc_name)
    im_seg =tifffile.imread(im_seg_name)
    im_seg_unique = np.unique(im_seg)
    im_irc_average = np.zeros((im_irc.shape[0],im_irc.shape[1]))
    """ for x in range(im_irc.shape[0]):
        for y in range(im_irc.shape[1]):
            im_irc_average[x][y] = round(np.average(im_irc[x][y])) """



exercise21(im_irc_name, im_seg_name)