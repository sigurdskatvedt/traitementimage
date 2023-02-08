# -*- coding: utf-8 -*-

import skimage
import skimage.io
import numpy as np
import scipy.ndimage
import skimage.util

# CalculerMesureHarris - Calcul de la mesure de Harris
# Usage:  R=CalculerMesureHarris(I,sigma,k)
# I = image a traiter
# sigma, k = parametres de l'algorithme de Harris
# 
# R = carte de la mesure de Harris pour l'image I
def CalculerMesureHarris(I,sigma,k):
	#Calcul des gradients
	Ix=scipy.ndimage.filters.convolve(I,np.array([[-1,0,1]]))
	skimage.io.imshow(Ix)
	Iy=scipy.ndimage.filters.convolve(I,np.array([[-1],[0],[1]]))
	skimage.io.imshow(Iy)
	
	IxIx=Ix*Ix
	IyIy=Iy*Iy
	IxIy=Ix*Iy

	A=scipy.ndimage.filters.gaussian_filter(IxIx,sigma)
	B=scipy.ndimage.filters.gaussian_filter(IyIy,sigma)
	C=scipy.ndimage.filters.gaussian_filter(IxIy,sigma)

	#tailleI=np.shape(I)
	#R=np.array((tailleI[0],tailleI[1]))
	#for i in np.range(0,tailleI[0]):
	#	for j in np.range(0,tailleI[1]):
	#		R[i,j]=A[i,j]*B[i,j]-C[i,j]*C[i,j]-k*(A[i,j]+B[i,j])**2
         
	R=(A*B-C*C)-k*((A+B)*(A+B))

	return R


# DetecterMaxLocaux - Detection des maxima locaux d'une image superieurs a un certain seuil
# Usage:  [pts,impts]=DetecterMaxLocauxSeuil(R,seuil)
# R = image dont on cherche les maxima locaux
# seuil = seuil en deca duquel on rejette les points
# 
# pts = coordonnees des maxima locaux de R superieurs au seuil
# impts = idem sous forme d'image "binaire"
def DetecterMaxLocauxSeuil(R,seuil):
	#Recherche des maxima locaux
	pts=[]
	taille=np.shape(R)
	impts=np.zeros((taille[0],taille[1]))
	impts=skimage.util.img_as_ubyte(impts)
	#On se donne une marge de maniere a etre bien sur d'eviter les effets de bords lies aux filtres (gradient, gaussienne) appliques precedemment
	margebord=3
	for i0 in range(margebord,taille[0]-margebord):
		for j0 in range(margebord,taille[1]-margebord):
			valtmp=R[i0,j0]
			if(valtmp<seuil):
				continue
			ismax=1
			for i in range(i0-1,i0+2):
				for j in range(j0-1,j0+2):
					if(i==i0 and j==j0):
						continue
					if(R[i,j]>valtmp):
						ismax=0
						break

				if(ismax==0):
					break
            
			if(ismax==0):
				continue
			impts[i0,j0]=255
			pts.append([i0,j0])

	return [pts,impts]

	
# Harris - Detection des points de Harris d'une image
# Usage:  [R,pts,impts]=Harris(I,sigma,k,seuil)
# I = image a traiter
# sigma, k = parametres de l'algorithme de Harris
# seuil = seuil sur la mesure de Harris
# 
# R = carte de la mesure de Harris pour l'image I
# pts = coordonnees des points d'interet detectes
# impts = idem sous forme d'image "binaire"
def Harris(I,sigma,k,seuil):
	R=CalculerMesureHarris(I,sigma,k)
	[pts,impts]=DetecterMaxLocauxSeuil(R,seuil)
	return [R,pts,impts]


# ExportPts - Ecriture dans un fichier texte des coordonnees des points d'interet
# Usage:  ExportPts(pts,adresse_fichier_export)
# pts = coordonnees des points d'interet detectes
# adresse_fichier_export = adresse du fichier texte dans lequel on ecrit les coordonnees des points de pts
def ExportPts(pts,adresse_fichier_export):
	nbpts=np.shape(pts)[0]
	fichier=open(adresse_fichier_export,"w")
	for n in range(0,nbpts):
		fichier.write(str(pts[n][1])+" "+str(pts[n][0])+"\n")
	fichier.close()

# SauverImageFlottant - Ecriture d'une image en flottant
# Usage: SauverImageFlottant(image,adresse_export_image)
# image  = image a enregistrer
# adresse_image = adresse d'enregistrement de l'image
def SauverImageFlottant(image,adresse_export_image):
	skimage.io.use_plugin('freeimage')
	image=np.array(image,dtype='float32')
	skimage.io.imsave(adresse_export_image,image)





#######################################
#Script principal

# Parametres de l'algorithme de Harris
sigma=0.5
k=0.05
seuil=100000

adresse_image="image027.ssech4.tif"
adresse_enregistrement_points="image027.ssech4.txt"
adresse_enregistrement_mesure_harris="R.tif"
adresse_enregistrement_image_points="impts.tif"


I=skimage.io.imread(adresse_image)
I=np.array(I,float)
[R,pts,impts]=Harris(I,sigma,k,seuil)
print str(np.shape(pts)[0])+" points"
#On exporte les points detectes dans un fichier texte
ExportPts(pts,adresse_enregistrement_points)
#On exporte les points detectes sous forme d'une image
skimage.io.imsave(adresse_enregistrement_image_points,impts)
#On exporte l'indice R
SauverImageFlottant(R,adresse_enregistrement_mesure_harris)
#skimage.io.imshow(R)









