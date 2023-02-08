# -*- coding: utf-8 -*-

import skimage
import skimage.io
import numpy as np
import scipy.ndimage
import skimage.util



# lecture_fichier_points - Lit un fichier de points d'interet au format texte avec coordonnees colonne ligne sur chaque ligne
# Usage:  pts=lecture_fichier_points(adresse_fichier_points)
def lecture_fichier_points(adresse_fichier_points):
	pts=np.genfromtxt(adresse_fichier_points)
	nbpts=np.shape(pts)[0]
	#Attention, on a enregistre les points en inversant ligne et colonne
	for n in range(0,nbpts):
		col=pts[n,0]
		li=pts[n,1]
		pts[n,0]=li
		pts[n,1]=col
	return pts


# filtrage_bord - Ne conserve que les points suffisamment eloignes des bords d'une image
# Usage:  ptsok=filtrage_bord(pts,rayon_bord,taille_image)
# pts = coordonnees des points detectes 
# rayon_bord   = distance au bord de l'image ou les points doivent etre elimines
# taille_image = taille de l'image
#
# ptsok = coordonnees des points conserves
def filtrage_bord(pts,rayon_bord,taille_image):
	ptsok=[]
	nbpts=np.shape(pts)[0]
	nl=taille_image[0]
	nc=taille_image[1]
	for n in range(nbpts):
		li=pts[n,0]
		col=pts[n,1]
		if(li<rayon_bord+1):
			continue
		if(col<rayon_bord+1):
  			continue
		if(li>nl-(rayon_bord+1)):
			continue
		if(col>nc-(rayon_bord+1)):
			continue
		ptsok.append([li,col])
	return ptsok


# MoyenneEcarttype - Calcule en chaque pixel de l'image la moyenne et l'ecart-type de l'image au sein d'une fenetre carree de cote 2xrayon_fenetre+1
# Usage:  [Moyenne,EcartType]=MoyenneEcarttype(Image,rayon_fenetre)
def MoyenneEcarttype(Image,rayon_fenetre):
	#Le filtre moyenne est un filtre separable
	#Creation des noyaux de convolution associes au filtre moyenne
	noyau_ligne=np.ones((1,2*rayon_fenetre+1))
	noyau_colonne=np.ones((2*rayon_fenetre+1,1))
	nbelts=(float)((2*rayon_fenetre+1)*(2*rayon_fenetre+1))
	Moyenne=scipy.ndimage.filters.convolve(scipy.ndimage.filters.convolve(Image,noyau_ligne),noyau_colonne)
	Moyenne=Moyenne/nbelts
	MoyenneCarre=scipy.ndimage.filters.convolve(scipy.ndimage.filters.convolve(Image*Image,noyau_ligne),noyau_colonne)
	MoyenneCarre=MoyenneCarre/nbelts
	EcartType=np.sqrt(MoyenneCarre-Moyenne*Moyenne)
	return [Moyenne,EcartType]


# tabulation_infopoint_vignette2vecteur - Recupere et stocke pour chaque point les informations utiles pour la correlation
#
# [ptsmoyect,ptsdesc]=tabulation_infopoint_vignette2vecteur(I,MoyenneI,EcarttypeI,ptsok,rayon_vignette_correlation)
# 
# I = image
# MoyenneI = Moyenne de I au sein d'une fenetre de la taille de la vignette de correlation
# EcarttypeI = Ecart-type de I au sein dune fenetre de la taille de la vignette de correlation
# ptsok = points sur lesquels on va centrer la vignette de correlation
# rayon_vignette_correlation  = rayon de la vignette de correlation
#
# ptsmoyect = moyenne et ecart-type de l'image au sein de la vignette de correlation pour chaque point de ptsok
# ptsdesc = descripteur contenant les valeurs de ses voisins au sein de la vignette de correlation pour chaque point de ptsok
def tabulation_infopoint_vignette2vecteur(I,ptsok,rayon_vignette_correlation):
	nbpts=np.shape(ptsok)[0]
	ptsmoyect=np.zeros((nbpts,2))
	ptsdesc=np.zeros((nbpts,(2*rayon_vignette_correlation+1)*(2*rayon_vignette_correlation+1)))
	for n in range(0,nbpts):
		li=ptsok[n][0]
		col=ptsok[n][1]
		ptsdesctmp=I[((int)(li-rayon_vignette_correlation)):((int)(li+rayon_vignette_correlation+1)),((int)(col-rayon_vignette_correlation)):((int)(col+rayon_vignette_correlation+1))]
		ptsdesctmp=ptsdesctmp.flatten()
		ptsdesc[n,:]=ptsdesctmp
		ptsmoyect[n,0]=np.mean(ptsdesctmp)
		ptsmoyect[n,1]=np.std(ptsdesctmp)
		#ptsmoyect[n,1]=sqrt(np.mean(ptsdesctmp*ptsdesctmp)-ptsmoyect[n,0]**2)

	return [ptsmoyect,ptsdesc]


# ExportAppariements - Ecriture dans un fichier texte des coordonnees de paires de points homologues 
# Usage : ExportAppariements(PtsHomologues,adresse_fichier_export)
# PtsHomologues  = coordonnees des points mis en correspondance
# adresse_fichier_export = adresse du fichier dans lequel on ecrit les coordonnees des points mis en correspondance
def ExportAppariements(PtsHomologues,adresse_fichier_export):
	nbptshomologues=np.shape(PtsHomologues)[0]
	fichier=open(adresse_fichier_export,"w")
	for n in range(0,nbptshomologues):
		fichier.write(str(PtsHomologues[n][0][1])+" "+str(PtsHomologues[n][0][0])+" "+str(PtsHomologues[n][1][1])+" "+str(PtsHomologues[n][1][0])+"\n")
	fichier.close()



# Appariement - Pour chaque point d'interet detecte dans l'image1, recherche de son homologue parmi ceux detectes dans l'image2
# Usage:  PtsHomologues=Appariement(pts1ok,pts1moyect,pts1desc,pts2ok,pts2moyect,pts2desc,seuil=-1)
# pts1ok = coordonnees des points detectes dans l'image 1
# pts1moyect = moyenne et ecart-type de l'image au sein de la vignette de correlation pour les points detectes dans l'image 1
# pts1desc = valeurs des pixels de l'image  au sein de la vignette de correlation pour les points detectes dans l'image 1
# pts2ok = coordonnees des points detectes dans l'image 2
# pts2moyect = moyenne et ecart-type de l'image au sein de la vignette de correlation pour les points detectes dans l'image 2
# pts2desc = valeurs des pixels de l'image  au sein de la vignette de correlation pour les points detectes dans l'image 2
# seuil =  seuil sur le coefficient de correlation de deca duquel l'appariement est rejete
#
# PtsHomologues = coordonnees des points mis en correspondance
def Appariement(pts1ok,pts1moyect,pts1desc,pts2ok,pts2moyect,pts2desc,seuil=-1):
	PtsHomologues=[]
	#nbeltsvignette=(2*rayon_vignette_correlation+1)*(2*rayon_vignette_correlation+1)
	nbeltsvignette=np.shape(pts1desc)[1]
	nbpts1=np.shape(pts1ok)[0]
	nbpts2=np.shape(pts2ok)[0]
	for i1 in range(0,nbpts1):
		moy1=pts1moyect[i1,0]
		ect1=pts1moyect[i1,1]
		bestscore=-100000
		ibest=0
		V1=pts1desc[i1,:].transpose()
		for i2 in range(0,nbpts2):
			moy2=pts2moyect[i2,0]
			ect2=pts2moyect[i2,1]
			coefcorreltmp=np.dot(pts2desc[i2,:],V1)/nbeltsvignette
			coefcorreltmp=(coefcorreltmp-moy1*moy2)/(ect1*ect2)
			#E((x-E(x))*(y-E(y)))=E(x.y-x.E(y)-y.E(x)+E(x).E(y))=E(x.y)-E(x).E(y)
			if(coefcorreltmp>bestscore):
				ibest=i2
				bestscore=coefcorreltmp
		if(seuil>0 and bestscore<seuil):
			continue
		PtsHomologues.append([pts1ok[i1],pts2ok[ibest]])
	return PtsHomologues





# MainAppariement - Pour chaque point d'interet detecte dans l'image1, recherche de son homologue parmi ceux detectes dans l'image2 a l'aide du coefficient de correlation normalise
# Usage:  MainAppariement(adresse_fichier_points_1, adresse_image_1, adresse_fichier_points_2, adresse_image_2, adresse_enregistrement_appariements, rayon_vignette_correlation, seuil_correl)
# adresse_fichier_points_1 = adresse du fichier contenant les coordonnees des points detectes dans l'image 1
# adresse_image_1 = adresse de l'image 1
# adresse_fichier_points_2 = adresse du fichier contenant les coordonnees des points detectes dans l'image 2
# adresse_image_2 = adresse de l'image 2
# adresse_enregistrement_appariements = adresse du fichier dans lequel on ecrit les coordonnees des points mis en correspondance
# rayon_vignette_correlation = "rayon" de la vignette de correlation (la longueur du cote de la fenetre de correlation vaut donc 2*rayon_vignette_correlation+1)
# seuil_correl =  seuil sur le coefficient de correlation de deca duquel l'appariement est rejete. Par defaut (seuil_correl=-1) tous les appariements sont conserves.
def MainAppariement(adresse_fichier_points_1, adresse_image_1, adresse_fichier_points_2, adresse_image_2, adresse_enregistrement_appariements, rayon_vignette_correlation, seuil_correl=-1):
	#Chargeons les images et passons les en flottants
	I1=skimage.io.imread(adresse_image_1)
	I1=np.array(I1,float)
	taille_I1=np.shape(I1)
	I2=skimage.io.imread(adresse_image_2)
	I2=np.array(I2,float)
	taille_I2=np.shape(I2)
	
	#Chargeons les points d'interet deja detectes en eliminant ceux trop pres des bords de l'image
	pts1=lecture_fichier_points(adresse_fichier_points_1)
	pts2=lecture_fichier_points(adresse_fichier_points_2)

	#On va exclure les points trop proches du bord.
	pts1ok=filtrage_bord(pts1,rayon_vignette_correlation,taille_I1)
	pts2ok=filtrage_bord(pts2,rayon_vignette_correlation,taille_I2)
		
	#Pour chaque point d'interet on stocke dans un vecteur les valeur des pixels compris dans sa vignette de correlation
	[pts1moyect,pts1desc]=tabulation_infopoint_vignette2vecteur(I1,pts1ok,rayon_vignette_correlation)
	[pts2moyect,pts2desc]=tabulation_infopoint_vignette2vecteur(I2,pts2ok,rayon_vignette_correlation)
	
	#Pour chaque point de l'image1, recherche de son homologue parmi ceux detectes dans l'image2
	PtsHomologues=Appariement(pts1ok,pts1moyect,pts1desc,pts2ok,pts2moyect,pts2desc,seuil_correl)

	#On peut maintenant ecrire le resultat dans un fichier texte
	ExportAppariements(PtsHomologues,adresse_enregistrement_appariements)





#######################################
#Script principal
adresse_fichier_points_1="image027.ssech4.pts"
adresse_image_1="image027.ssech4.tif"
adresse_fichier_points_2="image028.ssech4.pts"
adresse_image_2="image028.ssech4.tif"
adresse_enregistrement_appariements="image027.ssech4.-.image028.ssech4.result"

#parametre "rayon" reglant la taille de la vignette de correlation
rayon_vignette_correlation=10

#parametre "seuil" reglant le seuil sur le coefficient de correlation de deca duquel l'appariement est rejete
# si on le choisit avec une valeur negative, tous les appariements sont acceptes
seuil_correl=-1


MainAppariement(adresse_fichier_points_1, adresse_image_1, adresse_fichier_points_2, adresse_image_2, adresse_enregistrement_appariements, rayon_vignette_correlation, seuil_correl)




