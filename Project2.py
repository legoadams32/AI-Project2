#Project 2
#Max Adams

import cv2
import numpy as np


#Get all img paths
img_path1 = 'Images/image-1.bmp'
img_path2 = 'Images/image-2.bmp'
img_path3 = 'Images/image-3.bmp'
img_path4 = 'Images/image-4.bmp'
img_path5 = 'Images/image-5.bmp'
img_path6 = 'Images/image-6.bmp'
img_path7 = 'Images/image-7.bmp'
img_path8 = 'Images/image-8.bmp'
img_path9 = 'Images/image-9.bmp'
img_path10 = 'Images/image-10.bmp'
img_path11 = 'Images/image-11.bmp'


#Create the list of img_paths
img_paths = []

img_paths.append("img_path1")
img_paths.append("img_path2")
img_paths.append("img_path3")
img_paths.append("img_path4")
img_paths.append("img_path5")
img_paths.append("img_path6")
img_paths.append("img_path7")
img_paths.append("img_path8")
img_paths.append("img_path9")
img_paths.append("img_path10")
img_paths.append("img_path11")



def applyHisto(img):
	print("Enter Histogram")

	Img_width = img.shape[1]
	Img_height = img.shape[0]

	H = []

	for i in range(256):
		H.append(0)
	#Sets each value to 0

	for i in range(Img_height):
		for j in range(Img_width):
			z = img[i,j]
			H[z] +=1
		#Creates the histogram

	#Compute the commulative histogram
	for i in range(len(H)):
		H[i] = H[i-1] + H[i]

	#Normalize the commlative histogram

	for i in range(len(H)):
		H[i] = H[i] * 255 / (Img_width * Img_height)

	#Put the new image together
	for i in range(Img_height):
		for j in range(Img_width):
			z = img[i,j]
			img[i, j] = H[z]

	return img



def edge_dection(img, THRESHOLD):
	#Basic edge dection with no weights added to them. Only is using the x Direction dection

	print("Enter edge_dection")

	Img_width = img.shape[1]
	Img_height = img.shape[0]


	print(Img_width)
	print(Img_height)
	for i in range(Img_height):
		for j in range(Img_width):


			try:

				pos1 = int(img[i-1, j-1]) #top left
				pos2 = int(img[i-1, j]) #top middle
				pos3 = int(img[i-1, j+1]) #top right
				pos4 = int(img[i, j-1]) #center left
				pos5 = int(img[i, j]) #center
				pos6 = int(img[i, j+1]) # center right
				pos7 = int(img[i+1, j-1]) #bottom left
				pos8 = int(img[i+1, j]) #bottom middle
				pos9 = int(img[i+1, j+1]) #bottom right

				#X Direction
				xsum1 = img[i-1, j-1] + img[i-1, j] + img[i-1, j+1]
				ysum1 = img[i+1, j-1] + img[i+1, j] + img[i+1, j+1]

				value1 = xsum1-ysum1
				value1 = abs(value1)
				#print(value1)

				#Y direction
				xsum2 = img[i-1,j-1] + img[i,j-1] + img[i+1, j-1]
				ysum2 = img[i+1,j-1] + img[i+1,j] + img[i+1, j+1]

				value2 = xsum2 - ysum2
				value2 = abs(value2)

				#Diagonal 1
				xsum3 = int(img[i,j-1]) + int(img[i-1,j-1]) + int(img[i-1,j])
				ysum3 = int(img[i+1,j]) + int(img[i+1,j+1]) + int(img[i,j+1])

				#value3 = xsum3 - (ysum3)
				value3 = abs(xsum3) - abs(ysum3)

				#diagonal 2
				xsum4 = int(img[i,j-1]) + int(img[i+1,j-1]) + int(img[i+1, j])
				ysum4 = int(img[i-1,j]) + int(img[i-1, j+1]) + int(img[i, j+1])
			
				#value4 = xsum4 - (ysum4)
				value4 = abs(xsum4) - abs(ysum4)


				if value1 > THRESHOLD:
					img[i,j] = 255
				#if value2 > THRESHOLD:
					#Turn on the y-direction
					#img[i,j] = 255
				#elif value3 > THRESHOLD:
					#img[i,j] = 255
				#elif value4 > THRESHOLD:
					#img[i,j] = 255
				else:
					img[i,j] = 0
			except:
				pass


	return img

#Edge thining algo
def edge_thining(img):
	print("Enter edge_thining")

	Img_width = img.shape[1]
	Img_height = img.shape[0]

	fimg = np.empty([Img_height, Img_width], dtype=int)
	cimg = np.empty([Img_height, Img_width], dtype=int)

	#Add any points to the final img
	x = 0
	for i in range(Img_height):
		for j in range(Img_width):

			try:
				fimg = final_points(x, i, j, img, fimg)
			except:
				pass

	z = 0
	#(np.array_equal(fimg, img) == False)
	while(np.array_equal(fimg, img) != True):
		print("in the main loop", x)

		#The main for while loop

		#Check for contour points here
		for i in range(Img_height):
			for j in range(Img_width):

				try:
					#print("Checking contur points")
					cimg, img = contur_points(x, i, j, img, cimg)
				except:
					pass
			


		
		# Removing the contur points from the orignal image and re adding the final points
		for i in range(Img_height):
			for j in range(Img_width):

				#Checks the cimg for the new contur points
				try:
					if(cimg[i, j] != 255):
						#print("Removed the contour point")
						img[i,j] = 0
					else:
						pass
				except:
					pass

				#checks to make sure all the final points are still in the orignial image
				try:
					if(fimg[i,j] == 255):
						img[i,j] =255
						#print("maybe error")
					else:
						pass
				except:
					pass



		x = (x+1)%4

		for i in range(Img_height):
			for j in range(Img_width):
				try:
					fimg = final_points(x, i, j, img, fimg)
				except:
					pass

		

	return fimg


def contur_points(x, i, j, img, cimg):
	print("Enter contur_points", x)

	# postion in a 3 by 3 grid 1 is top left
	pos1 = (img[i-1, j-1]) #top left
	pos2 = (img[i-1, j]) #top middle
	pos3 = (img[i-1, j+1]) #top right
	pos4 = (img[i, j-1]) #center left
	pos5 = (img[i, j]) #center
	pos6 = (img[i, j+1]) # center right
	pos7 = (img[i+1, j-1]) #bottom left
	pos8 = (img[i+1, j]) #bottom middle
	pos9 = (img[i+1, j+1]) #bottom right

	if(pos5 == 255):
		#print("Enter the contour loop")

		#lower
		if((x == 0) and (pos8 == 0)):
			cimg[i,j] = 0
			#print("found a contour")
			return cimg, img

		#upper
		elif((x == 1) and (pos2 == 0)):
			cimg[i,j] = 0
			#print("found a contour")
			return cimg, img
		

		#left
		elif((x == 2) and (pos4 == 0)):
			cimg[i,j] = 0
			#print("found a contour")
			return cimg, img

		#right
		elif((x == 3) and (pos6 == 0)):
			cimg[i,j] = 0
			#print("found a contour")
			return cimg, img

		else:
			pass
		#There where no contour points at that location
	else:
		pass




def final_points(x, i, j, img, fimg):
	#Caculate final points
	#print("Enter final points", j)


	# postion in a 3 by 3 grid 1 is top left
	pos1 = int(img[i-1, j-1]) #top left
	pos2 = int(img[i-1, j]) #top middle
	pos3 = int(img[i-1, j+1]) #top right
	pos4 = int(img[i, j-1]) #center left
	pos5 = int(img[i, j]) #center
	pos6 = int(img[i, j+1]) # center right
	pos7 = int(img[i+1, j-1]) #bottom left
	pos8 = int(img[i+1, j]) #bottom middle
	pos9 = int(img[i+1, j+1]) #bottom right

	#Runs though the ai 
	if pos5 == 255:
		#Center pixel is black

		#a1
		if ((pos4 == 0) and (pos6 == 0)) and (((pos1 + pos2 + pos3) >254) and ((pos7 + pos8 + pos9) > 254)):
			#add center pixel to final img
			fimg[i,j] = 255
			#print("found a final pixel")
			return fimg

		#check a2
		elif((pos2 == 0) and (pos8 == 0)) and (((pos1 + pos4 + pos7) > 254) and ((pos3 +pos6 + pos9) > 254)):
			fimg[i,j] = 255
			#print("found a final pixel")
			return fimg

		#check a3
		elif((pos1 == 0) and (pos9 == 0)) and (((pos4 + pos7 +pos8) > 254) and ((pos2 + pos3 +pos6) > 254)):
			fimg[i,j] = 255
			#print("found a final pixel")
			return fimg

		#check a4
		elif((pos3 == 0) and (pos7 == 0)) and (((pos1 + pos2+ pos4) > 254) and ((pos6 + pos8 + pos9) > 254)):
			fimg[i,j] = 255
			#print("found a final pixel")
			return fimg

		else:
			pass

	else:
		#returns something and exits
		pass

	# Checking all of the b's
	if pos5 == 255 and x==0:
		#check b1 and b2

		#b1
		if ((pos6 == 0 and pos7 == 0 and pos8 == 255) and (pos1 + pos2 + pos3) > 254):
			fimg[i,j] = 255
			#print("found a final pixel")
			return fimg
		#b2
		elif ((pos3 == 0 and pos6 == 255 and pos8 == 0) and (pos1 + pos4 + pos7) > 254):
			fimg[i,j] = 255
			#print("found a final pixel")
			return fimg

		else:
			pass

	elif pos5 == 255 and x==1:
		#check b3 and b4

		#b3
		if((pos2 == 255 and pos3 == 0 and pos4 == 0) and (pos7 + pos8 + pos9) > 254):
			fimg[i,j] = 255
			#print("found a final pixel")
			return fimg

		#b4
		elif ((pos2 == 0 and pos4 == 255 and pos7 == 0) and (pos3 + pos6 + pos9) > 254):
			fimg[i,j] = 255
			#print("found a final pixel")
			return fimg

		else:
			pass

	elif pos5 == 255 and x==2:
		#checks b1 and b4

		#b1
		if ((pos6 == 0 and pos7 == 0 and pos8 == 255) and (pos1 + pos2 + pos3) > 254):
			fimg[i,j] = 255
			#print("found a final pixel")
			return fimg

		#b4
		elif ((pos2 == 0 and pos4 == 255 and pos7 == 0) and (pos3 + pos6 + pos9) > 254):
			fimg[i,j] = 255
			#print("found a final pixel")
			return fimg

		else:
			pass

	elif pos5 == 255 and x==3:
		#check for b2 and b3

		#b2
		if ((pos3 == 0 and pos6 == 255 and pos8 == 0) and (pos1 + pos4 + pos7) > 254):
			fimg[i,j] = 255
			#print("found a final pixel")
			return fimg

		#b3
		elif((pos2 == 255 and pos3 == 0 and pos4 == 0) and (pos7 + pos8 + pos9) > 254):
			fimg[i,j] = 255
			#print("found a final pixel")
			return fimg

		else:
			pass

	else:
		pass



	return fimg

def line_detection(final_img):

	Img_width = final_img.shape[1]
	Img_height = final_img.shape[0]

	#Creates the accumlator image
	aimg = np.zeros([Img_height, Img_width], dtype=int)

	for i in range(Img_height):
		for j in range(Img_width):

			if (final_img[i, j] != 255):
				for x in range 180: #increamnt x by 5
					

					x = x+5




if __name__ == '__main__':

	img_raw = cv2.imread(img_path1, 0)

	histImg = applyHisto(img_raw)

	edgeImg = edge_dection(histImg, 100)
	cv2.imshow('Edge dection',edgeImg)

	thin_img = edge_thining(edgeImg)
	final_image = line_detection()
	thin_img = cv2.normalize(src=thin_image, dst=None, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	cv2.imshow('Thinnig img', thin_image)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
