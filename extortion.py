import cv2
import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from PIL import Image
# %matplotlib auto
files = []
def load_images(Folder):
    images = []
    for filename in os.listdir(Folder):
        img = cv2.imread(os.path.join(Folder + filename))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        if img is not None:
            images.append(img)
        files.append(filename)
    return images

dir = "train/"
images = load_images(dir)

for i in range(len(images)):
    img = images[i]
    newimg =  np.zeros((img.shape), np.uint8)
    newimg[:] = img[0][0]
    
    img = cv2.subtract(img, newimg)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images[i] = img
x=3
y=3
itr=8
kernel = np.ones((x,y), np.uint8) 
for i in range (len(images)):
    images[i] = cv2.erode(images[i], kernel, iterations=itr) 
    # images[i] = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"
OUTPUT_FOLDER = "extracted_letter_images"


os.system('mkdir -p ' +CAPTCHA_IMAGE_FOLDER)
os.system('mkdir -p '+OUTPUT_FOLDER)


for i in range(len(images)):
    path = os.path.join(CAPTCHA_IMAGE_FOLDER, files[i])
    cv2.imwrite(path, images[i])



# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

x_start_MIN=165
x_end_MAX=0
# loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
    # grab the base filename as the text
    filename = os.path.basename(captcha_image_file)
    print(filename)
    captcha_correct_text = os.path.splitext(filename)[0]
    print(captcha_correct_text)
    # Load the image and convert it to grayscale
    image = cv2.imread(captcha_image_file)
    gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add some extra padding around the image
    gray = cv2.copyMakeBorder(gray2, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    
#     cv2.imshow("new image", gray)
#     cv2.waitKey(0); cv2.destroyAllWindows();
#     plt.imshow(gray)
    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
#     cv2.imshow("new image", thresh)
#     cv2.waitKey(0); cv2.destroyAllWindows();
    
    # find the contours (continuous blobs of pixels) the image
    #edged = cv2.Canny(gray, 30, 200) 
#     plt.imshow(thresh)

    ##########canny new#################
    sigma = 0.50
    v = np.median(thresh)

    #---- apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(thresh, lower, upper)
    # print(edged.shape)
    # cv2.imshow("aaa",edged)
    #################################
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    #contours = contours[0] if imutils.is_cv2() else contours[1]

    #####################################################################################
    ########################### CUSTOM CONTOUR ##########################################
    # _yLimit=thresh.shape[1]-1   #165
    # _xLimit=thresh.shape[0]-1   #615
    # print(_xLimit)
    # print(_yLimit)
    # y_axis_sum=0
    # _k=0
    # w, h = 200 , thresh.shape[0]
    # cstm_contour = np.zeros((w,h), np.uint8)
    # my_contours=[]
    # _i=0
    # for _i in range(0,_yLimit):
    # 	y_axis_sum=0
    # 	_j=0
    # 	for _j in range(0,_xLimit):
    # 		# print(str(_i) + " " + str(_j))
    # 		y_axis_sum+=thresh[_j][_i]
    # 	if(y_axis_sum > 0):
    # 		### Start partition ###
    # 		cstm_contour = np.zeros((w,h), np.uint8)
    # 		_k=0
    # 		while(y_axis_sum > 0):
    # 			y_axis_sum=0
    # 			for _j1 in range(0,_xLimit):
    # 				print(i,_j1)
    # 				y_axis_sum+=thresh[_j1][_i]
    # 			#for _i1 in range(0,_xLimit):
    # 				#cstm_contour[_i1][_k] = thresh[_i1][_i]
    # 			_i=_i+1
    # 			_k=_k+1
    # 		img = Image.fromarray(cstm_contour)
    # 		img.save('test'+str(_i)+'.png')

    # 		my_contours.append(cstm_contour)

    yLim = thresh.shape[0]
    xLim = thresh.shape[1]
    print(xLim,yLim)
    yLim_Sum = 0
    x = 0
    flag = 0

    x_coord = 0    
    y_coord = 0
    w_coord = 0
    h_coord = yLim-1

    HEIGHT_THRESH = 10
    THRESH_MINVALUE = yLim*255
    WIDTH_THRESH = 10
    letter=0
    # img = Image.fromarray(thresh)  #42075
    # img.save('thresh.png')
    print(thresh[0][0])
    while(xLim > x):
    	flag=0
    	yLim_Sum=0
    	for y in range(0,yLim):
    		yLim_Sum = yLim_Sum + thresh[y][x]
    		
    	if(yLim_Sum < THRESH_MINVALUE):
    		x_coord=x
    		while(yLim_Sum < THRESH_MINVALUE and x < xLim):
    			yLim_Sum = 0
    			for y1 in range(0,yLim):
    				yLim_Sum = yLim_Sum + thresh[y1][x]
    			x = x + 1
    			flag = 1
    		w_coord = x - x_coord
    		print(x_coord, y_coord ,w_coord)
    		print(thresh[y_coord][x_coord])
    	if(w_coord > WIDTH_THRESH and flag==1):
    		letter_text=captcha_correct_text[letter]
    		letter=letter+1
    		save_path = os.path.join(OUTPUT_FOLDER, letter_text)
    		count = counts.get(letter_text, 1)
    		p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
    		if not os.path.exists(save_path):
    			os.makedirs(save_path)
    		letter_image = thresh[y_coord+20 : y_coord + h_coord - 20, max(0,x_coord-10) :x_coord + w_coord + 10]
    		cv2.imwrite(p, letter_image)
    		counts[letter_text] = count + 1
    	if(flag==0):
    		x = x+1
#     xLim_sum=0
#     THRESH_MINVALUE = xLim*255
#     while(yLim > x):
#     	yLim_sum=0
#     	for y in range(0,xLim):
#     		yLim_sum = yLim_Sum+ thresh[x][y]
#     	if(yLim_Sum < THRESH_MINVALUE):
#     		x_start=x
#     	x = x+1
#     if(x_start_MIN > x_start ):
#     	x_start_MIN = x_start
#     x = 0
#     yLim=yLim-1
#     while(yLim >= x):
#     	yLim_sum=0
#     	for y in range(0,xLim):
#     		yLim_sum = yLim_Sum+ thresh[xLim][y]
#     	if(yLim_Sum < THRESH_MINVALUE):
#     		x_start=xLim
#     if(x_end_MAX < x_start ):
#     	x_end_MAX = x_start
    

# print(x_start_MIN)
# print(x_end_MAX)
    ########################### CUSTOM CONTOUR ##########################################
    #####################################################################################

#     letter_image_regions = []
#     letter_image_regions1 = []
#     letter_image_regions2=[]
#     #print(len(contours[0]))
#     # Now we can loop through each of the four contours and extract the letter
#     # inside of each one
    
#     for contour in contours[0]:
#         #contour = i
#         # Get the rectangle that contains the contour
#         (x, y, w, h) = cv2.boundingRect(contour)
#         # print(x, y , w, h)
        

#         # Compare the width and height of the contour to detect letters that
#         # are conjoined into one chunk
#         if w < 15 or h < 40:
#              continue
# #             # This contour is too wide to be a single letter!
# #             # Split it in half into two letter regions!
# #             half_width = int(w / 2)
# #             letter_image_regions.append((x, y, half_width, h))
# #             letter_image_regions.append((x + half_width, y, half_width, h))
#         else:
#             # This is a normal letter by itself
#             letter_image_regions1.append((x, y, w, h))

#     # If we found more or less than 4 letters in the captcha, our letter extraction
#     # didn't work correcly. Skip the image instead of saving bad training data!
#     #print(letter_image_regions)
#     # if len(letter_image_regions1) > 4:
#     #     continue
   
# ##############################modification for intersecting boxes######################    
#     for i in range(len(letter_image_regions1)):
#         (x, y , w, h) = letter_image_regions1[i]
#         j = (i+1)%len(letter_image_regions1)
#         flag = False
#         while(j != i):
#             (x1, y1, w1, h1) = letter_image_regions1[i]
#             if((x+w <= x1+w1) and (y+h <= y1+h1)):
#                 flag = True
#             j = (j+1)%len(letter_image_regions1)
#         if (flag == True):
#             letter_image_regions2.append((x, y, w, h))
# #######################################################################################
#     if (len(letter_image_regions2) > 4):
#     	print("contours:", len(letter_image_regions2))
#     	total_with_contuor=[]
#     	total=[]
#     	sumation=0;
#     	for k in range(0,len(letter_image_regions2)):
#     		sumation=0;
#     		(x, y , w, h) = letter_image_regions2[k]
#     		# print(str(x) + "-->" + str(x+w))
#     		# print(str(y) + "-->" + str(y+h))
#     		for _i in range(x,x+w):
#     			for j in range(y,y+h):
#     				# print(str(_i)+" _i j "+str(j))
#     				sumation+=(edged[j][_i])
#     		total_with_contuor.append((x,y,w,h,sumation))
#     		total.append(sumation)
#     	total.sort()
#     	print(len(total))
#     	fourth_largest=total[len(total)-4]
#     	for _i in range(0,len(total_with_contuor)):
#     		(x, y , w, h, t) = total_with_contuor[_i]
#     		if(t < fourth_largest):
#     			continue
#     		else:
#     			letter_image_regions.append((x,y,w,h))

#     # Sort the detected letter images based on the x coordinate to make sure
#     # we are processing them from left-to-right so we match the right image
#     # with the right letter
#     letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    
#     # Save out each letter as a single image
#     for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
#         # Grab the coordinates of the letter in the image
#         x, y, w, h = letter_bounding_box

#         # Extract the letter from the original image with a 2-pixel margin around the edge
#         letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

#         # Get the folder to save the image in
#         save_path = os.path.join(OUTPUT_FOLDER, letter_text)

#         # if the output directory does not exist, create it
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)

#         # write the letter image to a file
#         count = counts.get(letter_text, 1)
#         # p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
#         p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
#         cv2.imwrite(p, letter_image)

#         # increment the count for the current key
#         counts[letter_text] = count + 1
# #     break