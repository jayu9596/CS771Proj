import os
import cv2
file_path = "train/"
i = 0
os.system("rm test")
os.system("mkdir test")
os.mknod("test/codes.txt")
f = open("test/codes.txt","a")
for  filepath in os.listdir(file_path):
	src = "test/image%d.png" % i 
	f.write(os.path.splitext(filepath)[0]  + "\n")
	i = i +1
	#print(filepath)
	image = cv2.imread(file_path+filepath)
	cv2.imwrite(src, image)
f.close()