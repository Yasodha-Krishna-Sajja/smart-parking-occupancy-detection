import cv2

image = cv2.imread("dataset/PKLot/PUCPR/Sunny/2012-09-11/2012-09-11_15_16_58.jpg")

if image is None:
    print("Error: Image not found! Check your path.")
else:
    print("Image loaded successfully!")
    print("Image Shape:", image.shape)   
    print("Height:", image.shape[0])
    print("Width:", image.shape[1])
    print("Channels:", image.shape[2])


cv2.imshow("Parking Lot", image)
cv2.waitKey(0)          
cv2.destroyAllWindows() 


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Grayscale Image Shape:", gray_image.shape)  


cv2.imshow("Grayscale Parking Lot", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


resized_image = cv2.resize(image, (640, 360))      
print("Resized Image Shape:", resized_image.shape)


cv2.imshow("Resized Parking Lot", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
