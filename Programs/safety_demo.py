import cv2

# Load the image
img = cv2.imread('strawberry.jpeg')

if img is None:
    print("Image not found or failed to load")
else:
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
