import cv2
from matplotlib import pyplot as plt

# Making one variable which stores our image
image = cv2.imread('stop.jpg')

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

stop_data = cv2.CascadeClassifier('stop_data.xml')
found = stop_data.detectMultiScale(image_gray, minSize=(20,20))

amount_found = len(found)

if amount_found !=0:
    for (x,y,width,height) in found:
        cv2.rectangle(image_rgb,(x,y), (x + height, y+width), (0,255,0), 3)
        print("This is Stop Sign")
        image_text = cv2.putText(image_rgb, "This is stop sign", (150,430),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
plt.subplot(1,1,1)
plt.imshow(image_rgb)
plt.show()