import cv2
import xml.etree.ElementTree as ET
import json
import numpy as np

IMAGE_PATH = "dataset/PKLot/PUCPR/Sunny/2012-09-11/2012-09-11_16_24_53.jpg"
XML_PATH   = "dataset/PKLot/PUCPR/Sunny/2012-09-11/2012-09-11_16_24_53.xml"


image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Error: Image not found! Check your path.")
    exit()

print("Image loaded successfully!")


tree = ET.parse(XML_PATH)
root = tree.getroot()

slots = []

for space in root.findall('space'):

    
    slot_id = int(space.get('id'))

    
    occupied = space.get('occupied')
    if occupied is None:
        occupied = '0'              
    occupied = int(occupied)

    contour_points = []
    for point in space.find('contour').findall('point'):
        x = int(point.get('x'))
        y = int(point.get('y'))
        contour_points.append([x, y])

    
    slots.append({
        'id'       : slot_id,
        'occupied' : occupied,
        'contour'  : contour_points
    })

print(f"Total slots found: {len(slots)}")


output_image = image.copy()

empty_count    = 0
occupied_count = 0

for slot in slots:
    
    pts = np.array(slot['contour'], np.int32)
    pts = pts.reshape((-1, 1, 2))

    if slot['occupied'] == 0:
        color = (0, 255, 0)        
        empty_count += 1
    else:
        color = (0, 0, 255)       
        occupied_count += 1

    
    cv2.polylines(output_image, [pts], isClosed=True, color=color, thickness=2)

  
    cx = slot['contour'][0][0]
    cy = slot['contour'][0][1]
    cv2.putText(output_image, str(slot['id']), (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


cv2.putText(output_image, f"Empty: {empty_count}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(output_image, f"Occupied: {occupied_count}",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

print(f"Empty slots    : {empty_count}")
print(f"Occupied slots : {occupied_count}")

with open("slots.json", "w") as f:
    json.dump(slots, f, indent=4)
print("Slots saved to slots.json!")


cv2.imshow("Parking Slot Detection", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()