import cv2
import numpy as np
import time

img = cv2.imread("desk4.jpg")
img = img[..., ::-1]

HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

HUE = HLS[:, :, 0]              # Split attributes
LIGHT = HLS[:, :, 1]
SAT = HLS[:, :, 2]

cond1 = SAT < 30
cond2 = (HUE > 100) | (HUE < 130)
cond3 = (LIGHT > 120)
mask = cond1 & cond2 & cond3
img2 = img.copy()

mask_int = mask.astype(np.uint8)
kernel = np.ones((17,17))
mask_int = cv2.morphologyEx(mask_int, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((15,15))
mask_int = cv2.morphologyEx(mask_int, cv2.MORPH_OPEN, kernel)
img2 = img.copy()
img2[mask_int == 1] = 0

contours, hierarchy = cv2.findContours(mask_int, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img_cnt = img.copy()

area = 0
i = 0

for cnt in contours:
    aaa = cv2.contourArea(cnt)
    if aaa > area:
        area = aaa
        i_max = i
    i += 1

if len(contours) > 0:
    hull = [cv2.convexHull(contours[i_max])]
    epsilon = 0.3 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(contours[i_max], epsilon, True)
    pr = cv2.contourArea(approx) / (img.shape[0] * img.shape[1]) * 100
else:
    hull = []
    cnt = []
    approx = []
    pr = 0

color_contours = (0, 255, 0) # green - color for contours
color = (255, 0, 0) # blue - color for convex hull
img_cnt = img.copy()

# draw contour
#cv2.drawContours(img_cnt, contours, i_max, color_contours, 1, 8, hierarchy)
# draw ith convex hull object
#cv2.drawContours(image=img_cnt, contours=hull,contourIdx=-1, color=color, thickness=2, lineType=8)

print(pr)

if pr < 7:
    HUE = HLS[:, :, 0]  # Split attributes
    LIGHT = HLS[:, :, 1]
    SAT = HLS[:, :, 2]

    mask1 = mask

    b1 = (HUE < 16) & (HUE > 12)
    b2 = (SAT > 90) | (SAT < 130)
    cond3 = LIGHT > 100
    mask = b1 & b2 & cond3 & (1 - mask1)
    img2 = img.copy()
    # img2[mask] = 0

    mask_int = mask.astype(np.uint8)
    kernel = np.ones((17, 17))
    mask_int = cv2.morphologyEx(mask_int, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((15, 15))
    mask_int = cv2.morphologyEx(mask_int, cv2.MORPH_OPEN, kernel)
    img2 = img.copy()
    img2[mask_int == 1] = 0

    contours, hierarchy = cv2.findContours(mask_int, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    area = 0
    i = 0
    # img_cnt = img.copy()
    for cnt in contours:
        # epsilon = 0.0001*cv2.arcLength(cnt,True)
        # approx = cv2.approxPolyDP(cnt,epsilon,True)
        # cv2.drawContours(image=img_cnt, contours=[approx], contourIdx=-1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        aaa = cv2.contourArea(cnt)
        if aaa > area:
            area = aaa
            i_max = i
        i += 1

    hull = [cv2.convexHull(contours[i_max])]

    color_contours = (0, 255, 0)  # green - color for contours
    color = (255, 0, 0)  # blue - color for convex hull

    img_cnt = img.copy()

    epsilon = 0.2 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(contours[i_max], epsilon, True)
    #cv2.drawContours(image=img_cnt, contours=[approx], contourIdx=-1, color=(0, 0, 255), thickness=2,
                     #lineType=cv2.LINE_AA)


img = img_cnt
flag = 0

with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

net = cv2.dnn.readNetFromDarknet('yolov4-tiny-custom.cfg', 'yolov4-tiny.weights')
#net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')

model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

classIds, scores, boxes = model.detect(img, confThreshold=0.001, nmsThreshold=0.001)
last_time = time.time()
for (classId, score, box) in zip(classIds, scores, boxes):
    if classes[classId] == 'cat':
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                      color=(0, 255, 0), thickness=2)
        XX = int(box[0] + box[2])
        YY = int(box[1] + box[3])
        center_coordinates = (XX, YY)
        #print(center_coordinates)

        # Radius of circle
        #radius = 20

        # Blue color in BGR
        #color = (255, 0, 0)

        # Line thickness of 2 px
        #thickness = 2

        # Using cv2.circle() method
        # Draw a circle with blue line borders of thickness of 2 px
        #img = cv2.circle(img, center_coordinates, radius, color, thickness)
        ctr = np.array(hull).reshape((-1,1,2)).astype(np.int32)
        if cv2.pointPolygonTest(ctr, (XX,YY), False) >= 0:
            text = '%s: %.2f ALARM' % (classes[classId], score)
            print(score)
            cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 5,
                    color=(255, 0, 0), thickness=10)
            flag = 1
        else:
            text = '%s: %.2f,' % (classes[classId], score)
            print(score)
            cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 5,
                    color=(0, 255, 0), thickness=10)

if flag == 0:
    text = 'Everything is OK'
    cv2.putText(img, text, (img.shape[0]//4, img.shape[0]//2-30), cv2.FONT_HERSHEY_SIMPLEX, 4,
                color=(0, 0, 255), thickness=10)
print('Loop took {} seconds',format(time.time()-last_time))
cv2.drawContours(image=img_cnt, contours=hull,contourIdx=-1, color=color, thickness=2, lineType=8)
img_cnt = img_cnt[..., ::-1]
resized = cv2.resize(img_cnt, (1200,900), interpolation = cv2.INTER_AREA)
#cv2.imshow('Image', img_cnt)
cv2.imshow('Image', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()