import cv2
import numpy as np

#####################################
# get yolov3.weight in https://drive.google.com/file/d/1ZJ84QxFpzuDZ6wsNVGawDA3FXja96yOM/view?usp=sharing
#####################################
# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[int(i) - 1]
                 for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

img = cv2.VideoCapture(0)
img.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
img.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
width, height = 640, 480
while cv2.waitKey(33) < 0:
    # Loading image
    # img = cv2.imread("room_ser.jpg")
    # img = cv2.resize(img, None, fx=0.4, fy=0.4)
    # height, width, channels = img.shape
    ret, frame = img.read()
    frame = cv2.resize(frame, (640, 480))
    # Detecting objects
    blob = cv2.dnn.blobFromImage(
        frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

    # cv2.imshow("Image", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        cv2.imshow("Image", frame)
cv2.destroyAllWindows()
