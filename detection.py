import cv2
import numpy as np

# Load YOLOv4 model and configuration files
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize counters for each class
class_counts = {class_name: 0 for class_name in classes}

# Load video
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set blob as input to the network
    net.setInput(blob)

    # Perform forward pass
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    # Process detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Increment count for detected class
                class_name = classes[class_id]
                class_counts[class_name] += 1

    # Show the frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print counts for each class
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")
