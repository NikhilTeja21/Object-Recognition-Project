import cv2
import numpy as np

# Load YOLOv4 model and configuration files
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize dictionary to keep track of detected objects count
detected_objects_count = {class_name: 0 for class_name in classes}

# Load video
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

# Resize window
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", 800, 1000)

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
    # Get the indices of the output layers
    output_layer_indices = net.getUnconnectedOutLayers()

    # Get the names of the output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in output_layer_indices]

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
                detected_objects_count[class_name] += 1

                # Draw bounding box around the object
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Write counts for each class to a text file
output_file = "output.txt"
with open(output_file, "w") as f:
    for class_name, count in detected_objects_count.items():
        if count==0 : continue
        else : f.write(f"Number of {class_name}s detected: {count}\n")

print(f"Counts written to {output_file}")
