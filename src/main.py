import os
import cv2
from ultralytics import YOLO

# 1. Initialize the YOLO model (nano version is fast and lightweight)
model = YOLO("yolov8n.pt")

# 2. Setup paths
video_path = "data/input_vid/sample.mp4" 
output_folder = "data/output"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 3. Open the video file
cap = cv2.VideoCapture(video_path)
frame_count = 0
crop_count = 0

print("Processing video... Press 'q' to quit early.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break  # End of video

    frame_count += 1
    
    # Process every Nth frame to avoid saving thousands of nearly identical images
    # e.g., checking 1 frame per second on a 30fps video
    if frame_count % 30 != 0:
        continue

    # Run YOLO object detection on the frame
    results = model(frame, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class ID (e.g., 0 is usually 'person' in COCO dataset)
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Filter for 'person' (Class 0) with a confidence threshold above 50%
            if class_id == 0 and confidence > 0.5:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Crop the detected object from the frame
                cropped_img = frame[y1:y2, x1:x2]

                # Ensure the crop is valid (not empty)
                if cropped_img.size > 0:
                    crop_count += 1
                    output_path = os.path.join(output_folder, f"person_{crop_count}.jpg")
                    cv2.imwrite(output_path, cropped_img)

    # Optional: Display the frame while processing
    cv2.imshow("Security Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print(f"Processing complete! Saved {crop_count} cropped images to '{output_folder}'.")