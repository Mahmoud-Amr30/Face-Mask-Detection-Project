import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv11 model
model = YOLO("C:/Users/lordm/Desktop/facedetect/best.pt")  # Specify the path to the trained YOLOv11 model file (best.pt) to load it for inference

# Open the webcam
cap = cv2.VideoCapture(0)  # Use index 0 to access the default webcam for capturing video feed

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")  # Print an error message if the webcam fails to open, indicating a hardware or permission issue
    exit()  # Exit the program to avoid further execution

# Start an infinite loop to process video frames
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()  # Capture a single frame; ret is a boolean (True if frame is read successfully), frame is the image
    if not ret:
        print("Error: Failed to read frame")  # Print an error if frame capture fails, e.g., due to disconnection or end of stream
        break  # Exit the loop to stop processing

    # Run the YOLOv11 model on the frame
    results = model(frame)  # Perform inference on the frame to detect objects (faces with/without masks) and get detection results

    # Process the detection results
    for result in results:
        boxes = result.boxes  # Extract bounding boxes from the result, containing coordinates, confidence, and class info
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert box coordinates (top-left and bottom-right) to integers for drawing
            conf = box.conf[0]  # Get the confidence score of the detection (how certain the model is)
            cls = int(box.cls[0])  # Get the class index (e.g., 0 for "mask", 1 for "no_mask")
            label = model.names[cls]  # Get the class name (e.g., "mask" or "no_mask") from the model’s class dictionary

            # Determine the text and color based on the class
            if label == "mask":
                color = (0, 255, 0)  # Set green color for "mask" (with_mask) to indicate a positive detection
                text = f"with_mask ({conf:.2f})"  # Create label text with "with_mask" and confidence score (e.g., "with_mask (0.95)")
            else:
                color = (0, 0, 255)  # Set red color for "no_mask" to indicate absence of a mask
                text = f"no_mask ({conf:.2f})"  # Create label text with "no_mask" and confidence score (e.g., "no_mask (0.90)")

            # Draw the bounding box and text on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Draw a rectangle around the detected face with the specified color and thickness of 2 pixels
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Draw the label text above the box with font size 0.9 and thickness 2

    # Display the processed frame
    cv2.imshow("Mask Detection", frame)  # Show the frame with bounding boxes and labels in a window named "Mask Detection"

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Wait 1ms for a key press; if 'q' is pressed, break the loop to exit
        break  # Exit the loop to stop the program

# Release resources
cap.release()  # Release the webcam to free up the camera resource
cv2.destroyAllWindows()  # Close all OpenCV windows to clean up the display
