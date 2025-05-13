import cv2
import os
import sys
import time

def print_usage_and_exit():
    print("Usage: python collect_data.py <label_name> <num_samples>")
    sys.exit(1)

#collect arguments
if len(sys.argv) != 3:
    print("Error: Missing arguments.")
    print_usage_and_exit()

label_name, num_samples = sys.argv[1], sys.argv[2]
if not num_samples.isdigit():
    print("Error: <num_samples> must be an integer.")
    print_usage_and_exit()
num_samples = int(num_samples)

#create directory for storing data
data_dir = os.path.join('image_data', label_name)
os.makedirs(data_dir, exist_ok=True)

#set up webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible.")
    sys.exit(1)

print("Press 'a' to start/stop collecting data, 'q' to quit.")

start = False
count = 0
last_capture_time = time.time()

while count < num_samples:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        continue

    #draw ROI rectangle
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)

    # Overlay text
    status_text = "Collecting..." if start else "Paused"
    cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if start else (0, 0, 255), 2)
    cv2.putText(frame, f"Label: {label_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Collected: {count}/{num_samples}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        start = not start
        time.sleep(0.2)  # prevent accidental double toggle
    elif key == ord('q'):
        break


    if start and (time.time() - last_capture_time >= 0.25) and count < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        roi = frame[100:500, 100:500]
        save_path = os.path.join(data_dir, f"{count+1}.jpg")
        cv2.imwrite(save_path, roi)
        count += 1
        last_capture_time = time.time()
        print(f"Saved image {count}/{num_samples}")

#cleanup
cap.release()
cv2.destroyAllWindows()
print(f"\nFinished collecting {count} images for label: {label_name}")
