import cv2
import os
import sys

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

#initialize video capture
cap = cv2.VideoCapture(0)
start = False
count = 0

#create frame and collect data
while count < num_samples:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        continue
    
    #draw capture rectangle
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)

    if start:
        roi = frame[100:500, 100:500]
        cv2.imwrite(os.path.join(data_dir, f"{count + 1}.jpg"), roi)
        count += 1
    
    cv2.putText(frame, "Collecting data...", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Collecting data', frame)

    key = cv2.waitKey(10)
    if key == ord('a'):
        start = not start
    elif key == ord('q'):
        break
#end program
print(f"Collected {count} images for {label_name}.")
cap.release()
cv2.destroyAllWindows()