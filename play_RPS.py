import cv2
import numpy as np
from keras.models import load_model
from random import choice

#load
model = load_model("rock-paper-scissors-model.h5")
CLASS_NAMES = ["rock", "paper", "scissors", "none"]

#calculating winner logic
def calculate_winner(user, computer):
    if user == computer:
        return "Tie"
    rules = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
    return "User" if rules[user] == computer else "Computer"

#starts the webcam
cap = cv2.VideoCapture(0)
prev_move = None
winner = "Waiting..."
computer_move = "none"

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)  # User
    cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)  # Computer

    #get's human move
    roi = frame[100:500, 100:500]
    img = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), (227, 227))
    pred = model.predict(np.array([img]), verbose=0)
    user_move = CLASS_NAMES[np.argmax(pred[0])]

    #updates move and calls calulate winner
    if user_move != prev_move and user_move != "none":
        computer_move = choice(CLASS_NAMES[:-1])  
        winner = calculate_winner(user_move, computer_move)
    prev_move = user_move

    #displaying 
    cv2.putText(frame, f"Your Move: {user_move}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(frame, f"Computer's Move: {computer_move}", (750, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(frame, f"Winner: {winner}", (400, 600), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    #shows the computer's move
    if computer_move != "none":
        icon = cv2.imread(f"computer_move_images/{computer_move}.png")
        if icon is not None:
            icon = cv2.resize(icon, (400, 400))
            frame[100:500, 800:1200] = icon

    cv2.imshow("Rock Paper Scissors", frame)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
