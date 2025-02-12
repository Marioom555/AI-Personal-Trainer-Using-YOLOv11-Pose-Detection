import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import threading
import queue
import cvzone
import speech_recognition as sr

# Initialize the YOLO model and video capture
model = YOLO('yolo11n-pose.pt')
cap = cv2.VideoCapture(0)

up_thresh = 150
down_thresh = 90
push_up_left = False
push_up_right = False
combine = False
left_hand_counter = 0
right_hand_counter = 0
combine_counter = 0
mode = None

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('rate', 150)
engine.setProperty('voice', voices[1].id)
speech_queue = queue.Queue()

def listen_commands():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    while True:  
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                print("Listening...")
                audio = recognizer.listen(source)
                commands = recognizer.recognize_google(audio).lower()
                print(commands)
                if 'normal' in commands:
                    speak("Normal mode started")
                    set_mode('normal')
                elif 'combine' in commands:
                    speak("Combine mode started")
                    set_mode('combine')
                elif 'stop' in commands:
                    speak("Take care and goodbye")
                    set_mode('stop')
                    break
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError:
            print("Could not request results; check your network connection")

def set_mode(new_mode):
    global mode
    mode = new_mode
    
    
def speak(text):
    speech_queue.put(text)
    
def worker_speak():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Start threads
thread_speak = threading.Thread(target=worker_speak, daemon=True) # control the thread in the background 
thread_speak.start()
thread_listen = threading.Thread(target=listen_commands, daemon=True)
thread_listen.start()

# Main loop for video capture and processing
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 500))
    
    # Make predictions
    result = model.track(frame)
    
    if result[0].keypoints is not None:
        keypoints = result[0].keypoints.xy.cpu().numpy() # [[[]]]
        
        for keypoint in keypoints: #[[]]
            if len(keypoint) > 0:  # Check for presence of keypoints
                for i, point in enumerate(keypoint): #[]
                    cx, cy = int(point[0]), int(point[1])
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                    cvzone.putTextRect(frame, f'{i}', (cx, cy), 1, 1) 
                
                # Check if the necessary keypoints are available for calculating angles
                if mode and len(keypoint) > 10:
                    left_shoulder = (int(keypoint[5][0]), int(keypoint[5][1]))
                    left_elbow = (int(keypoint[7][0]), int(keypoint[7][1]))
                    left_wrist = (int(keypoint[9][0]), int(keypoint[9][1]))
                    
                    right_shoulder = (int(keypoint[6][0]), int(keypoint[6][1]))
                    right_elbow = (int(keypoint[8][0]), int(keypoint[8][1]))
                    right_wrist = (int(keypoint[10][0]), int(keypoint[10][1]))
                    
                    left_hand_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    right_hand_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    
                    cvzone.putTextRect(frame, f'Left Arm Angle: {int(left_hand_angle)}', (50, 50), 1, 1, colorR=(255, 0, 0))
                    cvzone.putTextRect(frame, f'Right Arm Angle: {int(right_hand_angle)}', (50, 80), 1, 1, colorR=(0, 255, 0))
                    
                    if mode == "normal":
                        combine_counter = 0
                        if left_hand_angle < down_thresh and not push_up_left:
                            push_up_left = True
                        elif left_hand_angle > up_thresh and push_up_left:
                            left_hand_counter += 1 
                            push_up_left = False
                            speak(f'Left {left_hand_counter}')
                        
                        if right_hand_angle < down_thresh and not push_up_right:
                            push_up_right = True
                        elif right_hand_angle > up_thresh and push_up_right:
                            right_hand_counter += 1 
                            push_up_right = False
                            speak(f'Right {right_hand_counter}')
                    
                    elif mode == "combine":   
                        right_hand_counter = 0
                        left_hand_counter = 0
                        if right_hand_angle <= down_thresh and left_hand_angle <= down_thresh and not combine:
                            combine = True
                        elif left_hand_angle >= up_thresh and right_hand_angle >= up_thresh and combine:
                            combine_counter += 1
                            combine = False
                            speak(f'Combine {combine_counter}')
                            
    if mode == 'normal':
        cvzone.putTextRect(frame, f'Left hand counter: {int(left_hand_counter)}', (50, 110), 1, 1, colorR=(0, 0, 0))
        cvzone.putTextRect(frame, f'Right hand counter: {int(right_hand_counter)}', (50, 140), 1, 1, colorR=(0, 0, 0))
    elif mode == 'combine':
        cvzone.putTextRect(frame, f'Combine counter: {int(combine_counter)}', (50, 170), 1, 1, colorR=(0, 0, 0))
        
    # Display the frame
    cv2.imshow("RGB", frame)

    # Exit on 'Esc' key press
    key = cv2.waitKey(1)
    if mode=='stop':
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
