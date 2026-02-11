import mediapipe as mp

print("MediaPipe version:", mp.__version__)

# Try to access solutions
try:
    mp_hands = mp.solutions.hands
    print("SUCCESS: mp.solutions.hands loaded!")
    print("Hands module:", mp_hands)
except AttributeError as e:
    print("Still failed:", e)