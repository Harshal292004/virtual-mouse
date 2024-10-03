import cv2
import numpy as np
import time 
import os 
from hand import handDetector
import pyautogui
import math 

# Constants
CURSOR_WIDTH = 30
CURSOR_HEIGHT = 30
SCROLL_THRESHOLD = 20
PINCH_THRESHOLD = 30
ANGLE_THRESHOLD_MIN = 20
ANGLE_THRESHOLD_MAX = 60
L_ANGLE_MIN = 75
L_ANGLE_MAX = 100

# Load cursor image
try:
    cursor = cv2.imread('Cursor/cursor.png', cv2.IMREAD_UNCHANGED)
    cursor = cv2.resize(cursor, (CURSOR_WIDTH, CURSOR_HEIGHT))
except Exception as e:
    print(f"Error loading cursor image: {e}")
    cursor = None

# Global variables
is_dragging = False
drag_start_x, drag_start_y = 0, 0
screen_width, screen_height = pyautogui.size()
scroll_mode = False
scroll_start_y = None
is_halt = False

# Setup
cap = cv2.VideoCapture(0)
pTime = 0
detector = handDetector(maxHands=1, detectionCon=0.9, trackCon=0.7)

def calculate_angle(a, b, c):
    ba = np.array([a[0]-b[0], a[1]-b[1]])
    bc = np.array([c[0]-b[0], c[1]-b[1]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def overlay_cursor(img, x, y, cursor, offset_x=(False,0), offset_y=(False,0)):
    if cursor is None:
        return img
    
    if offset_x[0]:
        x += offset_x[1]
    if offset_y[0]:
        y += offset_y[1]
    
    y1, y2 = max(y, 0), min(y + cursor.shape[0], img.shape[0])
    x1, x2 = max(x, 0), min(x + cursor.shape[1], img.shape[1])
    
    overlay = img[y1:y2, x1:x2]
    cursor_part = cursor[:y2-y1, :x2-x1]
    
    if cursor_part.shape[2] == 4:  # Check if cursor has an alpha channel
        alpha = cursor_part[:,:,3] / 255.0
        for c in range(3):
            overlay[:,:,c] = overlay[:,:,c] * (1-alpha) + cursor_part[:,:,c] * alpha
    else:
        overlay = cursor_part
    
    img[y1:y2, x1:x2] = overlay
    return img

def pinch(img, p1, p2, threshold=PINCH_THRESHOLD, draw_distance=False):
    distance = np.linalg.norm(np.array(p1) - np.array(p2))
    if draw_distance:
        cv2.putText(img, str(int(distance)), ((p1[0]+p2[0])//2-20, (p1[1]+p2[1])//2), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)
    return distance <= threshold

def move_cursor(img, fingers, lmList, prev_x, prev_y, draw_cursor=False):
    if tuple(fingers) == (0,1,0,0,0):
        x, y = lmList[8][1:]
        
        # Web cam dimensions
        frame_h, frame_w = img.shape[:2]

        # Map the coordinates from the webcam frame to the screen
        # Add padding to the frame to utilize the full screen
        padding = 100
        screen_x = np.interp(x, (padding, frame_w-padding), (0, screen_width))
        screen_y = np.interp(y, (padding, frame_h-padding), (0, screen_height))

        # Reduce smoothing and add speed multiplier
        smoothing = 2
        speed_multiplier = 1.5
        current_x = prev_x + (screen_x - prev_x) / smoothing
        current_y = prev_y + (screen_y - prev_y) / smoothing

        # Apply speed multiplier
        dx = (current_x - prev_x) * speed_multiplier
        dy = (current_y - prev_y) * speed_multiplier
        current_x = prev_x + dx
        current_y = prev_y + dy

        # Ensure the cursor stays within the screen bounds
        current_x = max(0, min(current_x, screen_width))
        current_y = max(0, min(current_y, screen_height))

        # Move the cursor
        pyautogui.moveTo(current_x, current_y)
        
        # Update previous positions
        prev_x, prev_y = current_x, current_y

        if draw_cursor:
            overlay_cursor(img, x, y, cursor, offset_y=(True,-20))
        
    return prev_x, prev_y

def left_click(img, fingers, lmList, draw=False):
    x5, y5 = lmList[5][1:]
    x8, y8 = lmList[8][1:]
    x12, y12 = lmList[12][1:]
    
    angle = calculate_angle([x8, y8], [x5, y5], [x12, y12])
    if draw:
        cv2.putText(img, str(int(angle)), (x5, y5-20), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

    if ANGLE_THRESHOLD_MIN < angle < ANGLE_THRESHOLD_MAX and tuple(fingers) == (0,1,1,0,0):
        pyautogui.click()

def right_click(fingers):
    if tuple(fingers) == (0,1,1,1,0):
        pyautogui.rightClick()

def double_click(fingers):
    if tuple(fingers) == (0,1,1,1,1):
        pyautogui.doubleClick()

def click_and_drag(img, is_dragging, drag_start_x, drag_start_y, lmList, draw=False, draw_cursor=False):
    x8, y8 = lmList[8][1:]
    x4, y4 = lmList[4][1:]
   
    is_pinched = pinch(img, [x8, y8], [x4, y4], threshold=PINCH_THRESHOLD, draw_distance=draw)
    
    if is_pinched:
        x_pinch, y_pinch = x4, y4
        if not is_dragging:
            is_dragging = True
            drag_start_x, drag_start_y = pyautogui.position()
            pyautogui.mouseDown()
        else:
            dx, dy = x_pinch - drag_start_x, y_pinch - drag_start_y
            new_x = min(max(dx + drag_start_x, 0), screen_width)
            new_y = min(max(dy + drag_start_y, 0), screen_height)
            pyautogui.moveTo(new_x, new_y)
        if draw_cursor:
            overlay_cursor(img, x_pinch, y_pinch, cursor)
    elif is_dragging:
        is_dragging = False
        pyautogui.mouseUp()
    
    return is_dragging, drag_start_x, drag_start_y

def scroll_up_and_down(img, lmList, fingers, scroll_start_y, scroll_threshold, scroll_mode=False, draw=False):
    x8, y8 = lmList[8][1:]
    x4, y4 = lmList[4][1:]
    x2, y2 = lmList[2][1:]
    
    L_angle = calculate_angle([x8, y8], [x2, y2], [x4, y4])

    if draw:
        cv2.putText(img, str(int(L_angle)), (x2, y2+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
            
    if L_ANGLE_MIN <= L_angle <= L_ANGLE_MAX and tuple(fingers) == (1, 1, 0, 0, 0):
        if not scroll_mode:
            scroll_mode = True
            scroll_start_y = y8
        else:
            y_diff = scroll_start_y - y8
            if abs(y_diff) > scroll_threshold:
                scroll_amount = int(y_diff)
                pyautogui.scroll(scroll_amount)
                scroll_start_y = y8
    else:
        scroll_mode = False
        scroll_start_y = None
    
    return scroll_mode, scroll_start_y

def zoom_in_out(img, fingers):
    if tuple(fingers) == (1,1,0,0,1):
        pyautogui.hotkey('ctrl', '=')
        time.sleep(0.5)
    elif tuple(fingers) == (1,1,1,1,1):
        pyautogui.hotkey('ctrl', '-')
        time.sleep(0.5)

def process_gestures(img, fingers, lmList, is_dragging, drag_start_x, drag_start_y, scroll_mode, scroll_start_y, prev_x, prev_y):
    if tuple(fingers) == (0,0,0,0,0):
        return True, is_dragging, drag_start_x, drag_start_y, scroll_mode, scroll_start_y, prev_x, prev_y
    
    prev_x, prev_y = move_cursor(img, fingers, lmList, prev_x, prev_y, True)
    left_click(img, fingers, lmList)
    right_click(fingers)
    double_click(fingers)
    is_dragging, drag_start_x, drag_start_y = click_and_drag(img, is_dragging, drag_start_x, drag_start_y, lmList)
    scroll_mode, scroll_start_y = scroll_up_and_down(img, lmList, fingers, scroll_start_y, SCROLL_THRESHOLD, scroll_mode)
    zoom_in_out(img, fingers)
    
    return False, is_dragging, drag_start_x, drag_start_y, scroll_mode, scroll_start_y, prev_x, prev_y

prev_x, prev_y = screen_width // 2, screen_height // 2 

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame")
        break

    img = cv2.flip(img, 1)

    detector.findHands(img)
    lmList = detector.findPosition(img)
    fingers = detector.fingersUp(lmList)

    if len(lmList) != 0:
        is_halt, is_dragging, drag_start_x, drag_start_y, scroll_mode, scroll_start_y, prev_x, prev_y = process_gestures(
            img, fingers, lmList, is_dragging, drag_start_x, drag_start_y, scroll_mode, scroll_start_y, prev_x, prev_y
        )
        if is_halt:
            break
            
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Virtual Mouse', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()