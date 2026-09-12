import threading
import time
import os
import cv2

# Global state shared across your entire app
is_connected = True

def _check_wifi_state():
    global is_connected
    state_file = '/sys/class/net/wlan0/operstate'
    
    while True:
        try:
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = f.read().strip()
                    is_connected = (state == "up")
            else:
                is_connected = False
        except Exception:
            is_connected = False
            
        time.sleep(1)

# Start the daemon the moment this file is imported anywhere
wifi_thread = threading.Thread(target=_check_wifi_state, daemon=True)
wifi_thread.start()

def apply_wifi_warning(frame):
    """
    Takes a frame, draws the warning IF disconnected, and returns it.
    Does not modify the original frame if connected.
    """
    global is_connected
    if is_connected:
        return frame

    # Get frame dimensions
    h, w = frame.shape[:2]
    
    text = "NO WI-FI CONNECTION"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    
    # Center the text
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (h + text_size[1]) // 2

    # Draw background box & text
    pad = 10
    cv2.rectangle(frame, 
                  (text_x - pad, text_y - text_size[1] - pad), 
                  (text_x + text_size[0] + pad, text_y + pad), 
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, 
                (0, 0, 255), thickness, cv2.LINE_AA)
                
    return frame
