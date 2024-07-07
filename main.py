import cv2
import numpy as np

# Load the video
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

# Define the quadrants on the right half of the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

quad_width = frame_width // 4  # Width of each quadrant (480)
quad_height = frame_height // 2  # Height of each quadrant (540)

# Define quadrant boundaries (on the right side)
quadrants = [
    (frame_width - quad_width, 0, frame_width, quad_height),  # Quadrant 1
    (frame_width - quad_width, quad_height, frame_width, frame_height),  # Quadrant 2
    (frame_width - 2 * quad_width, 0, frame_width - quad_width, quad_height),  # Quadrant 3
    (frame_width - 2 * quad_width, quad_height, frame_width - quad_width, frame_height)  # Quadrant 4
]

# Function to detect balls by color
def detect_balls(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    colors = {
        'red': ((0, 100, 100), (10, 255, 255)),
        'blue': ((100, 150, 0), (140, 255, 255)),
        'green': ((40, 70, 70), (80, 255, 255))
    }
    
    balls = []

    for color_name, (lower, upper) in colors.items():
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius > 5:
                balls.append((int(x), int(y), color_name))
    
    return balls

# Function to determine which quadrant the ball is in
def get_quadrant(x, y):
    for i, (x1, y1, x2, y2) in enumerate(quadrants):
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i + 1
    return None

# Track the balls and record events
events = []
prev_positions = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds
    balls = detect_balls(frame)
    
    for x, y, color in balls:
        quadrant = get_quadrant(x, y)
        if color in prev_positions:
            prev_quad = prev_positions[color]
            if prev_quad != quadrant:
                events.append((timestamp, quadrant, color, 'Entry'))
                events.append((timestamp, prev_quad, color, 'Exit'))
        else:
            events.append((timestamp, quadrant, color, 'Entry'))
        prev_positions[color] = quadrant

cap.release()

# Function to overlay text on the video frames
def overlay_text(frame, text, pos):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, pos, font, 1, (0, 255, 255), 2, cv2.LINE_AA)

# Save the processed video
cap = cv2.VideoCapture(video_path)
out = cv2.VideoWriter('processed_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (int(cap.get(3)), int(cap.get(4))))

prev_positions = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    balls = detect_balls(frame)

    for x, y, color in balls:
        quadrant = get_quadrant(x, y)
        if color in prev_positions:
            prev_quad = prev_positions[color]
            if prev_quad != quadrant:
                overlay_text(frame, f'Exit: Q{prev_quad} - {timestamp:.2f}', (x, y - 10))
                overlay_text(frame, f'Entry: Q{quadrant} - {timestamp:.2f}', (x, y + 10))
        else:
            overlay_text(frame, f'Entry: Q{quadrant} - {timestamp:.2f}', (x, y + 10))
        prev_positions[color] = quadrant

    out.write(frame)

cap.release()
out.release()

# Save events to a text file
with open('events.txt', 'w') as f:
    for event in events:
        f.write(f"{event[0]:.2f}, {event[1]}, {event[2]}, {event[3]}\n")
