from ultralytics import YOLO
import cv2
from playsound import playsound
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv()

def send_email_alert(frame_number, bag_id):
    sender_email = os.getenv('EMAIL_USER')
    receiver_email = "faizzameer16@gmail.com"
    password = os.getenv('EMAIL_PASS')  

    subject = "Abandoned Bag Alert!"
    body = f"An abandoned bag (ID: {bag_id}) was detected at frame {frame_number}."

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print(" Email alert sent!")
    except Exception as e:
        print(" Error sending email:", e)


model = YOLO('yolov8l.pt')

BAG_CLASSES = ['backpack', 'suitcase', 'handbag', 'bag']
DISTANCE_THRESHOLD = 75
ABANDON_THRESHOLD = 30  
ALERT_SOUND = 'alert.mp3'

cap = cv2.VideoCapture(0)

bag_tracks = {}
alerted_ids = set()  
frame_count = 0

def play_alert():
    playsound(ALERT_SOUND)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = model.track(frame, persist=True, conf=0.3)
    annotated_frame = results[0].plot()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            xyxy = box.xyxy[0]
            x1, y1, x2, y2 = map(int, xyxy)
            track_id = int(box.id[0]) if box.id is not None else None

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            print(f"Detected: {label} (ID: {track_id}) at ({center_x}, {center_y})")

            if label in BAG_CLASSES and track_id is not None:
                if track_id not in bag_tracks:
                    bag_tracks[track_id] = {'frames': 0, 'position': (center_x, center_y)}
                else:
                    bag_tracks[track_id]['frames'] += 1
                    bag_tracks[track_id]['position'] = (center_x, center_y)

                near_person = False
                for other in boxes:
                    other_cls = int(other.cls[0])
                    if model.names[other_cls] == 'person':
                        ox1, oy1, ox2, oy2 = map(int, other.xyxy[0])
                        person_center = ((ox1 + ox2) // 2, (oy1 + oy2) // 2)
                        dist = ((center_x - person_center[0]) ** 2 + (center_y - person_center[1]) ** 2) ** 0.5
                        if dist < DISTANCE_THRESHOLD:
                            near_person = True
                            bag_tracks[track_id]['frames'] = 0  
                            break

                
                if (not near_person and 
                    bag_tracks[track_id]['frames'] >= ABANDON_THRESHOLD and 
                    track_id not in alerted_ids):

                    print(f"[ALERT] Abandoned bag detected! Track ID: {track_id} at frame {frame_count}")
                    cv2.putText(annotated_frame, "ALERT: Abandoned Bag!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                    
                    if 'position' in bag_tracks[track_id]:
                        cv2.circle(annotated_frame, bag_tracks[track_id]['position'],
                                   DISTANCE_THRESHOLD, (0, 0, 255), 3)

                    
                    threading.Thread(target=play_alert, daemon=True).start()

                    
                    with open('alerts_log.txt', 'a') as log_file:
                        log_file.write(f"ALERT at frame {frame_count}: {label} ID {track_id}\n")

                    
                    alert_filename = f"alert_frame_{frame_count}.jpg"
                    cv2.imwrite(alert_filename, annotated_frame)
                    print(f"Saved alert frame as {alert_filename}")

                    threading.Thread(target=send_email_alert, args=(frame_count, track_id), daemon=True).start()
                    alerted_ids.add(track_id)

    cv2.imshow('Abandoned Bag Detector', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
