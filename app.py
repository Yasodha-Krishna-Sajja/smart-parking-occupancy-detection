from flask import Flask, render_template, request, jsonify, Response
import os
import cv2
import json
import numpy as np
import threading
import time
import sqlite3
from datetime import datetime
from tensorflow.keras.models import load_model

# ---- Create Flask App ----
app = Flask(__name__)

# ---- Upload Folder Settings ----
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---- Load Model and Slots Once (at startup) ----
print("Loading model...")
model = load_model("models/parking_model.h5")
print("Model loaded!")

with open("slots.json", "r") as f:
    slots = json.load(f)
print(f"Slots loaded: {len(slots)}")

# ---- Shared State (updated each frame) ----
current_stats = {
    "empty"    : 0,
    "occupied" : 0,
    "total"    : 100,
    "rate"     : 0
}
video_frames   = []
processing     = False
current_session_id = None

# ================================================
# DATABASE FUNCTIONS
# ================================================

DB_PATH = "parking_data.db"

# ---- Create Database and Tables ----
def init_database():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    # Sessions table → one row per video processed
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name   TEXT,
            date         TEXT,
            time         TEXT,
            total_frames INTEGER,
            avg_empty    REAL,
            avg_occupied REAL,
            avg_rate     REAL,
            peak_rate    REAL,
            min_rate     REAL
        )
    ''')

    # Frames table → one row per frame per session
    c.execute('''
        CREATE TABLE IF NOT EXISTS frames (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            frame_no   INTEGER,
            empty      INTEGER,
            occupied   INTEGER,
            rate       INTEGER
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized!")

# ---- Create New Session ----
def create_session(video_name):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    now  = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    c.execute('''
        INSERT INTO sessions
        (video_name, date, time, total_frames,
         avg_empty, avg_occupied, avg_rate,
         peak_rate, min_rate)
        VALUES (?, ?, ?, 0, 0, 0, 0, 0, 100)
    ''', (video_name, date, time_str))

    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return session_id

# ---- Save Frame Data ----
def save_frame(session_id, frame_no, empty, occupied, rate):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute('''
        INSERT INTO frames
        (session_id, frame_no, empty, occupied, rate)
        VALUES (?, ?, ?, ?, ?)
    ''', (session_id, frame_no, empty, occupied, rate))

    conn.commit()
    conn.close()

# ---- Update Session Summary ----
def update_session_summary(session_id):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    # Calculate averages from frames
    c.execute('''
        SELECT
            COUNT(*),
            AVG(empty),
            AVG(occupied),
            AVG(rate),
            MAX(rate),
            MIN(rate)
        FROM frames
        WHERE session_id = ?
    ''', (session_id,))

    row = c.fetchone()

    total_frames = row[0]
    avg_empty    = round(row[1] or 0, 1)
    avg_occupied = round(row[2] or 0, 1)
    avg_rate     = round(row[3] or 0, 1)
    peak_rate    = round(row[4] or 0, 1)
    min_rate     = round(row[5] or 0, 1)

    c.execute('''
        UPDATE sessions
        SET total_frames = ?,
            avg_empty    = ?,
            avg_occupied = ?,
            avg_rate     = ?,
            peak_rate    = ?,
            min_rate     = ?
        WHERE id = ?
    ''', (total_frames, avg_empty, avg_occupied,
          avg_rate, peak_rate, min_rate, session_id))

    conn.commit()
    conn.close()

# ---- Get All Sessions ----
def get_all_sessions():
    conn     = sqlite3.connect(DB_PATH)
    c        = conn.cursor()
    c.execute('''
        SELECT id, video_name, date, time,
               total_frames, avg_rate,
               peak_rate, min_rate
        FROM sessions
        ORDER BY id DESC
    ''')
    rows     = c.fetchall()
    conn.close()

    sessions = []
    for row in rows:
        sessions.append({
            "id"           : row[0],
            "video_name"   : row[1],
            "date"         : row[2],
            "time"         : row[3],
            "total_frames" : row[4],
            "avg_rate"     : row[5],
            "peak_rate"    : row[6],
            "min_rate"     : row[7]
        })
    return sessions

# ---- Get One Session With Frames ----
def get_session_detail(session_id):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    # Get session info
    c.execute('''
        SELECT id, video_name, date, time,
               total_frames, avg_empty,
               avg_occupied, avg_rate,
               peak_rate, min_rate
        FROM sessions WHERE id = ?
    ''', (session_id,))
    row = c.fetchone()

    if not row:
        conn.close()
        return None

    session = {
        "id"           : row[0],
        "video_name"   : row[1],
        "date"         : row[2],
        "time"         : row[3],
        "total_frames" : row[4],
        "avg_empty"    : row[5],
        "avg_occupied" : row[6],
        "avg_rate"     : row[7],
        "peak_rate"    : row[8],
        "min_rate"     : row[9]
    }

    # Get all frames for this session
    c.execute('''
        SELECT frame_no, empty, occupied, rate
        FROM frames
        WHERE session_id = ?
        ORDER BY frame_no
    ''', (session_id,))
    frame_rows = c.fetchall()

    frames = []
    for fr in frame_rows:
        frames.append({
            "frame_no" : fr[0],
            "empty"    : fr[1],
            "occupied" : fr[2],
            "rate"     : fr[3]
        })

    session["frames"] = frames
    conn.close()
    return session

# ---- Initialize Database at startup ----
init_database()

# ================================================
# HELPER FUNCTIONS
# ================================================

# ---- Extract Single Slot From Frame ----
def extract_slot(image, contour_points):
    pts        = np.array(contour_points, dtype=np.float32)
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    padding    = 2
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = w + padding * 2
    h = h + padding * 2
    return image[y:y+h, x:x+w]

# ---- Process One Frame (CNN prediction) ----
def process_frame(image):
    output_image   = image.copy()
    empty_count    = 0
    occupied_count = 0
    slot_crops     = []
    valid_slots    = []

    for slot in slots:
        crop = extract_slot(image, slot['contour'])
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue
        resized    = cv2.resize(crop, (64, 64))
        normalized = resized / 255.0
        slot_crops.append(normalized)
        valid_slots.append(slot)

    if len(slot_crops) == 0:
        return output_image, 0, 0

    batch       = np.array(slot_crops)
    predictions = model.predict(batch, verbose=0)

    for i, slot in enumerate(valid_slots):
        confidence     = predictions[i][0]
        contour_points = slot['contour']

        if confidence < 0.5:
            color = (0, 255, 0)
            empty_count += 1
        else:
            color = (0, 0, 255)
            occupied_count += 1

        pts = np.array(contour_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(output_image, [pts],
                      isClosed=True, color=color, thickness=2)

    return output_image, empty_count, occupied_count

# ---- Background Thread: Process Video ----
def process_video_thread(filepath):
    global current_stats, video_frames, processing
    global current_session_id

    processing   = True
    video_frames = []

    # Get video filename
    video_name = os.path.basename(filepath)

    # Create new session in database
    current_session_id = create_session(video_name)
    print(f"Session created: ID {current_session_id}")

    cap = cv2.VideoCapture(filepath)

    if not cap.isOpened():
        print("Error: Could not open video!")
        processing = False
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {total_frames} frames...")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        output_frame, empty, occupied = process_frame(frame)

        rate = round((occupied / 100) * 100)
        current_stats = {
            "empty"    : int(empty),
            "occupied" : int(occupied),
            "total"    : 100,
            "rate"     : int(rate)
        }

        # Save frame to database
        save_frame(current_session_id,
                   frame_count, empty, occupied, rate)

        _, buffer = cv2.imencode('.jpg', output_frame)
        video_frames.append(buffer.tobytes())

        time.sleep(0.5)

        print(f"Frame {frame_count}/{total_frames} → "
              f"Empty: {empty} | Occupied: {occupied}")

    cap.release()

    # Update session summary
    update_session_summary(current_session_id)
    print(f"Session {current_session_id} saved to database!")

    processing = False
    print("Video processing complete!")

# ================================================
# ROUTES
# ================================================

# ---- Home Route ----
@app.route('/')
def home():
    return render_template('home.html')
# ---- Page Routes ----
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/about')
def about():
    return render_template('about.html')
# ---- Upload Video Route ----
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded!'})

    file = request.files['video']

    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected!'})

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    print(f"Video uploaded: {file_path}")

    return jsonify({
        'success'  : True,
        'message'  : 'Video uploaded successfully!',
        'filename' : file.filename,
        'filepath' : file_path
    })

# ---- Start Detection Route ----
@app.route('/detect', methods=['POST'])
def detect():
    data     = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({'success': False, 'message': 'No filename!'})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': 'File not found!'})

    thread        = threading.Thread(
        target = process_video_thread,
        args   = (filepath,)
    )
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'message': 'Detection started!'})

# ---- Stream Video Frames Route ----
def generate_frames():
    last_index = 0
    while True:
        if last_index < len(video_frames):
            frame = video_frames[last_index]
            last_index += 1
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + frame + b'\r\n')
        else:
            time.sleep(0.3)

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ---- Get Current Stats Route ----
@app.route('/stats')
def get_stats():
    return jsonify(current_stats)

# ---- Get Processing Status Route ----
@app.route('/status')
def get_status():
    return jsonify({
        'processing'   : processing,
        'total_frames' : len(video_frames)
    })

# ---- Get All Sessions Route ----
@app.route('/sessions')
def get_sessions():
    sessions = get_all_sessions()
    return jsonify(sessions)

# ---- Get One Session Detail Route ----
@app.route('/session/<int:session_id>')
def get_session(session_id):
    session = get_session_detail(session_id)
    if not session:
        return jsonify({'error': 'Session not found!'})
    return jsonify(session)

# ---- Delete Session Route ----
@app.route('/session/<int:session_id>/delete', methods=['POST'])
def delete_session(session_id):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute('DELETE FROM frames WHERE session_id = ?',
              (session_id,))
    c.execute('DELETE FROM sessions WHERE id = ?',
              (session_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True,
                    'message': 'Session deleted!'})

# ---- Run App ----
if __name__ == '__main__':
    app.run(debug=True)