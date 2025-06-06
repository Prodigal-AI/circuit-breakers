from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import re
import cv2
import openpifpaf
from mistake_detection import analyze_exercise
from rep_count import extract_keypoints, preprocess_keypoints_for_lstm, generate_keypoints_video, smooth_keypoints

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SNAPSHOT_FOLDER'] = 'static/snapshots'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB upload limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SNAPSHOT_FOLDER'], exist_ok=True)

try:
    predictor = openpifpaf.Predictor(checkpoint='resnet50')
except Exception as e:
    print(f"[ERROR] OpenPifPaf Predictor: {e}")
    exit("Please ensure the 'resnet50' checkpoint is available.")

def save_snapshot(video_path, frame_number, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, "MISTAKE DETECTED", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(output_path, frame)
    cap.release()
    return ret

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(request.url)

    video_file = request.files['video']
    exercise_type = request.form.get('exercise_type', '').strip()

    if video_file.filename == '':
        return redirect(request.url)

    filename = re.sub(r'[^\w.-]', '_', video_file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(video_path)

    return redirect(url_for('analyze', video_path=video_path, exercise_type=exercise_type))

@app.route('/analyze', methods=['GET'])
def analyze():
    video_path = request.args.get('video_path')
    exercise_type = request.args.get('exercise_type')

    if not os.path.exists(video_path):
        return render_template('error.html', message="Uploaded video not found.")

    keypoints, extraction_error = extract_keypoints(video_path, predictor)

    if extraction_error:
        return render_template('error.html', message=extraction_error)

    if not keypoints:
        return render_template('error.html', message="No keypoints detected in video.")
    
    # Apply smoothing to keypoints
    keypoints = smooth_keypoints(keypoints)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    safe_name = re.sub(r'[^\w]', '_', base_name)
    keypoints_video_filename = f"keypoints_{safe_name}.mp4"
    keypoints_video_path = os.path.join(app.config['SNAPSHOT_FOLDER'], keypoints_video_filename)

    if not generate_keypoints_video(video_path, keypoints, keypoints_video_path):
        return render_template('error.html', message="Failed to generate keypoints video.")

    rep_count, mistakes, mistake_frames = analyze_exercise(keypoints, exercise_type)

    snapshot_paths = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()

    for frame_num in mistake_frames:
        if 0 <= frame_num < total_frames:
            snapshot_filename = f"mistake_{exercise_type}_{frame_num}.jpg"
            snapshot_path = os.path.join(app.config['SNAPSHOT_FOLDER'], snapshot_filename)
            if save_snapshot(video_path, frame_num, snapshot_path):
                snapshot_paths.append(snapshot_filename)

    return render_template('results.html',
                           exercise_type=exercise_type,
                           rep_count=rep_count,
                           mistakes=mistakes,
                           snapshots=snapshot_paths,
                           mistake_frames=mistake_frames,
                           keypoints_video=keypoints_video_filename)

@app.route('/snapshots/<filename>')
def serve_snapshot(filename):
    return send_from_directory(app.config['SNAPSHOT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')