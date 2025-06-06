import cv2
import numpy as np
import torch
import os
from scipy.signal import find_peaks, savgol_filter

def resize_with_padding(image, target_size=(640, 480)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    top = (target_size[1] - new_h) // 2
    bottom = target_size[1] - new_h - top
    left = (target_size[0] - new_w) // 2
    right = target_size[0] - new_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded, scale, left, top

def extract_keypoints(video_path, predictor):
    keypoints_per_frame = []
    detection_issues = False
    previous_keypoints = None
    previous_center = None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Could not open video file."

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            padded_frame, scale, pad_left, pad_top = resize_with_padding(frame_rgb)

            predictions, _, _ = predictor.numpy_image(padded_frame)

            if predictions:
                best_pred = None
                best_area = 0

                for pred in predictions:
                    kps = pred.data
                    if kps.shape[0] < 17:
                        continue
                    bbox = kps[:, :2]
                    area = np.ptp(bbox[:, 0]) * np.ptp(bbox[:, 1])
                    if area > best_area:
                        best_pred = pred
                        best_area = area

                if best_pred is None:
                    best_pred = max(predictions, key=lambda x: x.score)

                keypoints = best_pred.data
                keypoints[:, 0] = (keypoints[:, 0] - pad_left) / scale
                keypoints[:, 1] = (keypoints[:, 1] - pad_top) / scale

                keypoints_per_frame.append(keypoints)
                previous_keypoints = keypoints
            else:
                if previous_keypoints is not None:
                    adjusted_keypoints = previous_keypoints.copy()
                    adjusted_keypoints[:, 2] *= 0.8
                    keypoints_per_frame.append(adjusted_keypoints)
                    detection_issues = True
                else:
                    keypoints_per_frame.append(None)
                    detection_issues = True

            frame_count += 1

        cap.release()

        if not any(kps is not None for kps in keypoints_per_frame):
            return None, "No person detected in any frame."
        elif detection_issues:
            return keypoints_per_frame, "Person not consistently detected."
        else:
            return keypoints_per_frame, None

    except Exception as e:
        return None, f"Error during keypoint extraction: {e}"

def smooth_keypoints(keypoints_sequence, window_size=5):
    if len(keypoints_sequence) < window_size:
        return keypoints_sequence

    smoothed = []
    half_window = window_size // 2

    for i in range(len(keypoints_sequence)):
        if keypoints_sequence[i] is None:
            smoothed.append(None)
            continue

        start = max(0, i - half_window)
        end = min(len(keypoints_sequence), i + half_window + 1)
        window = [kp for kp in keypoints_sequence[start:end] if kp is not None]

        if not window:
            smoothed.append(keypoints_sequence[i])
            continue

        avg_kps = np.zeros_like(keypoints_sequence[i])
        total_weight = 0.0

        for kps in window:
            weight = np.mean(kps[:, 2])
            avg_kps += kps * weight
            total_weight += weight

        if total_weight > 0:
            avg_kps /= total_weight
            avg_kps[:, 2] = keypoints_sequence[i][:, 2]

        smoothed.append(avg_kps)

    return smoothed

def preprocess_keypoints_for_lstm(keypoints_sequence):
    processed_data = []
    num_expected_joints = 17

    for kps in keypoints_sequence:
        if kps is not None and kps.shape[0] == num_expected_joints:
            processed_data.append(kps[:, :2].flatten())
        else:
            processed_data.append(np.zeros(num_expected_joints * 2))

    return torch.tensor(np.array(processed_data), dtype=torch.float32).unsqueeze(0)

def calculate_angle(p1, p2, p3):
    p1 = np.array(p1[:2])
    p2 = np.array(p2[:2])
    p3 = np.array(p3[:2])

    v1 = p1 - p2
    v2 = p3 - p2

    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0

    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)

def get_exercise_angles(keypoints_sequence, exercise_type):
    angles_sequence = []

    def get_kp(kps_data, index):
        if kps_data is not None and len(kps_data) > index and len(kps_data[index]) >= 2:
            return kps_data[index][:2]
        return None

    for kps in keypoints_sequence:
        if kps is None or len(kps) < 17:
            angles_sequence.append(None)
            continue

        if exercise_type.lower() == 'pushup':
            shoulder = get_kp(kps, 6)
            elbow = get_kp(kps, 8)
            wrist = get_kp(kps, 10)
            hip = get_kp(kps, 12)
            ankle = get_kp(kps, 16)

            metrics = []
            if all(p is not None for p in [shoulder, elbow, wrist]):
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                metrics.append(elbow_angle)
            if all(p is not None for p in [shoulder, hip]):
                vertical_dist = (hip[1] - shoulder[1])
                metrics.append(vertical_dist)
            if all(p is not None for p in [shoulder, hip, ankle]):
                body_angle = calculate_angle(shoulder, hip, ankle)
                metrics.append(body_angle)

            if metrics:
                combined = 0.4*metrics[0] + 0.4*metrics[1] + 0.2*metrics[2] if len(metrics)==3 else \
                          0.5*metrics[0] + 0.5*metrics[1] if len(metrics)==2 else metrics[0]
                angles_sequence.append(combined)
            else:
                angles_sequence.append(None)
        elif exercise_type.lower() == 'pullup':
            shoulder = get_kp(kps, 6)
            elbow = get_kp(kps, 8)
            wrist = get_kp(kps, 10)
            if all(p is not None for p in [shoulder, elbow, wrist]):
                angles_sequence.append(calculate_angle(shoulder, elbow, wrist))
            else:
                angles_sequence.append(None)
        elif exercise_type.lower() == 'squat':
            hip = get_kp(kps, 12)
            knee = get_kp(kps, 14)
            ankle = get_kp(kps, 16)
            if all(p is not None for p in [hip, knee, ankle]):
                angles_sequence.append(calculate_angle(hip, knee, ankle))
            else:
                angles_sequence.append(None)
        elif exercise_type.lower() == 'legraises':
            shoulder = get_kp(kps, 6)
            hip = get_kp(kps, 12)
            knee = get_kp(kps, 14)
            if all(p is not None for p in [shoulder, hip, knee]):
                angles_sequence.append(calculate_angle(shoulder, hip, knee))
            else:
                angles_sequence.append(None)
        elif exercise_type.lower() == 'plank':
            angles_sequence.append(None)

    return angles_sequence

def count_reps_for_other_exercises(angles_sequence, exercise_type):
    if not angles_sequence or all(a is None for a in angles_sequence):
        return 0, []

    valid_data = [(i, a) for i, a in enumerate(angles_sequence) if a is not None]
    if not valid_data:
        return 0, []

    indices, values = zip(*valid_data)
    values = np.array(values)

    window_length = min(11, len(values))
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < 3:
        return 0, []

    smoothed = savgol_filter(values, window_length=window_length, polyorder=2)

    if exercise_type.lower() == 'pullup':
        down_thresh = 160
        up_thresh = 70
    elif exercise_type.lower() == 'squat':
        down_thresh = 90
        up_thresh = 160
    elif exercise_type.lower() == 'legraises':
        down_thresh = 170
        up_thresh = 100
    else:
        return 0, []

    state = 0
    rep_count = 0
    rep_frames = []

    for i, angle in enumerate(smoothed):
        original_idx = indices[i]

        if exercise_type.lower() in ['pullup', 'legraises']:
            if state == 0 and angle < up_thresh:
                state = 1
            elif state == 1 and angle > down_thresh:
                state = 2
            elif state == 2 and angle < up_thresh:
                rep_count += 1
                rep_frames.append(original_idx)
                state = 0
        else:
            if state == 0 and angle > up_thresh:
                state = 1
            elif state == 1 and angle < down_thresh:
                state = 2
            elif state == 2 and angle > up_thresh:
                rep_count += 1
                rep_frames.append(original_idx)
                state = 0

    return rep_count, rep_frames

def count_reps_from_angles(angles_sequence, exercise_type):
    if exercise_type.lower() != 'pushup':
        return count_reps_for_other_exercises(angles_sequence, exercise_type)

    # Filter out None values
    valid_data = [(i, angle_val) for i, angle_val in enumerate(angles_sequence) if angle_val is not None]
    if not valid_data or len(valid_data) < 10:
        return 0, []

    indices, values = zip(*valid_data)
    values = np.array(values)

    # Apply smoothing
    window_length = min(15, len(values))
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < 3:
        return 0, []

    smoothed_values = savgol_filter(values, window_length=window_length, polyorder=2)

    # Find peaks and valleys
    peaks, _ = find_peaks(smoothed_values, distance=10, prominence=5)
    valleys, _ = find_peaks(-smoothed_values, distance=10, prominence=5)

    # Combine and sort all critical points
    critical_points = sorted([(i, 'peak') for i in peaks] + [(i, 'valley') for i in valleys])
    
    # Count reps based on peak-valley-peak or valley-peak-valley sequences
    rep_count = 0
    rep_frames = []
    
    for i in range(1, len(critical_points)-1):
        prev_idx, prev_type = critical_points[i-1]
        curr_idx, curr_type = critical_points[i]
        next_idx, next_type = critical_points[i+1]
        
        # Look for peak-valley-peak pattern (push-up down and up)
        if prev_type == 'peak' and curr_type == 'valley' and next_type == 'peak':
            # Ensure the movement is significant enough
            if (smoothed_values[prev_idx] - smoothed_values[curr_idx] > 0.2 * (np.max(smoothed_values) - np.min(smoothed_values)) and
                smoothed_values[next_idx] - smoothed_values[curr_idx] > 0.2 * (np.max(smoothed_values) - np.min(smoothed_values))):
                rep_count += 1
                rep_frames.append(indices[next_idx])  # Mark the frame where rep is completed (at the top)

    return rep_count, rep_frames

def generate_keypoints_video(video_path, keypoints_sequence, output_path):
    print(f"Generating keypoints video from {video_path} to {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open input video")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("Error: Could not create video writer")
        cap.release()
        return False

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(keypoints_sequence) and keypoints_sequence[frame_idx] is not None:
            kps = keypoints_sequence[frame_idx]
            avg_confidence = np.mean([kp[2] for kp in kps if kp[2] > 0])

            for i, point in enumerate(kps):
                if point[2] > 0.1:
                    x, y = int(point[0]), int(point[1])
                    color = (0, min(255, int(255 * point[2])), 255)
                    cv2.circle(frame, (x, y), 5, color, -1)

            connections = [
                (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
                (5, 11), (6, 12), (11, 12),
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]
            for i, j in connections:
                if (i < len(kps) and j < len(kps) and
                    kps[i][2] > 0.1 and kps[j][2] > 0.1):
                    x1, y1 = int(kps[i][0]), int(kps[i][1])
                    x2, y2 = int(kps[j][0]), int(kps[j][1])
                    line_alpha = min(kps[i][2], kps[j][2])
                    color = (0, 0, int(255 * line_alpha))
                    cv2.line(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, f"Confidence: {avg_confidence:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    if not os.path.exists(output_path):
        print("Error: Output file was not created")
        return False
    if os.path.getsize(output_path) == 0:
        print("Error: Output file is empty")
        return False

    return True