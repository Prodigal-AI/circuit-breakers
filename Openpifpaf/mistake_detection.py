import numpy as np
import torch
import torch.nn as nn
from rep_count import preprocess_keypoints_for_lstm, calculate_angle, get_exercise_angles, count_reps_from_angles

class ExerciseLSTM(nn.Module):
    def __init__(self): # Corrected: __init__
        super(ExerciseLSTM, self).__init__() # Corrected: __init__
        self.lstm = nn.LSTM(input_size=34, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 2)  # Output layer expects 2 outputs

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 128).to(x.device)
        c0 = torch.zeros(1, x.size(0), 128).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        logits = self.fc(out[:, -1, :])  # shape: (batch_size, 2)
        probs = torch.softmax(logits, dim=1)  # Convert to probabilities
        return {'has_mistakes': probs[:, 1]}  # Return probability of class 1 = mistake


def load_model(path):
    model = ExerciseLSTM()
    try:
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    return model

# Load models
try:
    pushup_model = load_model('D:/Exercise Monitoring with pifpaf/models/push-up_lstm.pth')
    pullup_model = load_model('D:/Exercise Monitoring with pifpaf/models/pull-up_lstm.pth')
    squat_model = load_model('D:/Exercise Monitoring with pifpaf/models/squat_lstm.pth')
    legraises_model = load_model('D:/Exercise Monitoring with pifpaf/models/leg-raises_lstm.pth')
    plank_model = load_model('D:\Exercise Monitoring with pifpaf\models\plank_lstm.pth')
except Exception as e:
    print(f"Model loading failed: {e}")
    pushup_model = pullup_model = squat_model = legraises_model = plank_model = None

def analyze_exercise(keypoints_sequence, exercise_type, fps=30): # Added fps parameter for time-based sampling
    mistakes = set()
    mistake_frames = set()

    # Get rep count first
    angles_for_reps = get_exercise_angles(keypoints_sequence, exercise_type)
    rep_count, rep_frames = count_reps_from_angles(angles_for_reps, exercise_type)

    # Load corresponding model
    model = {
        'pushup': pushup_model,
        'pullup': pullup_model,
        'squat': squat_model,
        'legraises': legraises_model,
        'plank': plank_model
    }.get(exercise_type.lower(), None)

    if model is not None:
        processed_sequence = preprocess_keypoints_for_lstm(keypoints_sequence)
        with torch.no_grad():
            output = model(processed_sequence)
            if output['has_mistakes'].item() > 0.6:
                mistakes.add("General form deviation detected (AI-driven)")
                # For general AI-driven mistake, we can still sample to avoid too many frames
                # For simplicity, let's sample the start, middle, and end frames if a general mistake is detected
                total_frames = len(keypoints_sequence)
                if total_frames > 0:
                    mistake_frames.add(0) # Start
                    if total_frames > 1:
                        mistake_frames.add(total_frames // 2) # Middle
                    if total_frames > 2:
                        mistake_frames.add(total_frames - 1) # End


    def get_kp(kps_data, index):
        if kps_data is not None and len(kps_data) > index and len(kps_data[index]) >= 2:
            return kps_data[index][:2]
        return None

    # Helper function to register and sample mistake frames
    # consecutive_threshold: How many consecutive frames the mistake must occur to be registered
    # sampling_interval_frames: How many frames to skip between sampled mistake frames (e.g., 3*fps for every 3 seconds)
    def _register_mistake_with_sampling(label, frames_where_mistake_occurs, consecutive_threshold=5, sampling_interval_frames=None):
        if sampling_interval_frames is None:
            sampling_interval_frames = fps * 3 # Default to 3 seconds if fps is provided

        consecutive_count = 0
        prev_frame = -2
        last_sampled_frame = -sampling_interval_frames -1 # Ensure first eligible frame is sampled

        sorted_frames = sorted(list(frames_where_mistake_occurs))
        
        for frame in sorted_frames:
            if frame == prev_frame + 1:
                consecutive_count += 1
            else:
                consecutive_count = 1
            prev_frame = frame

            if consecutive_count >= consecutive_threshold:
                # Check if enough time has passed since the last sampled mistake frame for this label
                if frame >= last_sampled_frame + sampling_interval_frames:
                    mistakes.add(label)
                    mistake_frames.add(frame)
                    last_sampled_frame = frame

    # --- Exercise Specific Mistake Detection ---

    if exercise_type.lower() == 'plank':
        hips_sagging_frames = set()
        hips_raised_frames = set()

        for i, kps in enumerate(keypoints_sequence):
            if kps is None or len(kps) < 17:
                continue

            shoulder_r = get_kp(kps, 6)
            hip_r = get_kp(kps, 12)
            ankle_r = get_kp(kps, 16)

            if all(kp is not None for kp in [shoulder_r, hip_r, ankle_r]):
                body_angle = float(calculate_angle(shoulder_r, hip_r, ankle_r))
                
                if body_angle < 140: # This angle indicates a bend in the body, not a straight line
                    # Calculate a 'straight line' reference for hip position
                    # This uses the midpoint of shoulder and ankle Y-coordinates as a reference
                    # A more robust check might involve comparing hip-y to shoulder-y directly
                    line_y = float((shoulder_r[1] + ankle_r[1]) / 2)
                    hip_y = float(hip_r[1])

                    # Thresholds (e.g., 30 pixels deviation) should be tuned for your video resolution
                    if hip_y > line_y + 30: # Hip is significantly below the line (sagging)
                        hips_sagging_frames.add(i)
                    elif hip_y < line_y - 30: # Hip is significantly above the line (raised too high)
                        hips_raised_frames.add(i)
        
        _register_mistake_with_sampling("Hips sagging (Plank)", hips_sagging_frames, consecutive_threshold=fps//2) # ~0.5 sec
        _register_mistake_with_sampling("Hips raised too high (Plank)", hips_raised_frames, consecutive_threshold=fps//2)

    elif exercise_type.lower() == 'pushup':
        hip_sag_frames = set()
        elbow_flare_frames = set()
        speed_violation_frames = set()
        
        # For speed analysis, we need the original indices for consistent frame mapping
        valid_keypoints_for_speed = [(i, kps) for i, kps in enumerate(keypoints_sequence) if kps is not None and len(kps) >= 17]
        
        prev_shoulder_y = None
        frame_diffs = []
        original_indices_for_speed = []

        for i, kps in valid_keypoints_for_speed: # Iterate only over valid keypoints for this loop
            shoulder = get_kp(kps, 6)
            elbow = get_kp(kps, 8)
            wrist = get_kp(kps, 10)
            hip = get_kp(kps, 12)

            if all(kp is not None for kp in [shoulder, hip]):
                # Hip sag: hip Y-coordinate significantly below shoulder Y-coordinate
                # A good push-up maintains a relatively straight line.
                if hip[1] > shoulder[1] + 50: # Example: hip is 50 pixels below shoulder
                    hip_sag_frames.add(i)

            if all(kp is not None for kp in [shoulder, elbow, wrist]):
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                # Elbow flaring: angle between shoulder, elbow, wrist is too wide
                # Usually indicates elbows pointing outwards too much
                if elbow_angle > 110: # Example threshold for flaring (can be refined based on data)
                    elbow_flare_frames.add(i)

            # Speed violation (analyze movement between frames)
            if shoulder is not None and prev_shoulder_y is not None:
                frame_diffs.append(abs(shoulder[1] - prev_shoulder_y))
                original_indices_for_speed.append(i) # Store the index of the current frame
            prev_shoulder_y = shoulder[1] if shoulder is not None else None

        _register_mistake_with_sampling("Hips sagging (Push-up)", hip_sag_frames, consecutive_threshold=fps//2)
        _register_mistake_with_sampling("Elbows flaring out (Push-up)", elbow_flare_frames, consecutive_threshold=fps//2)

        # Speed violation: Analyze the collected frame differences
        if frame_diffs and original_indices_for_speed:
            avg_speed_window = int(fps * 0.5) # Look at average speed over 0.5 seconds
            if avg_speed_window < 1: avg_speed_window = 1 # Ensure at least 1 frame

            for j in range(len(frame_diffs) - avg_speed_window + 1):
                window_avg = np.mean(frame_diffs[j : j + avg_speed_window])
                # This threshold is highly dependent on resolution and how fast is 'too fast'
                # 10 pixels/frame at 30 FPS means 300 pixels/second movement, which can be fast
                if window_avg > 10: # Example: average movement of 10 pixels per frame is too fast
                    # Flag the last frame in the current window as a speed violation
                    speed_violation_frames.add(original_indices_for_speed[j + avg_speed_window - 1])
        
        _register_mistake_with_sampling("Moving too fast (Push-up)", speed_violation_frames, consecutive_threshold=fps*1) # Flag if fast for 1 second continuously

    elif exercise_type.lower() == 'pullup':
        kipping_frames = set()

        for i in range(1, len(keypoints_sequence)):
            prev_kps = keypoints_sequence[i - 1]
            curr_kps = keypoints_sequence[i]
            if curr_kps is None or prev_kps is None:
                continue

            prev_hip_r = get_kp(prev_kps, 12)
            hip_r = get_kp(curr_kps, 12)

            if all(kp is not None for kp in [prev_hip_r, hip_r]):
                # Kipping: significant vertical hip movement during the pull-up
                # Threshold of 20 pixels vertical movement indicates excessive swing
                if abs(float(hip_r[1]) - float(prev_hip_r[1])) > 20:
                    kipping_frames.add(i)

        _register_mistake_with_sampling("Kipping detected (Pull-up)", kipping_frames, consecutive_threshold=fps//2)

    elif exercise_type.lower() == 'squat':
        back_rounding_frames = set()
        knees_caving_frames = set()

        for i, kps in enumerate(keypoints_sequence):
            if kps is None or len(kps) < 17:
                continue

            hip_r = get_kp(kps, 12)
            knee_r = get_kp(kps, 14)
            ankle_r = get_kp(kps, 16)
            shoulder_r = get_kp(kps, 6)

            if all(kp is not None for kp in [knee_r, ankle_r]):
                # Knees caving/bowing: horizontal distance between knee and ankle
                # If knee is too far laterally from ankle (in X-direction)
                if abs(float(knee_r[0]) - float(ankle_r[0])) > 25: # Example threshold for x-offset
                    knees_caving_frames.add(i)

            if all(kp is not None for kp in [shoulder_r, hip_r, knee_r]):
                back_angle = float(calculate_angle(shoulder_r, hip_r, knee_r))
                # Rounded back: angle between shoulder, hip, knee is too small (closer to 90 than 180)
                if back_angle < 150: # Angle indicating rounded back (should be closer to 180 for straight)
                    back_rounding_frames.add(i)

        _register_mistake_with_sampling("Knees caving/bowing out (Squat)", knees_caving_frames, consecutive_threshold=fps//2)
        _register_mistake_with_sampling("Rounded back (Squat)", back_rounding_frames, consecutive_threshold=fps//2)

    elif exercise_type.lower() == 'legraises':
        swinging_frames = set()
        arching_back_frames = set()

        for i in range(1, len(keypoints_sequence)):
            prev_kps = keypoints_sequence[i - 1]
            curr_kps = keypoints_sequence[i]
            if curr_kps is None or prev_kps is None:
                continue

            ankle_r = get_kp(curr_kps, 16)
            prev_ankle_r = get_kp(prev_kps, 16)
            shoulder_r = get_kp(curr_kps, 6)
            hip_r = get_kp(curr_kps, 12)

            if all(kp is not None for kp in [ankle_r, prev_ankle_r]):
                movement = float(np.linalg.norm(np.array(ankle_r) - np.array(prev_ankle_r)))
                # Swinging: large movement of the ankle between frames
                if movement > 40: # Example threshold for significant ankle movement indicating swing
                    swinging_frames.add(i)

            if all(kp is not None for kp in [shoulder_r, hip_r]):
                # Lower back arching: hip moves significantly below shoulder vertically, indicating an arch
                if float(hip_r[1]) - float(shoulder_r[1]) > 25: # Example threshold
                    arching_back_frames.add(i)

        _register_mistake_with_sampling("Swinging detected (Leg Raises)", swinging_frames, consecutive_threshold=fps//2)
        _register_mistake_with_sampling("Lower back arching (Leg Raises)", arching_back_frames, consecutive_threshold=fps//2)


    return rep_count, list(mistakes), sorted(list(mistake_frames))