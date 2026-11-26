import re
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Lower-body landmark indices: hips (23, 24), knees (25, 26), ankles (27, 28), feet (29-32)
LOWER_BODY_INDICES = list(range(23, 33))
KNEE_ANKLE_INDICES = [25, 26, 27, 28]


class PoseExtractor:
    def __init__(self, model_path="models/pose_landmarker_full.task", smooth_alpha=0.65, vis_threshold=0.5):
        self.model_path = model_path
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.smooth_alpha = smooth_alpha
        self.vis_threshold = vis_threshold
        self._prev_raw = None

    @staticmethod
    def _calc_angle(a, b, c):
        """Computes the 3D angle (in radians, 0 to pi) at joint b formed by points a-b-c."""
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))

    @staticmethod
    def _weapon_axis(wrist, index_knuckle, pinky_knuckle):
        """
        Estimates the unit direction vector of a held stick/sword from grip landmarks.
        MediaPipe landmarks: Left (15, 19, 17) | Right (16, 20, 18)
        """
        grip_center = (index_knuckle + pinky_knuckle) / 2.0
        axis = grip_center - wrist
        return axis / (np.linalg.norm(axis) + 1e-6)

    def _smooth_landmarks(self, raw):
        if self._prev_raw is None:
            self._prev_raw = raw.copy()
        else:
            self._prev_raw = self.smooth_alpha * raw + (1.0 - self.smooth_alpha) * self._prev_raw
        return self._prev_raw

    def _build_feature_vector(self, raw, visibility):
        """Converts 33x3 landmarks and visibility scores into the normalized 115-D vector."""
        legs_visible = np.mean(visibility[KNEE_ANKLE_INDICES]) >= self.vis_threshold

        # 1. Hip-centered & torso-scaled body coordinates (99 features)
        hip_center = (raw[23] + raw[24]) / 2.0
        shoulder_center = (raw[11] + raw[12]) / 2.0
        torso_scale = np.linalg.norm(shoulder_center - hip_center) + 1e-6
        norm_3d = (raw - hip_center) / torso_scale

        if not legs_visible:
            norm_3d[LOWER_BODY_INDICES] = 0.0

        norm_coords = norm_3d.flatten()

        # 2. Key martial arts joint angles normalized to [0, 1] (8 features)
        angles = np.array([
            self._calc_angle(raw[11], raw[13], raw[15]),  # Left elbow
            self._calc_angle(raw[12], raw[14], raw[16]),  # Right elbow
            self._calc_angle(raw[23], raw[11], raw[13]),  # Left shoulder
            self._calc_angle(raw[24], raw[12], raw[14]),  # Right shoulder
            self._calc_angle(raw[11], raw[23], raw[25]) if legs_visible else 0.0,  # Left hip
            self._calc_angle(raw[12], raw[24], raw[26]) if legs_visible else 0.0,  # Right hip
            self._calc_angle(raw[23], raw[25], raw[27]) if legs_visible else 0.0,  # Left knee
            self._calc_angle(raw[24], raw[26], raw[28]) if legs_visible else 0.0,  # Right knee
        ], dtype=np.float32) / np.pi

        # 3. Virtual weapon direction unit vectors for both hands (6 features)
        left_blade_dir = self._weapon_axis(raw[15], raw[19], raw[17])
        right_blade_dir = self._weapon_axis(raw[16], raw[20], raw[18])

        # 4. Normalized wrist-to-wrist distance (1 feature)
        wrist_dist = np.array(
            [np.linalg.norm(raw[15] - raw[16]) / torso_scale], dtype=np.float32
        )

        # 5. Torso rotation / bladedness ratio (1 feature)
        shoulder_width_ratio = np.array(
            [abs(raw[11][0] - raw[12][0]) / torso_scale], dtype=np.float32
        )

        return np.concatenate(
            [norm_coords, angles, left_blade_dir, right_blade_dir, wrist_dist, shoulder_width_ratio]
        )

    def process_frame(self, frame_bgr, timestamp_ms):
        """
        Data Contract:
          Returns (smoothed_landmarks_33x3, feature_vector_115) or (None, None).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            self._prev_raw = None
            return None, None

        landmarks = result.pose_landmarks[0]
        raw = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        visibility = np.array([lm.visibility for lm in landmarks], dtype=np.float32)
        raw = self._smooth_landmarks(raw)

        return raw, self._build_feature_vector(raw, visibility)

    def extract_labeled_chapters_from_video(
        self,
        video_path,
        ocr_interval_sec=0.25,
        min_chapter_sec=1.0,
    ):
        """
        Detects on-screen text labels in a tutorial video to split it into chapters,
        extracts the peak held pose inside each chapter, and attaches the full chapter
        frame range for 2x loop demo playback.
        Returns a list of (label_name, raw_33x3, feature_vector_115, clip_info_dict) tuples.
        """
        from rapidocr_onnxruntime import RapidOCR

        abs_video_path = str(Path(video_path).resolve())
        cap = cv2.VideoCapture(abs_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step_frames = max(1, int(fps * ocr_interval_sec))
        ocr = RapidOCR()

        # Pass 1: Detect on-screen text changes to build (label, start_frame, end_frame) chapters
        chapters = []
        current_label = None
        chapter_start = 0

        for f_idx in range(0, total_frames, step_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame = cap.read()
            if not ok:
                break
            if f_idx % (step_frames * 4) == 0:
                cv2.waitKey(1)

            result, _ = ocr(frame)
            if not result:
                continue

            lines = [item[1].strip() for item in result if float(item[2]) > 0.70]
            raw_text = " ".join(lines)
            clean_label = re.sub(r"[^A-Za-z0-9 _-]", "", raw_text).strip()
            clean_label = re.sub(r"\s+", "_", clean_label)[:40]

            if len(clean_label) < 4 or any(
                skip in clean_label.upper()
                for skip in ("12_BASIC", "STATIONARY", "FOOTWORK", "SUBSCRIBE", "RIGHT_FOOT")
            ):
                continue

            if clean_label != current_label:
                if current_label is not None and (f_idx - chapter_start) >= int(fps * min_chapter_sec):
                    chapters.append((current_label, chapter_start, f_idx - 1))
                current_label = clean_label
                chapter_start = f_idx

        if current_label is not None and (total_frames - chapter_start) >= int(fps * min_chapter_sec):
            chapters.append((current_label, chapter_start, total_frames - 1))

        # Pass 2: Extract the most stable (lowest-motion) pose inside each chapter
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        extracted = []
        timestamp_ms = 0

        with vision.PoseLandmarker.create_from_options(options) as vid_landmarker:
            for idx, (label, start_f, end_f) in enumerate(chapters, start=1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                prev_raw = None
                best_motion = float("inf")
                best_pose = None

                search_start = start_f + int((end_f - start_f) * 0.25)
                search_end = start_f + int((end_f - start_f) * 0.85)

                for f_idx in range(start_f, end_f + 1):
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break
                    timestamp_ms += int(1000 / fps)

                    if f_idx < search_start or f_idx > search_end:
                        continue

                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    res = vid_landmarker.detect_for_video(mp_image, timestamp_ms)
                    if not res.pose_landmarks:
                        continue

                    lms = res.pose_landmarks[0]
                    raw = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
                    vis = np.array([lm.visibility for lm in lms], dtype=np.float32)

                    if prev_raw is not None:
                        motion = float(np.mean(np.linalg.norm(raw[11:21] - prev_raw[11:21], axis=1)))
                        if motion < best_motion:
                            best_motion = motion
                            best_pose = (raw.copy(), self._build_feature_vector(raw, vis))
                    prev_raw = raw

                if best_pose is not None:
                    clip_info = {
                        "video_path": abs_video_path,
                        "start_frame": start_f,
                        "end_frame": end_f,
                        "fps": float(fps),
                    }
                    extracted.append((f"{idx:02d}_{label}", best_pose[0], best_pose[1], clip_info))

        cap.release()
        return extracted

    def extract_stances_from_video(
        self,
        video_path,
        stillness_threshold=0.015,
        hold_seconds=0.5,
        cooldown_seconds=1.0,
        demo_seconds=1.8,
    ):
        """
        Fallback extractor: scans an unlabeled tutorial video for distinct held stances
        and stores the frame range leading up to each pose.
        Returns a list of (raw_33x3, feature_vector_115, clip_info_dict) tuples.
        """
        abs_video_path = str(Path(video_path).resolve())
        cap = cv2.VideoCapture(abs_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        min_hold_frames = max(3, int(fps * hold_seconds))
        cooldown_frames = int(fps * cooldown_seconds)
        clip_frames = max(10, int(fps * demo_seconds))

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        extracted_stances = []
        prev_raw = None
        still_count = 0
        cooldown = 0
        frame_idx = 0

        with vision.PoseLandmarker.create_from_options(options) as vid_landmarker:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                current_frame = frame_idx
                timestamp_ms = int((current_frame / fps) * 1000)
                frame_idx += 1
                if frame_idx % 30 == 0:
                    cv2.waitKey(1)

                if cooldown > 0:
                    cooldown -= 1
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = vid_landmarker.detect_for_video(mp_image, timestamp_ms)

                if not result.pose_landmarks:
                    prev_raw = None
                    still_count = 0
                    continue

                landmarks = result.pose_landmarks[0]
                raw = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
                visibility = np.array([lm.visibility for lm in landmarks], dtype=np.float32)

                if prev_raw is not None:
                    motion = np.mean(np.linalg.norm(raw[11:21] - prev_raw[11:21], axis=1))
                    if motion < stillness_threshold:
                        still_count += 1
                    else:
                        still_count = 0

                    if still_count >= min_hold_frames:
                        vec = self._build_feature_vector(raw, visibility)
                        clip_info = {
                            "video_path": abs_video_path,
                            "start_frame": max(0, current_frame - clip_frames),
                            "end_frame": current_frame,
                            "fps": float(fps),
                        }
                        extracted_stances.append((raw.copy(), vec, clip_info))
                        still_count = 0
                        cooldown = cooldown_frames

                prev_raw = raw

        cap.release()
        return extracted_stances

    def get_joint_angles_deg(self, raw):
        """Helper for HUD/Form Feedback: returns named joint angles in degrees."""
        if raw is None:
            return {}
        return {
            "left_elbow": round(np.degrees(self._calc_angle(raw[11], raw[13], raw[15])), 1),
            "right_elbow": round(np.degrees(self._calc_angle(raw[12], raw[14], raw[16])), 1),
            "left_shoulder": round(np.degrees(self._calc_angle(raw[23], raw[11], raw[13])), 1),
            "right_shoulder": round(np.degrees(self._calc_angle(raw[24], raw[12], raw[14])), 1),
            "left_knee": round(np.degrees(self._calc_angle(raw[23], raw[25], raw[27])), 1),
            "right_knee": round(np.degrees(self._calc_angle(raw[24], raw[26], raw[28])), 1),
        }

    def get_weapon_lines(self, raw, frame_shape, blade_length_ratio=0.35):
        """
        Helper for HUD: returns pixel endpoints ((x1, y1), (x2, y2)) for virtual blades/sticks.
        Automatically merges into a single two-handed blade if wrists are close together.
        """
        if raw is None:
            return {}

        h, w, _ = frame_shape
        torso_scale = np.linalg.norm(((raw[11] + raw[12]) / 2.0) - ((raw[23] + raw[24]) / 2.0)) + 1e-6
        wrist_gap = np.linalg.norm(raw[15] - raw[16]) / torso_scale

        def to_px(pt):
            return (int(pt[0] * w), int(pt[1] * h))

        if wrist_gap < 0.35:
            mid_wrist = (raw[15] + raw[16]) / 2.0
            mid_dir = (
                self._weapon_axis(raw[15], raw[19], raw[17])
                + self._weapon_axis(raw[16], raw[20], raw[18])
            )
            mid_dir /= np.linalg.norm(mid_dir) + 1e-6
            tip = mid_wrist + mid_dir * blade_length_ratio
            return {"two_handed": (to_px(mid_wrist), to_px(tip))}

        l_dir = self._weapon_axis(raw[15], raw[19], raw[17])
        r_dir = self._weapon_axis(raw[16], raw[20], raw[18])
        return {
            "left": (to_px(raw[15]), to_px(raw[15] + l_dir * blade_length_ratio)),
            "right": (to_px(raw[16]), to_px(raw[16] + r_dir * blade_length_ratio)),
        }