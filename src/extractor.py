import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseExtractor:
    def __init__(self, model_path="models/pose_landmarker_full.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

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
        Estimates the unit direction vector of a held stick/sword from the grip landmarks.
        MediaPipe landmarks: Left (15, 19, 17) | Right (16, 20, 18)
        """
        grip_center = (index_knuckle + pinky_knuckle) / 2.0
        axis = grip_center - wrist
        return axis / (np.linalg.norm(axis) + 1e-6)

    def process_frame(self, frame_bgr, timestamp_ms):
        """
        Data Contract:
          Returns (raw_landmarks_33x3, feature_vector_113) or (None, None).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return None, None

        landmarks = result.pose_landmarks[0]
        raw = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

        # 1. Hip-centered & torso-scaled body coordinates (99 features)
        hip_center = (raw[23] + raw[24]) / 2.0
        shoulder_center = (raw[11] + raw[12]) / 2.0
        torso_scale = np.linalg.norm(shoulder_center - hip_center) + 1e-6
        norm_coords = ((raw - hip_center) / torso_scale).flatten()

        # 2. Key martial arts joint angles, normalized to [0, 1] (8 features)
        angles = np.array([
            self._calc_angle(raw[11], raw[13], raw[15]),  # Left elbow
            self._calc_angle(raw[12], raw[14], raw[16]),  # Right elbow
            self._calc_angle(raw[23], raw[11], raw[13]),  # Left shoulder
            self._calc_angle(raw[24], raw[12], raw[14]),  # Right shoulder
            self._calc_angle(raw[11], raw[23], raw[25]),  # Left hip
            self._calc_angle(raw[12], raw[24], raw[26]),  # Right hip
            self._calc_angle(raw[23], raw[25], raw[27]),  # Left knee (stance depth)
            self._calc_angle(raw[24], raw[26], raw[28]),  # Right knee (stance depth)
        ], dtype=np.float32) / np.pi

        # 3. Virtual weapon direction unit vectors for both hands (6 features)
        left_blade_dir = self._weapon_axis(raw[15], raw[19], raw[17])
        right_blade_dir = self._weapon_axis(raw[16], raw[20], raw[18])

        # Concatenate into a single 113-D feature vector
        feature_vector = np.concatenate([norm_coords, angles, left_blade_dir, right_blade_dir])

        return raw, feature_vector