import cv2

# Upper & lower body skeleton connections for 33 MediaPipe landmarks
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms & Shoulders
    (15, 19), (16, 20),                                # Hands/Grip extension
    (11, 23), (12, 24), (23, 24),                      # Torso
    (23, 25), (25, 27), (24, 26), (26, 28),            # Legs
]


class MartialArtsHUD:
    def draw_skeleton(self, frame, raw_landmarks, color=(0, 255, 0), thickness=2):
        if raw_landmarks is None:
            return
        h, w, _ = frame.shape
        pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in raw_landmarks]

        for i, j in POSE_CONNECTIONS:
            cv2.line(frame, pts[i], pts[j], color, thickness)
        for idx in range(11, 29):
            cv2.circle(frame, pts[idx], 4, color, -1)

    def draw_ghost(self, frame, ghost_raw, label=""):
        """Draws a translucent cyan reference pose on screen."""
        if ghost_raw is None:
            return
        overlay = frame.copy()
        self.draw_skeleton(overlay, ghost_raw, color=(255, 255, 0), thickness=3)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        cv2.putText(
            frame, f"GHOST TARGET: {label}", (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )

    def draw_ui(self, frame, detected_name, score, history, status_msg):
        h, w, _ = frame.shape

        # Top banner: Live Match Status
        cv2.rectangle(frame, (0, 0), (w, 45), (20, 20, 20), -1)
        if detected_name:
            banner = f"MATCHED: {detected_name} ({score}%)"
            color = (0, 255, 120)
        else:
            banner = f"SCANNING... (Best: {score}%)"
            color = (200, 200, 200)
        cv2.putText(frame, banner, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Countdown / Recording status center alert
        if status_msg:
            cv2.putText(
                frame, status_msg, (w // 2 - 180, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 80, 255), 3
            )

        # Right panel: Previous Input Moves History
        cv2.rectangle(frame, (w - 310, 55), (w - 10, 270), (25, 25, 25), -1)
        cv2.putText(
            frame, "PREVIOUS MOVES:", (w - 295, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2
        )
        for idx, (m_name, m_score) in enumerate(reversed(history)):
            y_pos = 110 + idx * 26
            cv2.putText(
                frame, f"{idx + 1}. {m_name} ({m_score}%)", (w - 295, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
            )

        # Bottom controls legend
        controls = "[R] Record Stance   [1-9] Toggle Ghost Replay   [C] Clear Ghost   [Q] Quit"
        cv2.putText(frame, controls, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)