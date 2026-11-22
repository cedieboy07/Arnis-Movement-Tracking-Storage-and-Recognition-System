import cv2
import numpy as np

# Upper & lower body skeleton connections for 33 MediaPipe landmarks
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms & Shoulders
    (15, 19), (16, 20),                                # Hands / Grip extension
    (11, 23), (12, 24), (23, 24),                      # Torso
    (23, 25), (25, 27), (24, 26), (26, 28),            # Legs
]

JOINT_LABELS = {
    "right_elbow": "Right Elbow",
    "left_elbow": "Left Elbow",
    "right_shoulder": "Right Shoulder",
    "left_shoulder": "Left Shoulder",
    "right_knee": "Right Knee",
    "left_knee": "Left Knee",
}

# Soft, friendly BGR palette
PALETTE = {
    "card_bg": (34, 30, 28),         # Warm charcoal
    "card_border": (95, 88, 82),     # Subtle warm gray border
    "mint": (165, 245, 195),         # Soft mint green (match / good form)
    "sky": (245, 225, 130),          # Soft pastel cyan (ghost target)
    "peach": (135, 175, 255),        # Warm coral/peach (gentle correction / countdown)
    "gold": (130, 215, 255),         # Warm rattan gold
    "cream": (242, 245, 248),        # Soft off-white text
    "muted": (175, 175, 180),        # Muted secondary text
}


class MartialArtsHUD:
    def __init__(self, angle_tolerance_deg=18.0):
        self.angle_tolerance_deg = angle_tolerance_deg

    @staticmethod
    def _fill_rounded_rect(img, pt1, pt2, color, radius=12):
        """Draws a filled rectangle with smooth rounded corners."""
        x1, y1 = pt1
        x2, y2 = pt2
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return
        r = min(radius, w // 2, h // 2)

        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in [
            (x1 + r, y1 + r),
            (x2 - r, y1 + r),
            (x1 + r, y2 - r),
            (x2 - r, y2 - r),
        ]:
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)

    @staticmethod
    def _stroke_rounded_rect(img, pt1, pt2, color, radius=12, thickness=1):
        """Draws a thin rounded border."""
        x1, y1 = pt1
        x2, y2 = pt2
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return
        r = min(radius, w // 2, h // 2)

        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)

        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)

    def _draw_card(self, frame, pt1, pt2, bg_color=None, border_color=None, alpha=0.72, radius=12):
        """Draws a translucent, rounded UI card with a clean border."""
        h, w, _ = frame.shape
        x1, y1 = max(0, pt1[0]), max(0, pt1[1])
        x2, y2 = min(w, pt2[0]), min(h, pt2[1])
        if x2 <= x1 or y2 <= y1:
            return

        bg = bg_color or PALETTE["card_bg"]
        roi = frame[y1:y2, x1:x2]
        overlay = roi.copy()
        self._fill_rounded_rect(overlay, (0, 0), (x2 - x1, y2 - y1), bg, radius=radius)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)

        if border_color is not None:
            self._stroke_rounded_rect(frame, (x1, y1), (x2, y2), border_color, radius=radius, thickness=1)

    @staticmethod
    def align_ghost_to_user(ghost_raw, live_raw):
        """
        Aligns saved reference landmarks to the practitioner's live hip center
        and torso size so the ghost pose overlays naturally anywhere in frame.
        """
        if ghost_raw is None or live_raw is None:
            return ghost_raw

        g_hip = (ghost_raw[23] + ghost_raw[24]) / 2.0
        g_shoulder = (ghost_raw[11] + ghost_raw[12]) / 2.0
        g_scale = np.linalg.norm(g_shoulder[:2] - g_hip[:2]) + 1e-6

        l_hip = (live_raw[23] + live_raw[24]) / 2.0
        l_shoulder = (live_raw[11] + live_raw[12]) / 2.0
        l_scale = np.linalg.norm(l_shoulder[:2] - l_hip[:2]) + 1e-6

        scale_ratio = l_scale / g_scale
        aligned = ghost_raw.copy()
        aligned[:, :2] = (ghost_raw[:, :2] - g_hip[:2]) * scale_ratio + l_hip[:2]
        return aligned

    def draw_weapons(self, frame, raw_landmarks, extractor, is_ghost=False):
        """Renders smooth virtual rattan Arnis bastons."""
        if raw_landmarks is None or extractor is None:
            return

        weapons = extractor.get_weapon_lines(raw_landmarks, frame.shape, blade_length_ratio=0.32)
        for hand, (p_start, p_end) in weapons.items():
            if is_ghost:
                cv2.line(frame, p_start, p_end, PALETTE["sky"], 4, cv2.LINE_AA)
                cv2.circle(frame, p_end, 5, PALETTE["sky"], -1, cv2.LINE_AA)
            else:
                shaft = PALETTE["gold"] if hand == "two_handed" else (145, 210, 245)
                cv2.line(frame, p_start, p_end, (45, 40, 38), 9, cv2.LINE_AA)
                cv2.line(frame, p_start, p_end, shaft, 5, cv2.LINE_AA)
                cv2.circle(frame, p_start, 5, PALETTE["peach"], -1, cv2.LINE_AA)
                cv2.circle(frame, p_end, 4, PALETTE["cream"], -1, cv2.LINE_AA)

    def draw_skeleton(
        self,
        frame,
        raw_landmarks,
        color=None,
        thickness=3,
        extractor=None,
        is_ghost=False,
    ):
        if raw_landmarks is None:
            return
        h, w, _ = frame.shape
        pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in raw_landmarks]
        line_color = PALETTE["sky"] if is_ghost else PALETTE["mint"]

        if extractor is not None:
            self.draw_weapons(frame, raw_landmarks, extractor, is_ghost=is_ghost)

        for i, j in POSE_CONNECTIONS:
            cv2.line(frame, pts[i], pts[j], line_color, thickness, cv2.LINE_AA)

        # Cute double-ring joints for the user, soft dots for the ghost
        for idx in range(11, 29):
            if is_ghost:
                cv2.circle(frame, pts[idx], 4, line_color, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pts[idx], 6, PALETTE["cream"], -1, cv2.LINE_AA)
                cv2.circle(frame, pts[idx], 4, line_color, -1, cv2.LINE_AA)

    def draw_ghost(self, frame, ghost_raw, label="", live_raw=None, extractor=None):
        """Draws the translucent pastel-sky Ghost pose and a neat target badge."""
        if ghost_raw is None:
            return

        display_ghost = self.align_ghost_to_user(ghost_raw, live_raw)
        overlay = frame.copy()
        self.draw_skeleton(
            overlay,
            display_ghost,
            thickness=3,
            extractor=extractor,
            is_ghost=True,
        )
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # Target Stance Pill Badge (top-left)
        badge_text = f"Target: {label}"
        (tw, _), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
        self._draw_card(
            frame,
            (18, 72),
            (46 + tw, 106),
            border_color=PALETTE["sky"],
            alpha=0.78,
            radius=16,
        )
        cv2.circle(frame, (31, 89), 4, PALETTE["sky"], -1, cv2.LINE_AA)
        cv2.putText(
            frame, badge_text, (41, 94),
            cv2.FONT_HERSHEY_SIMPLEX, 0.56, PALETTE["sky"], 2, cv2.LINE_AA
        )

    def _compute_form_hints(self, live_angles, ghost_angles):
        """Returns up to 3 simple, encouraging form tips when angles differ from target."""
        if not live_angles or not ghost_angles:
            return []

        hints = []
        comparisons = [
            ("right_elbow", "Extend right arm", "Bend right elbow"),
            ("left_elbow", "Extend left arm", "Bend left elbow"),
            ("right_shoulder", "Raise right arm", "Lower right arm"),
            ("left_shoulder", "Raise left arm", "Lower left arm"),
            ("right_knee", "Straighten right leg", "Bend right knee more"),
            ("left_knee", "Straighten left leg", "Bend left knee more"),
        ]
        for key, if_too_bent, if_too_straight in comparisons:
            if key not in live_angles or key not in ghost_angles:
                continue
            diff = live_angles[key] - ghost_angles[key]
            if abs(diff) > self.angle_tolerance_deg:
                hints.append((abs(diff), if_too_bent if diff < 0 else if_too_straight))

        hints.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in hints[:3]]

    def _draw_telemetry_panel(self, frame, live_raw, ghost_raw, extractor, has_ghost_badge):
        """Draws a tidy rounded card with live joint angles and form tips."""
        if live_raw is None or extractor is None:
            return

        live_angles = extractor.get_joint_angles_deg(live_raw)
        ghost_angles = extractor.get_joint_angles_deg(ghost_raw) if ghost_raw is not None else {}
        hints = self._compute_form_hints(live_angles, ghost_angles)

        top_y = 116 if has_ghost_badge else 72
        panel_w = 248
        panel_h = 166 + (len(hints) * 22 + 32 if hints else 0)

        self._draw_card(
            frame,
            (18, top_y),
            (18 + panel_w, top_y + panel_h),
            border_color=PALETTE["card_border"],
            alpha=0.72,
            radius=14,
        )

        cv2.putText(
            frame, "Joint Angles", (32, top_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.54, PALETTE["sky"], 2, cv2.LINE_AA
        )

        y = top_y + 48
        for key, label in JOINT_LABELS.items():
            val = live_angles.get(key, 0.0)
            row_color = PALETTE["cream"]
            tag = f"{val:.0f} deg"

            if key in ghost_angles:
                diff = val - ghost_angles[key]
                if abs(diff) <= self.angle_tolerance_deg:
                    row_color = PALETTE["mint"]
                    tag = f"{val:.0f} deg  OK"
                else:
                    row_color = PALETTE["peach"]
                    tag = f"{val:.0f} deg ({diff:+.0f})"

            cv2.putText(
                frame, label, (32, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, PALETTE["muted"], 1, cv2.LINE_AA
            )
            cv2.putText(
                frame, tag, (142, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, row_color, 1, cv2.LINE_AA
            )
            y += 20

        if hints:
            y += 6
            cv2.line(frame, (32, y - 6), (18 + panel_w - 14, y - 6), PALETTE["card_border"], 1)
            y += 14
            cv2.putText(
                frame, "Quick Tips", (32, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, PALETTE["peach"], 2, cv2.LINE_AA
            )
            y += 21
            for hint in hints:
                cv2.circle(frame, (36, y - 4), 3, PALETTE["peach"], -1, cv2.LINE_AA)
                cv2.putText(
                    frame, hint, (46, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, PALETTE["cream"], 1, cv2.LINE_AA
                )
                y += 21

    def draw_ui(
        self,
        frame,
        detected_name,
        score,
        history,
        status_msg,
        live_raw=None,
        ghost_raw=None,
        extractor=None,
        ghost_label=None,
    ):
        h, w, _ = frame.shape

        # 1. Floating Top Status Pill + Smooth Match Meter
        pill_x1, pill_y1 = 18, 14
        pill_x2, pill_y2 = w - 18, 60
        if detected_name:
            status_text = f"Matched: {detected_name}"
            accent = PALETTE["mint"]
        elif live_raw is None:
            status_text = "Step into camera view to begin"
            accent = PALETTE["peach"]
        else:
            status_text = "Scanning stance..."
            accent = PALETTE["sky"]

        self._draw_card(
            frame,
            (pill_x1, pill_y1),
            (pill_x2, pill_y2),
            border_color=accent if detected_name else PALETTE["card_border"],
            alpha=0.78,
            radius=16,
        )

        # Status dot + text
        cv2.circle(frame, (36, 37), 6, accent, -1, cv2.LINE_AA)
        cv2.putText(
            frame, status_text, (52, 43),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, PALETTE["cream"], 2, cv2.LINE_AA
        )

        # Right side of top pill: Score percentage + rounded progress bar
        pct_text = f"{score:.0f}%"
        cv2.putText(
            frame, pct_text, (w - 235, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.56, accent, 2, cv2.LINE_AA
        )
        bar_x1, bar_y1 = w - 180, 31
        bar_x2, bar_y2 = w - 34, 43
        self._fill_rounded_rect(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (65, 62, 60), radius=6)
        fill_w = int((bar_x2 - bar_x1) * max(0.0, min(100.0, float(score))) / 100.0)
        if fill_w > 6:
            self._fill_rounded_rect(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), accent, radius=6)

        # 2. Left Card: Joint Angles & Quick Form Tips
        self._draw_telemetry_panel(
            frame, live_raw, ghost_raw, extractor, has_ghost_badge=bool(ghost_label)
        )

        # 3. Center Countdown Alert Pill
        if status_msg:
            pretty_msg = status_msg.replace("HOLD STANCE:", "Hold your stance:")
            (tw, th), _ = cv2.getTextSize(pretty_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)
            cx, cy = w // 2, h // 2
            self._draw_card(
                frame,
                (cx - tw // 2 - 28, cy - th - 20),
                (cx + tw // 2 + 28, cy + 20),
                border_color=PALETTE["peach"],
                alpha=0.85,
                radius=18,
            )
            cv2.putText(
                frame, pretty_msg, (cx - tw // 2, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, PALETTE["peach"], 2, cv2.LINE_AA
            )

        # 4. Right Card: Recent Moves History
        hist_x1, hist_y1 = w - 275, 72
        hist_x2, hist_y2 = w - 18, 305
        self._draw_card(
            frame,
            (hist_x1, hist_y1),
            (hist_x2, hist_y2),
            border_color=PALETTE["card_border"],
            alpha=0.72,
            radius=14,
        )
        cv2.putText(
            frame, "Recent Moves", (hist_x1 + 16, hist_y1 + 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.54, PALETTE["sky"], 2, cv2.LINE_AA
        )

        recent_history = list(reversed(history))[:7]
        if not recent_history:
            cv2.putText(
                frame, "No stances matched yet", (hist_x1 + 16, hist_y1 + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, PALETTE["muted"], 1, cv2.LINE_AA
            )
        else:
            for idx, (m_name, m_score) in enumerate(recent_history):
                y_pos = hist_y1 + 56 + idx * 28
                is_latest = idx == 0
                dot_color = PALETTE["mint"] if is_latest else PALETTE["muted"]
                text_color = PALETTE["cream"] if is_latest else PALETTE["muted"]
                display_name = m_name if len(m_name) <= 16 else m_name[:14] + ".."

                cv2.circle(frame, (hist_x1 + 20, y_pos - 4), 3, dot_color, -1, cv2.LINE_AA)
                cv2.putText(
                    frame, display_name, (hist_x1 + 30, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, text_color, 1, cv2.LINE_AA
                )
                cv2.putText(
                    frame, f"{m_score:.0f}%", (hist_x2 - 48, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, dot_color, 1, cv2.LINE_AA
                )

        # 5. Bottom Floating Controls Pill
        controls = "R: Record Stance   |   1-9: Ghost Replay   |   C: Clear Ghost   |   Q: Quit"
        (cw, _), _ = cv2.getTextSize(controls, cv2.FONT_HERSHEY_SIMPLEX, 0.47, 1)
        bx1 = max(18, (w - cw) // 2 - 20)
        bx2 = min(w - 18, (w + cw) // 2 + 20)
        self._draw_card(
            frame,
            (bx1, h - 42),
            (bx2, h - 12),
            border_color=PALETTE["card_border"],
            alpha=0.75,
            radius=15,
        )
        cv2.putText(
            frame, controls, (bx1 + 20, h - 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.47, PALETTE["cream"], 1, cv2.LINE_AA
        )