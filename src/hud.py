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
        mode="TRAIN",
    ):
        h, w, _ = frame.shape

        # 1. Floating Top Status Pill + Smooth Match Meter
        pill_x1, pill_y1 = 18, 14
        pill_x2, pill_y2 = w - 18, 60
        if mode == "TEACH":
            status_text = "Teaching Mode  |  Record or Import Stances"
            accent = PALETTE["gold"]
        elif detected_name:
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
            border_color=accent if (detected_name or mode == "TEACH") else PALETTE["card_border"],
            alpha=0.78,
            radius=16,
        )

        # Status dot + text
        cv2.circle(frame, (36, 37), 6, accent, -1, cv2.LINE_AA)
        cv2.putText(
            frame, status_text, (52, 43),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, PALETTE["cream"], 2, cv2.LINE_AA
        )

        # Right side of top pill: Score percentage + rounded progress bar (Training mode)
        if mode != "TEACH":
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

        # 3. Center Countdown / Status Alert Pill
        if status_msg:
            pretty_msg = status_msg.replace("HOLD STANCE:", "Hold your stance:")
            (tw, th), _ = cv2.getTextSize(pretty_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.90, 2)
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
                cv2.FONT_HERSHEY_SIMPLEX, 0.90, PALETTE["peach"], 2, cv2.LINE_AA
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
            for idx, entry in enumerate(recent_history):
                if isinstance(entry, dict):
                    m_name, m_score = entry["name"], entry["score"]
                else:
                    m_name, m_score = entry
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
        if mode == "TEACH":
            controls = "R: Record Stance   |   I: Import Video   |   M: Menu   |   Q: Quit"
        else:
            controls = "L: Folder   |   D: Demo   |   N/P: Next/Prev   |   S: Seq   |   T: Phone   |   M: Menu"

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

    # ---------------- OVERLAY HELPERS (MENU, INPUT, DEMO, DRILLS) ----------------

    def draw_menu(self, frame, move_count):
        """Renders the startup mode selection card using the groupmate's palette."""
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (18, 17, 16), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        card_w, card_h = 560, 320
        x1, y1 = (w - card_w) // 2, (h - card_h) // 2
        x2, y2 = x1 + card_w, y1 + card_h

        self._draw_card(
            frame, (x1, y1), (x2, y2),
            border_color=PALETTE["sky"], alpha=0.88, radius=18
        )
        cv2.putText(
            frame, "MARTIAL ARTS STANCE GUIDE", (x1 + 36, y1 + 52),
            cv2.FONT_HERSHEY_SIMPLEX, 0.80, PALETTE["sky"], 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Saved Stances in Library: {move_count}", (x1 + 36, y1 + 84),
            cv2.FONT_HERSHEY_SIMPLEX, 0.54, PALETTE["muted"], 1, cv2.LINE_AA
        )

        self._draw_card(
            frame, (x1 + 36, y1 + 112), (x2 - 36, y1 + 178),
            border_color=PALETTE["gold"], alpha=0.85, radius=12
        )
        cv2.putText(
            frame, "[1]  TEACHING MODE", (x1 + 54, y1 + 140),
            cv2.FONT_HERSHEY_SIMPLEX, 0.64, PALETTE["gold"], 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, "Record live stances or import tutorial videos", (x1 + 54, y1 + 164),
            cv2.FONT_HERSHEY_SIMPLEX, 0.47, PALETTE["cream"], 1, cv2.LINE_AA
        )

        self._draw_card(
            frame, (x1 + 36, y1 + 194), (x2 - 36, y1 + 260),
            border_color=PALETTE["mint"], alpha=0.85, radius=12
        )
        cv2.putText(
            frame, "[2]  TRAINING MODE", (x1 + 54, y1 + 222),
            cv2.FONT_HERSHEY_SIMPLEX, 0.64, PALETTE["mint"], 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, "Watch 2x video demos, practice Ghost & Phone Drills", (x1 + 54, y1 + 246),
            cv2.FONT_HERSHEY_SIMPLEX, 0.47, PALETTE["cream"], 1, cv2.LINE_AA
        )

        cv2.putText(
            frame, "Press 1 or 2 to start   |   Press Q to quit", (x1 + 110, y2 - 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.49, PALETTE["muted"], 1, cv2.LINE_AA
        )

    def prompt_text_in_window(self, base_frame, title, default_text=""):
        """Collects text input inside the OpenCV window without blocking Windows UI events."""
        text = default_text
        while True:
            canvas = base_frame.copy()
            h, w, _ = canvas.shape

            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (15, 14, 13), -1)
            cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)

            box_w, box_h = min(720, w - 80), 150
            x1, y1 = (w - box_w) // 2, (h - box_h) // 2
            x2, y2 = x1 + box_w, y1 + box_h
            self._draw_card(
                canvas, (x1, y1), (x2, y2),
                border_color=PALETTE["sky"], alpha=0.92, radius=16
            )

            cv2.putText(
                canvas, title, (x1 + 22, y1 + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, PALETTE["sky"], 2, cv2.LINE_AA
            )
            cv2.putText(
                canvas, f"> {text}_", (x1 + 22, y1 + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.64, PALETTE["cream"], 2, cv2.LINE_AA
            )
            cv2.putText(
                canvas, "Press ENTER to confirm   |   ESC to cancel", (x1 + 22, y1 + 126),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, PALETTE["muted"], 1, cv2.LINE_AA
            )

            cv2.imshow("Martial Arts Guide", canvas)
            key = cv2.waitKey(30) & 0xFF

            if key in (13, 10):
                return text.strip()
            elif key == 27:
                return ""
            elif key in (8, 127):
                text = text[:-1]
            elif 32 <= key <= 126:
                text += chr(key)

    def draw_demo_clip(self, frame, video_frame, label=""):
        """Renders the imported tutorial video clip card before live accuracy tracking begins."""
        if video_frame is None:
            return

        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (18, 16, 15), -1)
        cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

        vh, vw, _ = video_frame.shape
        target_h = min(430, h - 180)
        scale = target_h / float(vh)
        target_w = min(w - 120, int(vw * scale))
        resized = cv2.resize(video_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

        x1 = (w - target_w) // 2
        y1 = (h - target_h) // 2 + 15
        x2 = x1 + target_w
        y2 = y1 + target_h

        self._draw_card(
            frame, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8),
            border_color=PALETTE["sky"], alpha=0.92, radius=16
        )
        frame[y1:y2, x1:x2] = resized

        banner = f"WATCH DEMO: {label}   (ESC to skip | D anytime to replay)"
        (tw, _), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
        cx = w // 2
        self._draw_card(
            frame, (cx - tw // 2 - 22, 54), (cx + tw // 2 + 22, 96),
            border_color=PALETTE["sky"], alpha=0.90, radius=16
        )
        cv2.putText(
            frame, banner, (cx - tw // 2, 82),
            cv2.FONT_HERSHEY_SIMPLEX, 0.56, PALETTE["sky"], 2, cv2.LINE_AA
        )

    def draw_drill_banner(self, frame, title, items, current_idx, hold_progress=0.0):
        """Renders a top-center sequence card for Guided Sequence or Phone-Number Recall."""
        h, w, _ = frame.shape
        card_w = min(560, max(440, w - 580))
        x1 = (w - card_w) // 2
        x2 = x1 + card_w
        y1, y2 = 72, 152

        self._draw_card(
            frame, (x1, y1), (x2, y2),
            border_color=PALETTE["sky"], alpha=0.84, radius=14
        )
        cv2.putText(
            frame, title, (x1 + 16, y1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.47, PALETTE["sky"], 2, cv2.LINE_AA
        )

        cursor_x = x1 + 16
        step_y = y1 + 52
        for idx, item in enumerate(items):
            if idx < current_idx:
                color = PALETTE["mint"]
                prefix = "[OK] "
            elif idx == current_idx:
                color = PALETTE["peach"]
                prefix = "> "
            else:
                color = PALETTE["muted"]
                prefix = ""

            token = f"{prefix}{item}"
            (tw, _), _ = cv2.getTextSize(token, cv2.FONT_HERSHEY_SIMPLEX, 0.47, 1)
            if cursor_x + tw > x2 - 16:
                break
            cv2.putText(
                frame, token, (cursor_x, step_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 2 if idx == current_idx else 1, cv2.LINE_AA
            )
            cursor_x += tw + 16

        bar_x1, bar_y1 = x1 + 16, y2 - 14
        bar_x2, bar_y2 = x2 - 16, y2 - 6
        self._fill_rounded_rect(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (65, 62, 60), radius=4)
        fill_w = int((bar_x2 - bar_x1) * max(0.0, min(1.0, hold_progress)))
        if fill_w > 4:
            self._fill_rounded_rect(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), PALETTE["mint"], radius=4)

    def select_compilation_in_window(self, base_frame, compilations_dict, active_comp=None):
        """
        Lets the user pick which folder/compilation of stances to practice.
        Returns the chosen compilation folder name, 'ALL', or None if cancelled.
        """
        comp_names = list(compilations_dict.keys())
        if not comp_names:
            return None

        selected_idx = 0
        if active_comp in comp_names:
            selected_idx = comp_names.index(active_comp)

        while True:
            canvas = base_frame.copy()
            h, w, _ = canvas.shape

            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (15, 14, 13), -1)
            cv2.addWeighted(overlay, 0.74, canvas, 0.26, 0, canvas)

            row_h = 46
            box_w = 580
            box_h = min(h - 80, 175 + (len(comp_names) + 1) * row_h)
            x1, y1 = (w - box_w) // 2, (h - box_h) // 2
            x2, y2 = x1 + box_w, y1 + box_h

            self._draw_card(
                canvas, (x1, y1), (x2, y2),
                border_color=PALETTE["sky"], alpha=0.92, radius=18
            )
            cv2.putText(
                canvas, "SELECT STANCE COMPILATION", (x1 + 28, y1 + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, PALETTE["sky"], 2, cv2.LINE_AA
            )
            cv2.putText(
                canvas, "Choose a folder from moves_db/ to practice:", (x1 + 28, y1 + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, PALETTE["muted"], 1, cv2.LINE_AA
            )

            options = [(name, len(compilations_dict[name])) for name in comp_names]
            total_count = sum(count for _, count in options)
            options.append(("ALL", total_count))

            y = y1 + 96
            for idx, (name, count) in enumerate(options):
                if y + row_h > y2 - 42:
                    break
                is_sel = idx == selected_idx
                border = PALETTE["mint"] if is_sel else PALETTE["card_border"]
                label_color = PALETTE["mint"] if is_sel else PALETTE["cream"]
                display = "All Compilations Combined" if name == "ALL" else name.replace("_", " ")
                num_tag = "0" if name == "ALL" else str(idx + 1)

                self._draw_card(
                    canvas, (x1 + 26, y), (x2 - 26, y + row_h - 6),
                    border_color=border, alpha=0.88 if is_sel else 0.65, radius=10
                )
                cv2.putText(
                    canvas, f"[{num_tag}]  {display}", (x1 + 42, y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, label_color, 2 if is_sel else 1, cv2.LINE_AA
                )
                count_txt = f"{count} stances"
                cv2.putText(
                    canvas, count_txt, (x2 - 135, y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, PALETTE["gold"], 1, cv2.LINE_AA
                )
                y += row_h

            cv2.putText(
                canvas, "W/S or 1-9: Select   |   ENTER: Start   |   ESC: Back",
                (x1 + 28, y2 - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, PALETTE["muted"], 1, cv2.LINE_AA
            )

            cv2.imshow("Martial Arts Guide", canvas)
            key = cv2.waitKey(30) & 0xFF

            if key in (13, 10):  # Enter
                return options[selected_idx][0]
            elif key == 27:      # Esc
                return None
            elif key in (ord("w"), ord("W"), 82):
                selected_idx = (selected_idx - 1) % len(options)
            elif key in (ord("s"), ord("S"), 84):
                selected_idx = (selected_idx + 1) % len(options)
            elif key == ord("0"):
                return "ALL"
            elif ord("1") <= key <= ord("9"):
                pick = key - ord("1")
                if pick < len(comp_names):
                    return comp_names[pick]