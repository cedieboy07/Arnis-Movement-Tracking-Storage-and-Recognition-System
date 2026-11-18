# Team Division, Data Contracts & Git Workflow

## 1. Git Branching Rules
* **Never push directly to `main`.**
* Create and work exclusively on your assigned feature branch:
  ```bash
  git checkout -b feature/your-module-name
  ```
* When your feature is tested locally with `python main.py`, push your branch and open a **Pull Request (PR)** into `main` on GitHub.
* Always run `git pull origin main` before starting new work so you stay synced with merged PRs.

---

## 2. Shared Data Contracts (Do Not Break)
To allow parallel development across all 4 modules without breaking `main.py`, every module must respect these input/output formats:

* **`PoseExtractor.process_frame(frame_bgr, timestamp_ms)`**
  * **Returns:** `(raw_landmarks, feature_vector)` or `(None, None)` if no person is detected.
  * `raw_landmarks`: `np.ndarray` of shape `(33, 3)` — EMA-smoothed normalized image coordinates `(x, y, z)`.
  * `feature_vector`: `np.ndarray` of shape `(115,)` (`float32`) combining:
    * `99` hip-centered, torso-scaled body coordinates
    * `8` normalized joint angles (elbows, shoulders, hips, knees)
    * `6` left/right virtual weapon direction unit vectors
    * `1` normalized wrist-to-wrist grip distance (1-handed vs. 2-handed detection)
    * `1` shoulder-width rotation ratio (squared vs. bladed stance detection)
* **`PoseMatcher.match(live_vec, saved_moves)`**
  * **Returns:** `(best_name, score_percent)` where `best_name` is `str` or `None` (if below threshold), and `score_percent` is `float` (`0.0` to `100.0`).
* **`MoveLibrary.moves` Dictionary Schema**
  * Maps `stance_name (str)` → `{"name": str, "vector": torch.Tensor(115,), "raw": np.ndarray(33, 3)}`.

---

## 3. Module Ownership & Tasks

### Role 1: HUD & Ghost Replay Lead (`feature/hud`)
* **Owner:** Member 1
* **Primary File:** `src/hud.py`
* **Responsibilities:**
  * Render the live user skeleton and translucent **Ghost Skeleton** target pose.
  * Display the live match status banner, recording countdown alert, and scrolling **Previous Moves** panel.
* **Ready-to-Use Helpers Available from `PoseExtractor`:**
  * Call `extractor.get_weapon_lines(raw_lm, frame.shape)` to get pixel endpoints `((x1, y1), (x2, y2))` for rendering virtual Arnis bastons, swords, or lightsabers (automatically merges into a two-handed blade when wrists are close together).
  * Call `extractor.get_joint_angles_deg(raw_lm)` to display live elbow, shoulder, and knee angles or form-correction hints on screen.

---

### Role 2: Stance & Weapon-Axis Extractor Lead (`feature/extractor`)
* **Owner:** Member 2
* **Primary File:** `src/extractor.py`
* **Responsibilities:**
  * Run MediaPipe `PoseLandmarker` (`RunningMode.VIDEO`) and apply EMA smoothing to prevent landmark jitter.
  * Normalize coordinates relative to hip midpoint and torso scale so camera distance doesn't affect match scores.
  * Extract martial arts joint angles, virtual blade vectors, 1-handed/2-handed grip distance, and torso rotation ratio.
* **Upcoming Improvements:**
  * Add lower-body visibility fallback (zeroing out knee/ankle features when legs are cropped out of the webcam frame).
  * Add frame-to-frame wrist velocity deltas to support dynamic slash recognition.

---

### Role 3: Move Library & Recorder Lead (`feature/library`)
* **Owner:** Member 3
* **Primary File:** `src/library.py`
* **Responsibilities:**
  * Save and load recorded stances as `.pt` files inside `moves_db/`.
  * Maintain the rolling session queue (`self.history`) of previously recognized moves without duplicate spam.
* **Upcoming Improvements:**
  * Add support for deleting/renaming saved stances and exporting session combo logs.
  * Extend storage schema to hold multi-frame sequences `(T, 115)` for dynamic strikes/slashes.

---

### Role 4: PyTorch Matcher Engine Lead (`feature/matcher`)
* **Owner:** Member 4
* **Primary File:** `src/matcher.py`
* **Responsibilities:**
  * Run real-time PyTorch Cosine Similarity (`F.cosine_similarity`) on CPU/GPU between the live `(115,)` tensor and all saved stance tensors.
  * Tune similarity thresholds (`default = 0.88`) to prevent false positives while keeping transitions responsive.
* **Upcoming Improvements:**
  * Add sliding-window sequence matching (e.g., frame-averaged cosine distance or Dynamic Time Warping) so the system recognizes moving slashes and combos in addition to held guards.