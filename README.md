# Arnis & Martial Arts Movement Tracking, Storage, and Recognition System

A real-time martial arts and swordplay stance tracker built with **OpenCV**, **MediaPipe**, and **PyTorch**. Record any guard or stance (Arnis, Kendo/Kenjutsu, Longsword, Rapier, or Lightsaber), organize stances into compilation folders, watch 2x demonstration clips, overlay a semi-transparent **Ghost Skeleton** with live joint-angle feedback, and test your form with guided sequences and memory recall drills.

---

## 1. Prerequisites
- **Python 3.11 (64-bit)** is required *(MediaPipe does not support Python 3.14+ yet. Check installed versions on Windows with `py -0`)*.
- A working webcam.

---

## 2. Quick Setup

### Step 1: Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/Arnis-Movement-Tracking-Storage-and-Recognition-System.git](https://github.com/YOUR_USERNAME/Arnis-Movement-Tracking-Storage-and-Recognition-System.git)
cd Arnis-Movement-Tracking-Storage-and-Recognition-System
```

### Step 2: Create & Activate Virtual Environment (Windows PowerShell)
```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

> **Note:** If Windows Defender or drive permissions block creating `venv` on `D:\`, create it on `C:\` instead:
> ```powershell
> py -3.11 -m venv C:\venvs\arnis-env
> C:\venvs\arnis-env\Scripts\Activate.ps1
> ```

### Step 3: Install Dependencies & Download the Pose Model
```powershell
pip install -r requirements.txt
pip install rapidocr_onnxruntime

mkdir -p models
curl.exe -L -o models/pose_landmarker_full.task [https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task)
```

### Step 4: Run the Application
```powershell
python main.py
```

---

## 3. How to Use the Program

### Step 1: Startup Menu
When the camera window launches, select a mode using your keyboard:
- **`1` — Teaching Mode:** Record new reference stances live or import stances from tutorial videos.
- **`2` — Training Mode:** Select a saved stance folder/compilation and practice with Ghost overlays, video demos, and drills.
- **`Q` — Quit:** Close the application.

### Step 2: Building Stances (Teaching Mode)
1. **Record a Live Stance (`R`):**
   - Press **`R`** to trigger a 3-second countdown (`Hold your stance: 3.0s`).
   - Step into frame with your upper body (and legs, if full stance is needed) visible, hold the pose steady, and type a name when prompted inside the window.
   - Live recordings are saved inside `moves_db/Custom_Stances/`.
2. **Import from a Tutorial Video (`I` — Experimental):**
   - Press **`I`**, paste the path to an `.mp4` tutorial video, and enter a folder name for `moves_db/`.
   - The system scans for on-screen chapter titles (or motion pauses) to save each stance and link its video snippet for demo playback.

### Step 3: Practicing & Running Drills (Training Mode)
1. **Choose a Stance Compilation (`L`):**
   - Upon entering Training Mode (or pressing **`L`** anytime), select a specific compilation folder (e.g., *12 Basic Strikes*, *12 Basic Blocks*, *Custom Stances*) or press **`0`** to combine all folders.
2. **Follow the Ghost Skeleton & Joint Telemetry:**
   - Align your live skeleton and virtual baston with the pastel-sky **Ghost Pose**.
   - Check the **Joint Angles** card on the left for live degree measurements and **Quick Tips** (e.g., *"Extend right arm"*, *"Bend left knee more"*).
   - If a stance was imported from a video, a **2-loop video demo** plays automatically before accuracy grading starts. Press **`ESC`** to skip the demo early or **`D`** anytime to replay it.
3. **Run Assessment Drills:**
   - **Guided Sequence (`S`):** Steps through every stance in the active folder in order. Holding **>= 80% accuracy** for `0.6s` automatically advances to the next move.
   - **Phone-Number Recall Test (`T`):** Hides the Ghost Skeleton and generates a randomized 4-stance combo to perform from memory at **>= 80% accuracy**.

---

## 4. Controls (Inside Camera Window)

| Mode | Key | Action |
| :--- | :--- | :--- |
| **Startup Menu** | `1` / `2` | Enter **Teaching Mode** (`1`) or **Training Mode** (`2`) |
| **Teaching Mode** | `R` | Start a 3-second countdown to record and name a new stance |
| **Teaching Mode** | `I` | Import stances and demo clips from an `.mp4` tutorial video |
| **Training Mode** | `L` | Switch active stance compilation folder inside `moves_db/` |
| **Training Mode** | `N` / `P` | Cycle to **Next** (`N`) or **Previous** (`P`) Ghost stance |
| **Training Mode** | `G` | Jump directly to a specific stance number (`1` to `N`) |
| **Training Mode** | `1` – `9` | Quick-select stances `1` through `9` in the active folder |
| **Training Mode** | `D` / `ESC` | Replay (`D`) or skip (`ESC`) the 2x video demonstration |
| **Training Mode** | `S` | Start **80% Auto-Advance Guided Sequence** |
| **Training Mode** | `T` | Start blind **Phone-Number Recall Test** |
| **Training Mode** | `X` | Delete the currently active Ghost stance from disk |
| **Training Mode** | `C` | Clear the active Ghost Skeleton overlay |
| **Global** | `M` / `Q` | Return to **Main Menu** (`M`) or **Quit** the application (`Q`) |

---

## 5. Project Structure & Team Modules

```text
├── models/                  # MediaPipe .task model (ignored by Git)
├── moves_db/                # Categorized stance folders stored as PyTorch .pt tensors
│   ├── Custom_Stances/
│   └── ...
├── src/
│   ├── hud.py               # Role 1: OpenCV rendering, Ghost overlay, telemetry & demo UI
│   ├── extractor.py         # Role 2: MediaPipe PoseLandmarker, joint angles, weapon axis & OCR import
│   ├── library.py           # Role 3: Compilation folder storage, load/delete & move history queue
│   └── matcher.py           # Role 4: PyTorch Cosine Similarity recognition engine
├── main.py                  # Main webcam, mode state machine & drill control loop
├── requirements.txt
└── TEAM_ROLES.md            # Git branching rules & module data contracts
```

---

## 6. Known Limitations
- **Automated Video Import (Experimental):** OCR chapter detection and motion-stillness heuristics depend heavily on video production quality. Fast transitions, decorative title overlays, or multi-person demonstrations can cause transitional poses or misaligned demo clips to be extracted. Recording references directly in **Teaching Mode (`R`)** provides the highest accuracy, and unwanted imported poses can be removed with **`X`**.
- **2D Monocular Depth Ambiguity:** Forward thrusts along the camera's depth (z-axis) are less sensitive than lateral or diagonal strikes.
- **Single-Practitioner Tracking:** Configured for one user in frame at a time under even lighting without severe limb occlusion.
- **Static Posture Grading:** Evaluates peak stance alignment and joint angles rather than continuous swing velocity or impact force.

---

## 7. Git Workflow for Teammates
Never push directly to `main`. Create your assigned feature branch before making changes:
```bash
git checkout -b feature/your-module-name
```