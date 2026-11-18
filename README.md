# Arnis & Martial Arts Movement Tracking, Storage, and Recognition System

A real-time martial arts and swordplay stance tracker built with **OpenCV**, **MediaPipe**, and **PyTorch**. Record any guard or stance (Arnis, Kendo/Kenjutsu, Longsword, Rapier, or Lightsaber), overlay a semi-transparent **Ghost Skeleton** to practice your form, and track your recognized stance history in real time.

---

## 1. Prerequisites
* **Python 3.11 (64-bit)** is required *(MediaPipe does not support Python 3.14+ yet. Check installed versions on Windows with `py -0`)*.
* A working webcam.

---

## 2. Quick Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Arnis-Movement-Tracking-Storage-and-Recognition-System.git
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

mkdir -p models
curl.exe -L -o models/pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
```

### Step 4: Run the Application
```powershell
python main.py
```

---

## 3. Controls (Inside Camera Window)

| Key | Action |
| :--- | :--- |
| **`R`** | Start a 3-second countdown to record and name a new stance |
| **`1` – `9`** | Toggle the **Ghost Skeleton** overlay for a saved stance |
| **`C`** | Clear the active Ghost Skeleton overlay |
| **`Q`** | Quit the application |

---

## 4. Project Structure & Team Modules

```text
├── models/                  # MediaPipe .task model (ignored by Git)
├── moves_db/                # Saved stances stored as PyTorch .pt tensors
├── src/
│   ├── hud.py               # Role 1: OpenCV rendering, Ghost overlay & move history UI
│   ├── extractor.py         # Role 2: MediaPipe PoseLandmarker, joint angles & weapon axis
│   ├── library.py           # Role 3: Save/load stances & session move history queue
│   └── matcher.py           # Role 4: PyTorch Cosine Similarity recognition engine
├── main.py                  # Main webcam & keyboard control loop
├── requirements.txt
└── TEAM_ROLES.md            # Git branching rules & module data contracts
```

---

## 5. Git Workflow for Teammates
Never push directly to `main`. Create your assigned feature branch before making changes:
```bash
git checkout -b feature/your-module-name
```