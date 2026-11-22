import time
import cv2
from src.extractor import PoseExtractor
from src.matcher import PoseMatcher
from src.library import MoveLibrary
from src.hud import MartialArtsHUD


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # Request 720p HD resolution for a wider field of view
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Allow resizing or maximizing the display window
    window_name = "Martial Arts Guide"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    extractor = PoseExtractor("models/pose_landmarker_full.task")
    matcher = PoseMatcher(threshold=0.88)
    library = MoveLibrary("moves_db")
    hud = MartialArtsHUD()

    record_deadline = None
    active_ghost = None
    start_time = time.time()

    print("\n=== Martial Arts Stance & Move Guide Ready ===")
    print("Press 'R' in the camera window to record a new stance.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror webcam for natural movement feedback
        frame = cv2.flip(frame, 1)
        timestamp_ms = int((time.time() - start_time) * 1000)

        raw_lm, norm_vec = extractor.process_frame(frame, timestamp_ms)

        # Resolve active Ghost target landmarks if selected
        ghost_raw = (
            library.moves[active_ghost]["raw"]
            if (active_ghost and active_ghost in library.moves)
            else None
        )

        # Draw scale-aligned Ghost overlay + target baston
        if ghost_raw is not None:
            hud.draw_ghost(
                frame,
                ghost_raw,
                label=active_ghost,
                live_raw=raw_lm,
                extractor=extractor,
            )

        # Draw live user skeleton + virtual Arnis baston
        hud.draw_skeleton(
            frame,
            raw_lm,
            thickness=3,
            extractor=extractor,
        )

        # Handle 3-second recording countdown
        status_msg = ""
        if record_deadline is not None:
            remaining = record_deadline - time.time()
            if remaining > 0:
                status_msg = f"HOLD STANCE: {remaining:.1f}s"
            else:
                record_deadline = None
                if norm_vec is not None:
                    cv2.imshow(window_name, frame)
                    cv2.waitKey(1)
                    move_name = input(
                        "\nEnter name for this stance (e.g. Arnis Open Guard): "
                    )
                    if move_name.strip():
                        library.save_move(move_name.strip(), norm_vec, raw_lm)
                        active_ghost = move_name.strip()
                else:
                    print("[WARN] No body detected during capture. Try again.")

        # Live recognition against stored moves
        detected_name, score = matcher.match(norm_vec, library.moves)
        if detected_name:
            library.log_detected_move(detected_name, score)

        # Render full HUD overlay (status pill, joint telemetry, form tips, history)
        hud.draw_ui(
            frame,
            detected_name,
            score,
            library.history,
            status_msg,
            live_raw=raw_lm,
            ghost_raw=ghost_raw,
            extractor=extractor,
            ghost_label=active_ghost,
        )
        cv2.imshow(window_name, frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r") and record_deadline is None:
            print("\n[RECORDING] Get into your stance! Capturing in 3 seconds...")
            record_deadline = time.time() + 3.0
        elif key == ord("c"):
            active_ghost = None
        elif ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            names = library.get_move_names()
            if idx < len(names):
                active_ghost = names[idx]
                print(f"[GHOST] Displaying target stance: {active_ghost}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()