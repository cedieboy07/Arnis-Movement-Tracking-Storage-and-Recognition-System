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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    extractor = PoseExtractor("models/pose_landmarker_full.task")
    matcher = PoseMatcher(threshold=0.88)
    library = MoveLibrary("moves_db")
    hud = MartialArtsHUD()

    mode = "MENU"  # "MENU", "TEACH", or "TRAIN"
    record_deadline = None
    active_ghost = None
    start_time = time.time()

    print("\n=== Martial Arts Stance & Move Guide Ready ===")
    print("Select [1] Teaching Mode or [2] Training Mode in the camera window.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        timestamp_ms = int((time.time() - start_time) * 1000)

        # ---------------- STARTUP MENU ----------------
        if mode == "MENU":
            hud.draw_menu(frame, len(library.moves))
            cv2.imshow("Martial Arts Guide", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("1"):
                mode = "TEACH"
                record_deadline = None
                print("\n[MODE] Entered TEACHING MODE. Press 'R' to record a stance.")
            elif key == ord("2"):
                mode = "TRAIN"
                names = library.get_move_names()
                active_ghost = names[0] if names else None
                print("\n[MODE] Entered TRAINING MODE. Press '1-9' to pick a target stance.")
            continue

        # Run pose extraction for both TEACH and TRAIN modes
        raw_lm, norm_vec = extractor.process_frame(frame, timestamp_ms)

        # ---------------- TEACHING MODE ----------------
        if mode == "TEACH":
            hud.draw_skeleton(frame, raw_lm, extractor=extractor)

            status_msg = ""
            if record_deadline is not None:
                remaining = record_deadline - time.time()
                if remaining > 0:
                    status_msg = f"HOLD STANCE: {remaining:.1f}s"
                else:
                    record_deadline = None
                    if norm_vec is not None:
                        cv2.imshow("Martial Arts Guide", frame)
                        cv2.waitKey(1)
                        move_name = input("\nEnter name for this stance (e.g. Arnis Open Guard): ").strip()
                        if move_name:
                            try:
                                library.save_move(move_name, norm_vec, raw_lm)
                            except ValueError as exc:
                                print(f"[WARN] Could not save stance: {exc}")
                    else:
                        print("[WARN] No body detected during capture. Try again.")

            hud.draw_ui(
                frame,
                detected_name=None,
                score=0.0,
                history=library.history,
                status_msg=status_msg,
                live_raw=raw_lm,
                ghost_raw=None,
                extractor=extractor,
                ghost_label=None,
                mode="TEACH",
            )
            cv2.imshow("Martial Arts Guide", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                mode = "MENU"
                record_deadline = None
            elif key == ord("r") and record_deadline is None:
                print("\n[RECORDING] Get into your stance! Capturing in 3 seconds...")
                record_deadline = time.time() + 3.0

        # ---------------- TRAINING MODE ----------------
        elif mode == "TRAIN":
            ghost_raw = (
                library.moves[active_ghost]["raw"]
                if (active_ghost and active_ghost in library.moves)
                else None
            )

            if ghost_raw is not None:
                hud.draw_ghost(
                    frame,
                    ghost_raw,
                    label=active_ghost,
                    live_raw=raw_lm,
                    extractor=extractor,
                )

            hud.draw_skeleton(frame, raw_lm, extractor=extractor)

            detected_name, score = matcher.match(norm_vec, library.moves)
            if detected_name:
                library.log_detected_move(detected_name, score)

            hud.draw_ui(
                frame,
                detected_name=detected_name,
                score=score,
                history=library.history,
                status_msg="",
                live_raw=raw_lm,
                ghost_raw=ghost_raw,
                extractor=extractor,
                ghost_label=active_ghost,
                mode="TRAIN",
            )
            cv2.imshow("Martial Arts Guide", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                mode = "MENU"
            elif key == ord("c"):
                active_ghost = None
            elif ord("1") <= key <= ord("9"):
                idx = key - ord("1")
                names = library.get_move_names()
                if idx < len(names):
                    active_ghost = names[idx]
                    print(f"[GHOST] Target stance: {active_ghost}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()