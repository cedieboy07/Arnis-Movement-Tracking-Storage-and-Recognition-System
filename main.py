import random
import time
from pathlib import Path
import cv2
import torch
from src.extractor import PoseExtractor
from src.matcher import PoseMatcher
from src.library import MoveLibrary
from src.hud import MartialArtsHUD

PASS_THRESHOLD = 80.0   # 80% match required to advance to the next stance
HOLD_REQUIRED = 0.6     # Must hold >= 80% for 0.6 seconds so accidental swings don't trigger


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    extractor = PoseExtractor("models/pose_landmarker_full.task")
    matcher = PoseMatcher(threshold=0.80)
    library = MoveLibrary("moves_db")
    hud = MartialArtsHUD()

    mode = "MENU"             # "MENU", "TEACH", or "TRAIN"
    train_submode = "FREE"    # "FREE", "SEQUENCE" (80% guided), or "PHONE" (blind recall test)
    record_deadline = None
    active_ghost = None

    # Sequence & Phone-Number Drill State
    sequence_list = []
    phone_digits = []
    seq_idx = 0
    hold_start = None
    banner_msg = ""
    banner_until = 0.0

    # Demo Playback State
    demo_cap = None
    demo_playing = False
    demo_start_time = 0.0
    last_target = None
    current_target = None

    start_time = time.time()

    print("\n=== Martial Arts Stance & Move Guide Ready ===")
    print("Select [1] Teaching Mode or [2] Training Mode in the camera window.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        now = time.time()
        timestamp_ms = int((now - start_time) * 1000)

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
                print("\n[MODE] TEACHING MODE: Press 'R' to record live or 'I' to import a tutorial video.")
            elif key == ord("2"):
                mode = "TRAIN"
                train_submode = "FREE"
                names = library.get_move_names()
                active_ghost = names[0] if names else None
                last_target = None
                current_target = None
                demo_playing = False
                print("\n[MODE] TRAINING MODE: Press 'S' for 80% Guided Sequence or 'T' for Phone-Number Recall Test.")
            continue

        raw_lm, norm_vec = extractor.process_frame(frame, timestamp_ms)

        # ---------------- TEACHING MODE ----------------
        if mode == "TEACH":
            hud.draw_skeleton(frame, raw_lm, extractor=extractor)

            status_msg = banner_msg if now < banner_until else ""
            if record_deadline is not None:
                remaining = record_deadline - now
                if remaining > 0:
                    status_msg = f"HOLD STANCE: {remaining:.1f}s"
                else:
                    record_deadline = None
                    if norm_vec is not None:
                        move_name = hud.prompt_text_in_window(frame, "Enter name for this stance:")
                        if move_name:
                            try:
                                library.save_move(move_name, norm_vec, raw_lm)
                                banner_msg = f"Saved: {move_name}"
                                banner_until = time.time() + 2.0
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
                record_deadline = now + 3.0
            elif key == ord("i") and record_deadline is None:
             video_path = hud.prompt_text_in_window(frame, "Enter path to .mp4 tutorial video:").strip('"')
             if video_path:
                 default_folder = Path(video_path).stem
                 folder_name = (
                     hud.prompt_text_in_window(
                         frame,
                         "Folder name in moves_db/ for this video:",
                         default_text=default_folder,
                     )
                     or default_folder
                 )
                 try:
                     print(f"[IMPORT] Scanning '{video_path}' into moves_db/{folder_name}/...")
                     chapters = extractor.extract_labeled_chapters_from_video(video_path)
                     if chapters:
                         for stance_name, s_raw, s_vec, s_clip in chapters:
                             library.save_move(stance_name, s_vec, s_raw, compilation=folder_name)
                             library.moves[stance_name]["demo_clip"] = s_clip
                             torch.save(
                                 library.moves[stance_name],
                                 library._move_path(stance_name, compilation=folder_name),
                             )
                         library.set_active_compilation(library._sanitize_filename(folder_name))
                         banner_msg = f"Saved {len(chapters)} stances to '{folder_name}'!"
                         banner_until = time.time() + 2.5
                     else:
                         prefix = (
                             hud.prompt_text_in_window(
                                 frame,
                                 "No text labels found. Enter stance prefix:",
                                 default_text="Tutorial",
                             )
                             or "Tutorial"
                         )
                         stances = extractor.extract_stances_from_video(video_path)
                         for idx, (s_raw, s_vec, s_clip) in enumerate(stances, start=1):
                             stance_name = f"{prefix}_Step_{idx}"
                             library.save_move(stance_name, s_vec, s_raw, compilation=folder_name)
                             library.moves[stance_name]["demo_clip"] = s_clip
                             torch.save(
                                 library.moves[stance_name],
                                 library._move_path(stance_name, compilation=folder_name),
                             )
                         library.set_active_compilation(library._sanitize_filename(folder_name))
                         banner_msg = f"Saved {len(stances)} stances to '{folder_name}'!"
                         banner_until = time.time() + 2.5
                 except Exception as exc:
                     print(f"[ERROR] Video import failed: {exc}")

        # ---------------- TRAINING MODE ----------------
        elif mode == "TRAIN":
            names = library.get_move_names()

            # Determine active target stance for SEQUENCE or PHONE submodes
            target_stance = None
            if train_submode in {"SEQUENCE", "PHONE"} and seq_idx < len(sequence_list):
                target_stance = sequence_list[seq_idx]
                active_ghost = target_stance if train_submode == "SEQUENCE" else None

            # Auto-play demo clip once whenever target stance changes (skipped in blind PHONE mode)
            if not demo_playing:
                current_target = (
                    target_stance
                    if train_submode == "SEQUENCE"
                    else (active_ghost if train_submode == "FREE" else None)
                )
                if current_target != last_target:
                    last_target = current_target
                    clip_meta = library.moves.get(current_target, {}).get("demo_clip") if current_target else None
                    if isinstance(clip_meta, dict) and Path(clip_meta["video_path"]).exists():
                        if demo_cap is not None:
                            demo_cap.release()
                        demo_cap = cv2.VideoCapture(clip_meta["video_path"])
                        demo_cap.set(cv2.CAP_PROP_POS_FRAMES, clip_meta["start_frame"])
                        demo_playing = True
                        demo_start_time = now

            # Play full stance chapter clip for 2 loops before accuracy detection starts
            clip_meta = library.moves.get(current_target, {}).get("demo_clip") if current_target else None
            if demo_playing and isinstance(clip_meta, dict) and demo_cap is not None:
                fps = clip_meta.get("fps", 30.0)
                start_f = clip_meta["start_frame"]
                end_f = clip_meta["end_frame"]
                clip_len = max(1, end_f - start_f + 1)
                total_play_frames = clip_len * 2  # 2 full loops

                elapsed_frames = int((now - demo_start_time) * fps)
                if elapsed_frames < total_play_frames:
                    loop_num = (elapsed_frames // clip_len) + 1
                    desired_frame = start_f + (elapsed_frames % clip_len)

                    current_pos = int(demo_cap.get(cv2.CAP_PROP_POS_FRAMES))
                    # Re-seek once when Loop 2 wraps back to start_f
                    if desired_frame < current_pos:
                        demo_cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                        current_pos = start_f

                    vid_frame = None
                    while current_pos <= desired_frame:
                        ok, grabbed = demo_cap.read()
                        if not ok:
                            break
                        vid_frame = grabbed
                        current_pos += 1

                    if vid_frame is not None:
                        hud.draw_demo_clip(frame, vid_frame, label=f"{current_target} (Loop {loop_num}/2)")
                        cv2.imshow("Martial Arts Guide", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                        elif key == ord("m"):
                            mode = "MENU"
                            train_submode = "FREE"
                            demo_playing = False
                        elif key == 27:  # ESC skips demo early
                            demo_playing = False
                            hold_start = None
                        continue

                demo_playing = False
                hold_start = None

            ghost_raw = (
                library.moves[active_ghost]["raw"]
                if (active_ghost and active_ghost in library.moves)
                else None
            )

            # Build numbered label (e.g., "[14/32] 01_LEFT_TEMPLE")
            ghost_badge = active_ghost
            if active_ghost and active_ghost in names:
                pos = names.index(active_ghost) + 1
                ghost_badge = f"[{pos}/{len(names)}] {active_ghost}"

            if ghost_raw is not None:
                hud.draw_ghost(
                    frame,
                    ghost_raw,
                    label=ghost_badge,
                    live_raw=raw_lm,
                    extractor=extractor,
                )

            hud.draw_skeleton(frame, raw_lm, extractor=extractor)

            # Evaluate against target stance (in SEQUENCE/PHONE) or full library (in FREE)
            if target_stance and target_stance in library.moves:
                detected_name, score = matcher.match(norm_vec, {target_stance: library.moves[target_stance]})
            else:
                detected_name, score = matcher.match(norm_vec, library.moves)

            if detected_name:
                library.log_detected_move(detected_name, score)

            # Handle 80% Auto-Advance Progression for SEQUENCE & PHONE drills
            hold_progress = 0.0
            if target_stance is not None:
                if score >= PASS_THRESHOLD:
                    if hold_start is None:
                        hold_start = now
                    elapsed = now - hold_start
                    hold_progress = min(1.0, elapsed / HOLD_REQUIRED)
                    if elapsed >= HOLD_REQUIRED:
                        hold_start = None
                        seq_idx += 1
                        if seq_idx >= len(sequence_list):
                            label = "PHONE SEQUENCE COMPLETE!" if train_submode == "PHONE" else "SEQUENCE COMPLETE!"
                            banner_msg = label
                            banner_until = now + 3.0
                            train_submode = "FREE"
                            active_ghost = None
                            last_target = None
                else:
                    hold_start = None

            status_msg = banner_msg if now < banner_until else ""
            hud.draw_ui(
                frame,
                detected_name=detected_name,
                score=score,
                history=library.history,
                status_msg=status_msg,
                live_raw=raw_lm,
                ghost_raw=ghost_raw,
                extractor=extractor,
                ghost_label=ghost_badge,
                mode="TRAIN",
            )

            # Render sequence / phone-number drill card if active
            if train_submode == "SEQUENCE" and sequence_list:
                hud.draw_drill_banner(
                    frame,
                    f"GUIDED SEQUENCE ({seq_idx + 1}/{len(sequence_list)}) - Hold >= 80%",
                    sequence_list[seq_idx : seq_idx + 3],
                    0,
                    hold_progress,
                )
            elif train_submode == "PHONE" and phone_digits:
                hud.draw_drill_banner(
                    frame,
                    "PHONE SEQUENCE RECALL (Perform numbers in order, >= 80%)",
                    phone_digits,
                    seq_idx,
                    hold_progress,
                )

            cv2.imshow("Martial Arts Guide", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                mode = "MENU"
                train_submode = "FREE"
                demo_playing = False
            elif key == ord("c"):
                train_submode = "FREE"
                active_ghost = None
                last_target = None
                demo_playing = False
            elif key == ord("d"):
                replay_target = target_stance if train_submode in {"SEQUENCE", "PHONE"} else active_ghost
                clip_meta = library.moves.get(replay_target, {}).get("demo_clip") if replay_target else None
                if isinstance(clip_meta, dict) and Path(clip_meta["video_path"]).exists():
                    if demo_cap is not None:
                        demo_cap.release()
                    demo_cap = cv2.VideoCapture(clip_meta["video_path"])
                    demo_cap.set(cv2.CAP_PROP_POS_FRAMES, clip_meta["start_frame"])
                    current_target = replay_target
                    demo_playing = True
                    demo_start_time = now
                    hold_start = None
            elif key == ord("n") and names:
                train_submode = "FREE"
                curr_idx = names.index(active_ghost) if active_ghost in names else -1
                active_ghost = names[(curr_idx + 1) % len(names)]
            elif key == ord("p") and names:
                train_submode = "FREE"
                curr_idx = names.index(active_ghost) if active_ghost in names else 0
                active_ghost = names[(curr_idx - 1) % len(names)]
            elif key == ord("g") and names:
                choice = hud.prompt_text_in_window(frame, f"Jump to stance number (1-{len(names)}):")
                if choice.isdigit() and 1 <= int(choice) <= len(names):
                    train_submode = "FREE"
                    active_ghost = names[int(choice) - 1]
            elif key == ord("x") and active_ghost:
                if library.delete_move(active_ghost):
                    banner_msg = f"Deleted: {active_ghost}"
                    banner_until = now + 2.0
                    names = library.get_move_names()
                    active_ghost = names[0] if names else None
                    last_target = None
            elif key == ord("s") and names:
                train_submode = "SEQUENCE"
                sequence_list = list(names)
                seq_idx = 0
                hold_start = None
                last_target = None
            elif key == ord("t") and names:
                train_submode = "PHONE"
                chosen_indices = [random.randrange(len(names)) for _ in range(4)]
                phone_digits = [f"#{i + 1} ({names[i]})" for i in chosen_indices]
                sequence_list = [names[i] for i in chosen_indices]
                seq_idx = 0
                hold_start = None
                last_target = None
                demo_playing = False
            elif key == ord("l") and library.compilations:
             chosen = hud.select_compilation_in_window(
                 frame, library.compilations, library.active_compilation
             )
             if chosen:
                 library.set_active_compilation(chosen)
                 train_submode = "FREE"
                 names = library.get_move_names()
                 active_ghost = names[0] if names else None
                 last_target = None
                 current_target = None
                 demo_playing = False
            elif ord("1") <= key <= ord("9"):
                idx = key - ord("1")
                if idx < len(names):
                    train_submode = "FREE"
                    active_ghost = names[idx]

    if demo_cap is not None:
        demo_cap.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()