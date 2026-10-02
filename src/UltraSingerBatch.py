"""Process all audio files of a folder with UltraSinger.

Example:
    python UltraSingerBatch.py "H:/Songs" -o "H:/output" --copy_to "C:/UltraStar/songs" --language es

Every option that is not known to this script is passed on to UltraSinger.py.
Each song runs in its own process, so GPU memory is released between songs and a failing
song does not stop the others. Finished songs are remembered in <output>/.batch_done.json
and skipped on the next run (use --force to process them again).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".aac"}
STATE_FILE_NAME = ".batch_done.json"
LOG_FOLDER_NAME = "batch_logs"
ULTRASINGER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UltraSinger.py")


def find_audio_files(input_dir: str, recursive: bool) -> list[str]:
    files = []
    for root, dirs, names in os.walk(input_dir):
        dirs.sort()
        for name in sorted(names):
            if os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS:
                files.append(os.path.join(root, name))
        if not recursive:
            break
    return files


def load_state(state_path: str) -> dict:
    try:
        with open(state_path, encoding="utf-8") as file:
            return json.load(file)
    except (OSError, ValueError):
        return {}


def save_state(state_path: str, state: dict) -> None:
    with open(state_path, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, ensure_ascii=False)


def list_song_folders(output_dir: str) -> set[str]:
    return {
        name for name in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, name)) and name != LOG_FOLDER_NAME
    }


def copy_songs(output_dir: str, folders: set[str], target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)
    for folder in sorted(folders):
        shutil.copytree(os.path.join(output_dir, folder), os.path.join(target_dir, folder), dirs_exist_ok=True)
        print(f"    copied to {os.path.join(target_dir, folder)}")


def last_error_line(log_path: str) -> str:
    try:
        with open(log_path, encoding="utf-8", errors="replace") as file:
            lines = [line.strip() for line in file if line.strip()]
    except OSError:
        return ""
    return lines[-1] if lines else ""


def process_song(audio_path: str, output_dir: str, extra_args: list[str], log_path: str) -> bool:
    command = [sys.executable, "-W", "ignore", ULTRASINGER_SCRIPT, "-i", audio_path, "-o", output_dir, *extra_args]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=os.path.dirname(ULTRASINGER_SCRIPT))
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", help="Folder with the audio files")
    parser.add_argument("-o", "--output", required=True, help="Output folder for the UltraStar songs")
    parser.add_argument("--copy_to", help="Also copy every finished song folder to this folder (e.g. the game's songs folder)")
    parser.add_argument("--recursive", action="store_true", help="Also look into sub folders")
    parser.add_argument("--force", action="store_true", help="Process songs again that were already done")
    args, extra_args = parser.parse_known_args()

    if not os.path.isdir(args.input_dir):
        print(f"Input folder not found: {args.input_dir}")
        return 2

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, LOG_FOLDER_NAME)
    os.makedirs(log_dir, exist_ok=True)
    state_path = os.path.join(output_dir, STATE_FILE_NAME)
    state = load_state(state_path)

    audio_files = find_audio_files(args.input_dir, args.recursive)
    print(f"Found {len(audio_files)} audio file(s) in {args.input_dir}")

    done, skipped, failed = [], [], []
    for index, audio_path in enumerate(audio_files, start=1):
        name = os.path.basename(audio_path)
        key = os.path.abspath(audio_path)
        mtime = os.path.getmtime(audio_path)
        prefix = f"[{index}/{len(audio_files)}] {name}"

        if not args.force and state.get(key) == mtime:
            print(f"{prefix}: already done, skipping")
            skipped.append(name)
            continue

        print(f"{prefix}: processing ...", flush=True)
        before = list_song_folders(output_dir)
        log_path = os.path.join(log_dir, os.path.splitext(name)[0] + ".log")
        if process_song(os.path.abspath(audio_path), output_dir, extra_args, log_path):
            state[key] = mtime
            save_state(state_path, state)
            done.append(name)
            print("    done")
            if args.copy_to:
                copy_songs(output_dir, list_song_folders(output_dir) - before, args.copy_to)
        else:
            failed.append((name, last_error_line(log_path)))
            print(f"    FAILED, see {log_path}")

    print(f"\nFinished: {len(done)} done, {len(skipped)} skipped, {len(failed)} failed")
    for name, reason in failed:
        print(f"  - {name}: {reason}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
