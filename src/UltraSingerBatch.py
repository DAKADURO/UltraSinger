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
import re
import shutil
import subprocess
import sys

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".aac"}
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov")
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


def entry_mtime(entry):
    """State entries are {"mtime": ..., "folders": [...]}; older versions stored only the mtime"""
    return entry.get("mtime") if isinstance(entry, dict) else entry


def entry_folders(entry) -> list[str]:
    return entry.get("folders", []) if isinstance(entry, dict) else []


def find_video_next_to(audio_path: str) -> str | None:
    """A video with the same name next to the audio file, e.g. song.mp3 + song.mp4"""
    stem = os.path.splitext(audio_path)[0]
    for extension in VIDEO_EXTENSIONS:
        if os.path.isfile(stem + extension):
            return stem + extension
    return None


def remove_song_folders(folders: list[str], *parent_dirs: str) -> None:
    for parent in parent_dirs:
        if not parent:
            continue
        for folder in folders:
            shutil.rmtree(os.path.join(parent, folder), ignore_errors=True)


def restore_names(output_dir: str, new_folders: set[str], replaced: list[str]) -> set[str]:
    """UltraSinger names a second run 'Song (1)'. When it replaces 'Song', give it the old name back."""
    result = set()
    for folder in new_folders:
        match = re.match(r"^(.*) \(\d+\)$", folder)
        if match and match.group(1) in replaced and not os.path.exists(os.path.join(output_dir, match.group(1))):
            os.rename(os.path.join(output_dir, folder), os.path.join(output_dir, match.group(1)))
            folder = match.group(1)
        result.add(folder)
    return result


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


def interpreter_without_launcher(python: str) -> tuple[str, dict]:
    """Python to start and the environment it needs.

    The python.exe of a virtual environment made by uv is a launcher that starts the base python. On some
    systems the launcher does not find it ("No Python at ..."). Then the base python is started directly and
    told which virtual environment to use, which is what the launcher does."""
    config = os.path.join(os.path.dirname(os.path.dirname(python)), "pyvenv.cfg")
    try:
        with open(config, encoding="utf-8") as file:
            for line in file:
                key, _, value = line.partition("=")
                if key.strip() == "home":
                    base = os.path.join(value.strip(), os.path.basename(python))
                    if os.path.isfile(base):
                        return base, {"__PYVENV_LAUNCHER__": python}
    except OSError:
        pass
    return python, {}


def process_song(audio_path: str, output_dir: str, extra_args: list[str], log_path: str) -> bool:
    python, python_env = interpreter_without_launcher(sys.executable)
    command = [python, "-W", "ignore", ULTRASINGER_SCRIPT, "-i", audio_path, "-o", output_dir, *extra_args]
    env = {**os.environ, **python_env, "PYTHONIOENCODING": "utf-8"}
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

        previous = state.get(key)
        if not args.force and previous is not None and entry_mtime(previous) == mtime:
            print(f"{prefix}: already done, skipping")
            skipped.append(name)
            continue

        print(f"{prefix}: processing ...", flush=True)
        before = list_song_folders(output_dir)
        log_path = os.path.join(log_dir, os.path.splitext(name)[0] + ".log")
        song_args = list(extra_args)
        video_path = None if "--video" in extra_args else find_video_next_to(audio_path)
        if video_path:
            print(f"    using video {os.path.basename(video_path)}")
            song_args += ["--video", os.path.abspath(video_path)]
        if process_song(os.path.abspath(audio_path), output_dir, song_args, log_path):
            replaced = entry_folders(previous)
            remove_song_folders(replaced, output_dir, args.copy_to)
            new_folders = restore_names(output_dir, list_song_folders(output_dir) - before, replaced)
            state[key] = {"mtime": mtime, "folders": sorted(new_folders)}
            save_state(state_path, state)
            done.append(name)
            print("    done")
            if args.copy_to:
                copy_songs(output_dir, new_folders, args.copy_to)
        else:
            failed.append((name, last_error_line(log_path)))
            print(f"    FAILED, see {log_path}")

    print(f"\nFinished: {len(done)} done, {len(skipped)} skipped, {len(failed)} failed")
    for name, reason in failed:
        print(f"  - {name}: {reason}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
