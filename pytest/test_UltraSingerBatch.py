"""Tests for UltraSingerBatch.py"""

import os
import tempfile
import unittest

from src.UltraSingerBatch import (
    entry_folders,
    entry_mtime,
    find_audio_files,
    find_video_next_to,
    interpreter_without_launcher,
    load_state,
    restore_names,
    save_state,
)


def _touch(*parts: str) -> str:
    path = os.path.join(*parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        file.write("x")
    return path


class FindAudioFilesTest(unittest.TestCase):
    def test_only_audio_files_sorted_and_not_recursive_by_default(self):
        with tempfile.TemporaryDirectory() as root:
            _touch(root, "b.mp3")
            _touch(root, "a.FLAC")
            _touch(root, "notes.txt")
            _touch(root, "sub", "c.mp3")

            names = [os.path.basename(f) for f in find_audio_files(root, recursive=False)]

            self.assertEqual(names, ["a.FLAC", "b.mp3"])

    def test_recursive(self):
        with tempfile.TemporaryDirectory() as root:
            _touch(root, "a.mp3")
            _touch(root, "sub", "c.mp3")

            names = [os.path.basename(f) for f in find_audio_files(root, recursive=True)]

            self.assertEqual(names, ["a.mp3", "c.mp3"])


class StateTest(unittest.TestCase):
    def test_roundtrip_and_missing_file(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "state.json")
            self.assertEqual(load_state(path), {})

            save_state(path, {"song.mp3": {"mtime": 1.5, "folders": ["A - B"]}})

            self.assertEqual(load_state(path), {"song.mp3": {"mtime": 1.5, "folders": ["A - B"]}})

    def test_old_state_format_is_still_understood(self):
        self.assertEqual(entry_mtime(12.5), 12.5)
        self.assertEqual(entry_folders(12.5), [])
        self.assertEqual(entry_mtime({"mtime": 3.0, "folders": ["x"]}), 3.0)
        self.assertEqual(entry_folders({"mtime": 3.0, "folders": ["x"]}), ["x"])


class VideoTest(unittest.TestCase):
    def test_finds_video_with_same_name(self):
        with tempfile.TemporaryDirectory() as root:
            audio = _touch(root, "song.mp3")
            video = _touch(root, "song.mp4")

            self.assertEqual(find_video_next_to(audio), video)

    def test_no_video(self):
        with tempfile.TemporaryDirectory() as root:
            audio = _touch(root, "song.mp3")
            _touch(root, "other.mp4")

            self.assertIsNone(find_video_next_to(audio))


class InterpreterWithoutLauncherTest(unittest.TestCase):
    def _make_venv(self, root: str, base_exists: bool = True, home: bool = True):
        venv_python = _touch(root, "venv", "Scripts", "python.exe")
        if base_exists:
            _touch(root, "base", "python.exe")
        if home:
            with open(os.path.join(root, "venv", "pyvenv.cfg"), "w", encoding="utf-8") as file:
                file.write(f"home = {os.path.join(root, 'base')}\nversion_info = 3.12.13\n")
        return venv_python

    def test_base_python_is_started_with_the_venv_as_environment(self):
        with tempfile.TemporaryDirectory() as root:
            venv_python = self._make_venv(root)

            python, env = interpreter_without_launcher(venv_python)

            self.assertEqual(python, os.path.join(root, "base", "python.exe"))
            self.assertEqual(env, {"__PYVENV_LAUNCHER__": venv_python})

    def test_pythonw_gets_pythonw(self):
        with tempfile.TemporaryDirectory() as root:
            venv_python = _touch(root, "venv", "Scripts", "pythonw.exe")
            _touch(root, "base", "pythonw.exe")
            with open(os.path.join(root, "venv", "pyvenv.cfg"), "w", encoding="utf-8") as file:
                file.write(f"home = {os.path.join(root, 'base')}\n")

            python, _ = interpreter_without_launcher(venv_python)

            self.assertEqual(python, os.path.join(root, "base", "pythonw.exe"))

    def test_without_config_or_base_python_nothing_changes(self):
        with tempfile.TemporaryDirectory() as root:
            venv_python = self._make_venv(root, home=False)
            self.assertEqual(interpreter_without_launcher(venv_python), (venv_python, {}))

        with tempfile.TemporaryDirectory() as root:
            venv_python = self._make_venv(root, base_exists=False)
            self.assertEqual(interpreter_without_launcher(venv_python), (venv_python, {}))


class RestoreNamesTest(unittest.TestCase):
    def test_new_run_gets_the_old_name_back(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "Artist - Song (1)"))

            result = restore_names(root, {"Artist - Song (1)"}, ["Artist - Song"])

            self.assertEqual(result, {"Artist - Song"})
            self.assertTrue(os.path.isdir(os.path.join(root, "Artist - Song")))

    def test_unrelated_numbered_folder_is_kept(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "Other (1)"))

            result = restore_names(root, {"Other (1)"}, ["Artist - Song"])

            self.assertEqual(result, {"Other (1)"})


if __name__ == "__main__":
    unittest.main()
