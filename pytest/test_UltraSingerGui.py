"""Tests for UltraSingerGui.py (the logic around the window, not the window itself)"""

import os
import tempfile
import unittest

from src.UltraSingerGui import (
    Settings,
    build_command,
    detect_game_song_folder,
    last_line,
    load_settings,
    parse_batch_line,
    save_settings,
)


class ParseBatchLineTest(unittest.TestCase):
    def test_start_of_a_song(self):
        self.assertEqual(
            parse_batch_line("[2/6] Nacho Vegas & Christina - Me He Perdido.mp3: processing ...\n"),
            ("start", (2, 6, "Nacho Vegas & Christina - Me He Perdido.mp3")),
        )

    def test_skipped_song(self):
        self.assertEqual(parse_batch_line("[1/3] a.mp3: already done, skipping"), ("skipped", (1, 3, "a.mp3")))

    def test_result_of_a_song(self):
        self.assertEqual(parse_batch_line("    done"), ("done", None))
        self.assertEqual(parse_batch_line("    FAILED, see C:\\logs\\a.log"), ("failed", None))

    def test_summary(self):
        self.assertEqual(parse_batch_line("Finished: 4 done, 1 skipped, 2 failed"), ("finished", (4, 1, 2)))

    def test_other_lines_and_color_codes(self):
        self.assertIsNone(parse_batch_line("    copied to C:\\songs\\x"))
        self.assertEqual(parse_batch_line("\x1b[32m    done\x1b[0m"), ("done", None))


class BuildCommandTest(unittest.TestCase):
    def test_all_options(self):
        settings = Settings(input_dir="in", output_dir="out", game_dir="game", language="Español",
                            online_lyrics=True, copy_to_game=True, recursive=True, force=True)

        command = build_command("python", settings)

        self.assertEqual(command[:2], ["python", "-u"])
        self.assertTrue(command[2].endswith("UltraSingerBatch.py"))
        self.assertEqual(command[3:], ["in", "-o", "out", "--copy_to", "game", "--recursive", "--force",
                                       "--language", "es", "--online_lyrics"])

    def test_minimal_options_and_automatic_language(self):
        settings = Settings(input_dir="in", output_dir="out", game_dir="game", language="Detectar automáticamente",
                            online_lyrics=False, copy_to_game=False, recursive=False, force=False)

        self.assertEqual(build_command("python", settings)[3:], ["in", "-o", "out"])

    def test_copy_needs_a_game_folder(self):
        settings = Settings(input_dir="in", output_dir="out", game_dir="", copy_to_game=True,
                            online_lyrics=False, recursive=False)

        self.assertNotIn("--copy_to", build_command("python", settings))


class SettingsTest(unittest.TestCase):
    def test_roundtrip_and_unknown_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "settings.json")
            save_settings(Settings(input_dir="songs", language="English", force=True), path)

            loaded = load_settings(path)

            self.assertEqual((loaded.input_dir, loaded.language, loaded.force), ("songs", "English", True))

    def test_missing_file_gives_defaults(self):
        loaded = load_settings(os.path.join(tempfile.gettempdir(), "does-not-exist-ultrasinger.json"))

        self.assertEqual(loaded.language, "Español")
        self.assertTrue(loaded.online_lyrics)


class HelpersTest(unittest.TestCase):
    def test_detect_game_song_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(detect_game_song_folder(["no-such-folder", folder]), folder)
            self.assertEqual(detect_game_song_folder(["no-such-folder"]), "")

    def test_last_line_of_a_log(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "a.log")
            with open(path, "w", encoding="utf-8") as file:
                file.write("first\n\x1b[34m[UltraSinger] Transcribing\x1b[0m\n\n")

            self.assertEqual(last_line(path), "[UltraSinger] Transcribing")
            self.assertEqual(last_line(os.path.join(folder, "missing.log")), "")


if __name__ == "__main__":
    unittest.main()
