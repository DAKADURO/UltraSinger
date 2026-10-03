"""Small window to process a folder of songs with UltraSinger (uses UltraSingerBatch.py)."""

import json
import os
import queue
import re
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, fields
from tkinter import BooleanVar, StringVar, Tk, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from UltraSingerBatch import (
    ULTRASINGER_SCRIPT,
    LOG_FOLDER_NAME,
    STATE_FILE_NAME,
    entry_mtime,
    find_audio_files,
    load_state,
)

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".ultrasinger_gui.json")
BATCH_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UltraSingerBatch.py")
ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Language shown in the window -> value of --language (None: let Whisper detect it)
LANGUAGES = {
    "Español": "es",
    "English": "en",
    "Português": "pt",
    "Français": "fr",
    "Italiano": "it",
    "Deutsch": "de",
    "Català": "ca",
    "Detectar automáticamente": None,
}
GAME_SONG_FOLDERS = [
    r"C:\Program Files (x86)\UltraStar WorldParty\songs",
    r"C:\Program Files\UltraStar WorldParty\songs",
    r"C:\Program Files (x86)\UltraStar Deluxe\songs",
    r"C:\Program Files\UltraStar Deluxe\songs",
]

PENDING, RUNNING, DONE, FAILED, SKIPPED, STOPPED = "Pendiente", "Procesando…", "Hecha", "Fallo", "Ya estaba hecha", "Detenida"

START_LINE = re.compile(r"^\[(\d+)/(\d+)\] (.+): (processing \.\.\.|already done, skipping)$")
FINISHED_LINE = re.compile(r"^Finished: (\d+) done, (\d+) skipped, (\d+) failed")


@dataclass
class Settings:
    input_dir: str = ""
    output_dir: str = ""
    game_dir: str = ""
    language: str = "Español"
    online_lyrics: bool = True
    copy_to_game: bool = True
    recursive: bool = True
    force: bool = False


def load_settings(path: str = SETTINGS_PATH) -> Settings:
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, ValueError):
        return Settings(game_dir=detect_game_song_folder())
    known = {field.name for field in fields(Settings)}
    return Settings(**{key: value for key, value in data.items() if key in known})


def save_settings(settings: Settings, path: str = SETTINGS_PATH) -> None:
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(asdict(settings), file, indent=2, ensure_ascii=False)
    except OSError:
        pass


def detect_game_song_folder(candidates: list[str] = None) -> str:
    for folder in candidates if candidates is not None else GAME_SONG_FOLDERS:
        if os.path.isdir(folder):
            return folder
    return ""


def build_command(python: str, settings: Settings) -> list[str]:
    command = [python, "-u", BATCH_SCRIPT, settings.input_dir, "-o", settings.output_dir]
    if settings.copy_to_game and settings.game_dir:
        command += ["--copy_to", settings.game_dir]
    if settings.recursive:
        command.append("--recursive")
    if settings.force:
        command.append("--force")
    language = LANGUAGES.get(settings.language)
    if language:
        command += ["--language", language]
    if settings.online_lyrics:
        command.append("--online_lyrics")
    return command


def parse_batch_line(line: str):
    """Event of a line printed by UltraSingerBatch.py: (kind, data) or None"""
    line = ANSI.sub("", line.rstrip("\r\n"))
    match = START_LINE.match(line)
    if match:
        kind = "start" if match.group(4).startswith("processing") else "skipped"
        return kind, (int(match.group(1)), int(match.group(2)), match.group(3))
    stripped = line.strip()
    if stripped == "done":
        return "done", None
    if stripped.startswith("FAILED"):
        return "failed", None
    match = FINISHED_LINE.match(line)
    if match:
        return "finished", tuple(int(value) for value in match.groups())
    return None


def python_for_batch() -> str:
    """The console python next to the one running the window (pythonw has no console to print to)"""
    executable = sys.executable
    if os.path.basename(executable).lower() == "pythonw.exe":
        return os.path.join(os.path.dirname(executable), "python.exe")
    return executable


def last_line(path: str) -> str:
    try:
        with open(path, "rb") as file:
            file.seek(0, os.SEEK_END)
            file.seek(max(0, file.tell() - 4096))
            text = file.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    lines = [ANSI.sub("", line).strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


class App:
    def __init__(self, root: Tk):
        self.root = root
        root.title("UltraSinger")
        root.geometry("860x640")
        root.minsize(720, 520)
        self.settings = load_settings()

        self.input_dir = StringVar(value=self.settings.input_dir)
        self.output_dir = StringVar(value=self.settings.output_dir)
        self.game_dir = StringVar(value=self.settings.game_dir)
        self.language = StringVar(value=self.settings.language)
        self.online_lyrics = BooleanVar(value=self.settings.online_lyrics)
        self.copy_to_game = BooleanVar(value=self.settings.copy_to_game)
        self.recursive = BooleanVar(value=self.settings.recursive)
        self.force = BooleanVar(value=self.settings.force)
        self.status = StringVar(value="Elige una carpeta con canciones.")

        self.process = None
        self.events = queue.Queue()
        self.rows = {}  # batch index -> tree item
        self.option_widgets = []  # disabled while songs are processed
        self.current_index = 0
        self.current_log = None
        self.output_for_logs = ""
        self.stopping = False

        self.build_widgets()
        self.refresh_songs()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(200, self.poll)

    # ----- widgets
    def build_widgets(self):
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        self.folder_row(frame, 0, "Canciones", self.input_dir, self.choose_input)
        self.folder_row(frame, 1, "Resultados", self.output_dir, self.choose_output)
        self.game_row(frame, 2)

        options = ttk.Frame(frame)
        options.grid(row=3, column=0, columnspan=3, sticky="we", pady=(8, 4))
        ttk.Label(options, text="Idioma:").pack(side="left")
        self.language_box = ttk.Combobox(options, textvariable=self.language, values=list(LANGUAGES), state="readonly", width=24)
        self.language_box.pack(side="left", padx=(4, 16))
        self.option_widgets.append(self.language_box)
        for text, variable in (
            ("Buscar la letra en LRCLIB", self.online_lyrics),
            ("Copiar al juego", self.copy_to_game),
            ("Incluir subcarpetas", self.recursive),
            ("Rehacer las ya hechas", self.force),
        ):
            check = ttk.Checkbutton(options, text=text, variable=variable, command=self.refresh_songs if variable is self.recursive else None)
            check.pack(side="left", padx=(0, 12))
            self.option_widgets.append(check)

        columns = ("song", "state")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=10)
        self.tree.heading("song", text="Canción")
        self.tree.heading("state", text="Estado")
        self.tree.column("song", width=560, anchor="w")
        self.tree.column("state", width=140, anchor="w")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=4)
        scroll.grid(row=4, column=2, sticky="ns", pady=4)
        frame.rowconfigure(4, weight=3)

        buttons = ttk.Frame(frame)
        buttons.grid(row=5, column=0, columnspan=3, sticky="we", pady=4)
        self.start_button = ttk.Button(buttons, text="Empezar", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="Parar", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=6)
        ttk.Button(buttons, text="Abrir resultados", command=lambda: self.open_folder(self.output_dir.get())).pack(side="left")
        self.progress = ttk.Progressbar(buttons, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=12)

        ttk.Label(frame, textvariable=self.status, anchor="w").grid(row=6, column=0, columnspan=3, sticky="we")
        self.log = ScrolledText(frame, height=9, state="disabled", font=("Consolas", 9))
        self.log.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=(4, 0))
        frame.rowconfigure(7, weight=2)

    def folder_row(self, frame, row, label, variable, command):
        ttk.Label(frame, text=label + ":").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(frame, textvariable=variable).grid(row=row, column=1, sticky="we", padx=6)
        button = ttk.Button(frame, text="Elegir…", command=command)
        button.grid(row=row, column=2)
        self.option_widgets.append(button)

    def game_row(self, frame, row):
        ttk.Label(frame, text="Juego:").grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(frame, textvariable=self.game_dir)
        entry.grid(row=row, column=1, sticky="we", padx=6)
        button = ttk.Button(frame, text="Elegir…", command=self.choose_game)
        button.grid(row=row, column=2)
        self.option_widgets += [button, entry]

    # ----- choosing folders
    def choose_input(self):
        folder = filedialog.askdirectory(title="Carpeta con las canciones", initialdir=self.input_dir.get() or None)
        if folder:
            self.input_dir.set(os.path.normpath(folder))
            if not self.output_dir.get():
                self.output_dir.set(os.path.join(os.path.normpath(folder), "UltraSinger"))
            self.refresh_songs()

    def choose_output(self):
        folder = filedialog.askdirectory(title="Carpeta para los resultados", initialdir=self.output_dir.get() or None)
        if folder:
            self.output_dir.set(os.path.normpath(folder))
            self.refresh_songs()

    def choose_game(self):
        folder = filedialog.askdirectory(title="Carpeta de canciones del juego", initialdir=self.game_dir.get() or None)
        if folder:
            self.game_dir.set(os.path.normpath(folder))

    @staticmethod
    def open_folder(folder: str):
        if folder and os.path.isdir(folder):
            os.startfile(folder)

    # ----- song list
    def refresh_songs(self):
        if self.process:
            return
        self.tree.delete(*self.tree.get_children())
        self.rows = {}
        folder = self.input_dir.get()
        if not folder or not os.path.isdir(folder):
            return
        files = find_audio_files(folder, self.recursive.get())
        state = load_state(os.path.join(self.output_dir.get(), STATE_FILE_NAME)) if self.output_dir.get() else {}
        pending = 0
        for index, path in enumerate(files, start=1):
            done = entry_mtime(state.get(os.path.abspath(path))) == os.path.getmtime(path)
            pending += 0 if done else 1
            self.rows[index] = self.tree.insert("", "end", values=(os.path.relpath(path, folder), SKIPPED if done else PENDING))
        self.status.set(f"{len(files)} canciones encontradas, {pending} por procesar.")
        self.progress.configure(value=0, maximum=max(1, len(files)))

    def set_state(self, index: int, state: str):
        item = self.rows.get(index)
        if item:
            self.tree.set(item, "state", state)
            self.tree.see(item)

    # ----- running
    def read_settings(self) -> Settings:
        return Settings(
            input_dir=self.input_dir.get().strip(),
            output_dir=self.output_dir.get().strip(),
            game_dir=self.game_dir.get().strip(),
            language=self.language.get(),
            online_lyrics=self.online_lyrics.get(),
            copy_to_game=self.copy_to_game.get(),
            recursive=self.recursive.get(),
            force=self.force.get(),
        )

    def start(self):
        settings = self.read_settings()
        if not os.path.isdir(settings.input_dir):
            messagebox.showwarning("UltraSinger", "Elige primero la carpeta con las canciones.")
            return
        if not settings.output_dir:
            messagebox.showwarning("UltraSinger", "Elige la carpeta para los resultados.")
            return
        if settings.online_lyrics and not messagebox.askyesno(
            "UltraSinger",
            "Para buscar la letra se envían el artista y el título de cada canción a lrclib.net.\n\n¿Continuar?",
        ):
            return
        save_settings(settings)
        self.refresh_songs()
        self.set_running(True)
        self.stopping = False
        self.add_log(f"> {' '.join(build_command(python_for_batch(), settings))}\n")
        self.process = subprocess.Popen(
            build_command(python_for_batch(), settings),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            cwd=os.path.dirname(ULTRASINGER_SCRIPT),
        )
        self.output_for_logs = settings.output_dir
        threading.Thread(target=self.read_output, args=(self.process,), daemon=True).start()

    def read_output(self, process):
        for line in process.stdout:
            self.events.put(("line", line))
        process.wait()
        self.events.put(("exit", process.returncode))

    def stop(self):
        if self.process and self.process.poll() is None:
            self.stopping = True
            subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def set_running(self, running: bool):
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        for widget in self.option_widgets:
            try:
                widget.configure(state="disabled" if running else ("readonly" if widget is self.language_box else "normal"))
            except Exception:
                pass

    def poll(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "line":
                    self.handle_line(data)
                else:
                    self.on_exit(data)
        except queue.Empty:
            pass
        if self.process and self.current_log:
            tail = last_line(self.current_log)
            if tail:
                self.status.set(tail[:140])
        self.root.after(500, self.poll)

    def handle_line(self, line: str):
        self.add_log(ANSI.sub("", line))
        event = parse_batch_line(line)
        if not event:
            return
        kind, data = event
        if kind in ("start", "skipped"):
            index, total, name = data
            self.current_index = index
            self.progress.configure(maximum=total)
            if kind == "start":
                self.set_state(index, RUNNING)
                self.current_log = os.path.join(self.output_for_logs, LOG_FOLDER_NAME, os.path.splitext(name)[0] + ".log")
            else:
                self.set_state(index, SKIPPED)
                self.progress.configure(value=index)
        elif kind == "done":
            self.set_state(self.current_index, DONE)
            self.progress.configure(value=self.current_index)
        elif kind == "failed":
            self.set_state(self.current_index, FAILED)
            self.progress.configure(value=self.current_index)
        elif kind == "finished":
            done, skipped, failed = data
            self.status.set(f"Terminado: {done} hechas, {skipped} saltadas, {failed} con fallo.")

    def on_exit(self, code):
        if self.stopping:
            self.set_state(self.current_index, STOPPED)
            self.status.set("Detenido.")
        elif code not in (0, 1):
            self.status.set(f"El proceso terminó con un error (código {code}). Mira el registro.")
        self.process = None
        self.current_log = None
        self.set_running(False)

    def add_log(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def on_close(self):
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno("UltraSinger", "Hay canciones procesándose. ¿Parar y salir?"):
                return
            self.stop()
        save_settings(self.read_settings())
        self.root.destroy()


def main():
    root = Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
