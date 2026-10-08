"""Concise terminal progress without saving diagnostic output."""

import ast
import os
import re
import subprocess
import sys


MANAGED_ENV = "HYRCAP_MANAGED_CONSOLE"
TOTAL_PREFIX = "HYRCAP_TRAINING_EPOCHS="


class Progress:
    def __init__(self):
        self.total_epochs = 0
        self.detector_epochs = 0
        self.completed_detector_epochs = 0
        self.last_detector_epoch = 0

    def render(self, line):
        line = line.strip()
        if line.startswith(TOTAL_PREFIX):
            try:
                self.total_epochs = int(line[len(TOTAL_PREFIX):])
            except ValueError:
                pass
            return None
        match = re.fullmatch(
            r"Detector epoch (\d+)/(\d+) batch (\d+)/(\d+) loss ([\d.eE+\-]+)",
            line,
        )
        if match:
            epoch, epochs, batch, batches, loss = match.groups()
            if int(epoch) < self.last_detector_epoch:
                self.completed_detector_epochs += self.detector_epochs
            self.last_detector_epoch = int(epoch)
            self.detector_epochs = int(epochs)
            total = self.total_epochs or self.detector_epochs
            current = self.completed_detector_epochs + int(epoch)
            return f"Training epoch {current}/{total} batch {batch}/{batches} loss {loss}"
        if line.startswith("Calibration {"):
            try:
                row = ast.literal_eval(line.removeprefix("Calibration "))
                epoch = self.completed_detector_epochs + self.detector_epochs + int(row["epoch"])
                loss = float(row["loss"])
            except (ValueError, SyntaxError, TypeError, KeyError):
                return None
            total = self.total_epochs or epoch
            return f"Training epoch {epoch}/{total} loss {loss:.5f}"
        match = re.fullmatch(r"Inference (\d+)/(\d+); ([\d.]+)s", line)
        if match:
            current, total, elapsed = match.groups()
            return f"Evaluating {current}/{total}; {elapsed}s"
        match = re.fullmatch(r"Detection mAP: ([\d.]+)%", line)
        if match:
            return f"mAP: {match.group(1)}%"
        if line.startswith(("Downloading dataset: ", "Dataset ready: ")):
            return line
        if line.startswith("Unified checkpoint: "):
            return line.replace("Unified checkpoint: ", "Checkpoint: ", 1)
        if line.startswith("Results: "):
            return line
        return None


def run_cli(script):
    """Forward progress and exit status; discard all other child output."""
    arguments = sys.argv[1:]
    environment = os.environ.copy()
    environment[MANAGED_ENV] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-B", "-u", str(script), *arguments]
    if "--help" in arguments or "-h" in arguments:
        return subprocess.call(command, env=environment)
    print("Running...", flush=True)
    progress = Progress()
    process = None
    try:
        process = subprocess.Popen(
            command, env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        with process.stdout:
            for line in process.stdout:
                visible = progress.render(line)
                if visible is not None:
                    print(visible, flush=True)
        code = process.wait()
    except KeyboardInterrupt:
        print("Run interrupted.", flush=True)
        return 130
    except Exception:
        print("Run failed.", flush=True)
        return 1
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    if code:
        print("Run failed.", flush=True)
        return code if code > 0 else 128 - code
    print("Completed.", flush=True)
    return 0
