from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import traceback
from collections import OrderedDict
from typing import Dict

from PySide6.QtGui import QIcon
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QCheckBox,
)

DEFAULT_DIR = r"C:\Program Files (x86)\Common Files\Native Instruments\Massive"


# ════════════════════════════════════════════════════════════════════════════
# Helper to locate resources in dev & PyInstaller
# ════════════════════════════════════════════════════════════════════════════


def resource_path(relative: str) -> str:
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(__file__))
    return os.path.join(base_path, relative)


# ════════════════════════════════════════════════════════════════════════════
# Load label map
# ════════════════════════════════════════════════════════════════════════════
try:
    with open(resource_path("labels.json"), "r", encoding="utf-8") as fh:
        LABEL_MAP: Dict[str, str] = json.load(fh)
except Exception as exc:
    LABEL_MAP = {}
    print(f"[WARN] labels.json not loaded: {exc}")


# ════════════════════════════════════════════════════════════════════════════
# NIMD parser (unchanged)
# ════════════════════════════════════════════════════════════════════════════
class _Entry:
    is_list = False  # type: ignore


class _EntryList:
    is_list = True  # type: ignore


def _parse_root(fp):
    if fp.read(4) != b"NIMD":
        raise ValueError("Not a NIMD file")
    fp.read(8)
    return _parse_list(b"", fp)


def _parse_entry(fp):
    is_list, _, nlen = struct.unpack("<bll", fp.read(9))
    name = fp.read(nlen)
    if is_list:
        return _parse_list(name, fp)
    off, size = struct.unpack("<ll", fp.read(8))
    n = _Entry()
    n.name, n.data_offset, n.data_size = name.decode(), off, size
    return n


def _parse_list(name, fp):
    _, cnt = struct.unpack("<ll", fp.read(8))
    node = _EntryList()
    node.name = name.decode() if name else ""
    node.entries = [_parse_entry(fp) for _ in range(cnt)]
    return node


def _walk(n, base=""):
    if n.is_list:
        base2 = os.path.join(base, n.name) if n.name else base
        for s in n.entries:
            yield from _walk(s, base2)
    else:
        yield (os.path.join(base, n.name) if base else n.name).replace(
            "\\", "/"
        ), n.data_offset, n.data_size


# ════════════════════════════════════════════════════════════════════════════
# Build wavetable index
# ════════════════════════════════════════════════════════════════════════════


def build_index(tables: str, backup: str) -> "OrderedDict[str, Dict]":
    with open(tables, "rb") as cur, open(backup, "rb") as bak:
        root = _parse_root(cur)
        idx: "OrderedDict[str, Dict]" = OrderedDict()
        for name, abs_off, chunk in _walk(root):
            if not name.lower().endswith(".wav"):
                continue
            cur.seek(abs_off)
            hdr = cur.read(min(chunk, 256))
            p = hdr.find(b"data")
            if p == -1:
                continue
            pcm_size = struct.unpack_from("<I", hdr, p + 4)[0]
            pcm_off = abs_off + p + 8
            cur.seek(pcm_off)
            data_cur = cur.read(pcm_size)
            bak.seek(pcm_off)
            data_bak = bak.read(pcm_size)
            idx[name] = {
                "byte_offset": pcm_off,
                "pcm_size": pcm_size,
                "samples": pcm_size // 2,
                "overridden": data_cur != data_bak,
            }
    return idx


# ════════════════════════════════════════════════════════════════════════════
# GUI
# ════════════════════════════════════════════════════════════════════════════
class WavetableInjector(QWidget):
    def __init__(self, initial_tables: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Massive Injector by Dion Timmer")
        icon_path = resource_path("dtico3.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            print(f"[WARN] Icon not found: {icon_path}")
        self.resize(750, 900)
        self.setMinimumSize(750, 900)
        self.setMaximumSize(750, 900)
        self.tables_path: str | None = None
        self.backup_path: str | None = None
        self.index: "OrderedDict[str, Dict]" = OrderedDict()
        self.sort_key: str = "None"
        self.sort_desc: bool = False

        root_lay = QVBoxLayout(self)

        # ─── Top path chooser ──────────────────────────────────────────────
        top = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select TABLES.DAT …")
        self.path_edit.setReadOnly(True)
        top.addWidget(self.path_edit)
        browse = QPushButton("Browse")
        browse.clicked.connect(self.choose_tables)
        top.addWidget(browse)
        root_lay.addLayout(top)

        # ─── Sort bar + actions ────────────────────────────────────────────
        sort_bar = QHBoxLayout()
        sort_bar.addWidget(QLabel("Sort by:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["None", "Type", "Label", "Samples"])
        self.sort_combo.currentTextChanged.connect(self.on_sort_changed)
        sort_bar.addWidget(self.sort_combo)

        self.desc_check = QCheckBox("Descending")
        self.desc_check.toggled.connect(self.on_desc_toggled)
        sort_bar.addWidget(self.desc_check)

        # ─── Action buttons ───────────────────────────────────────────────
        self.export_btn = QPushButton("Export TABLES.DAT…")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_tables)
        sort_bar.addWidget(self.export_btn)

        self.restore_all_btn = QPushButton("Restore All")
        self.restore_all_btn.setEnabled(False)
        self.restore_all_btn.clicked.connect(self.restore_all)
        sort_bar.addWidget(self.restore_all_btn)

        sort_bar.addStretch()
        root_lay.addLayout(sort_bar)

        # ─── Table ────────────────────────────────────────────────────────
        self.table = QTableWidget()
        self.table.setSelectionMode(QTableWidget.NoSelection)
        root_lay.addWidget(self.table)

        if initial_tables:
            self.load_tables(initial_tables)

    # ─── DAT selection ───────────────────────────────────────────────
    def choose_tables(self):
        start_dir = DEFAULT_DIR if os.path.isdir(DEFAULT_DIR) else ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select TABLES.DAT",
            start_dir,
            filter="DAT files (*.DAT)",
        )
        if path:
            self.load_tables(path)

    def load_tables(self, path: str):
        self.tables_path = path
        self.path_edit.setText(path)
        self.backup_path = os.path.join(os.path.dirname(path), "TABLES_ORIGINAL.DAT")
        if not os.path.exists(self.backup_path):
            shutil.copy(path, self.backup_path)
            QMessageBox.information(self, "Backup created", self.backup_path)
        try:
            self.index = build_index(self.tables_path, self.backup_path)
            self.populate_table()
            self.export_btn.setEnabled(True)
            self.restore_all_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ─── Sort handling ─────────────────────────────────────────────────
    def on_sort_changed(self, text: str):
        self.sort_key = text
        self.populate_table()

    def on_desc_toggled(self, state: bool):
        self.sort_desc = state
        self.populate_table()

    # ─── Export TABLES.DAT ──────────────────────────────────────────────
    def export_tables(self):
        if not self.tables_path:
            QMessageBox.warning(self, "No file loaded", "Please open TABLES.DAT first.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not out_dir:
            return

        try:
            with open(self.tables_path, "rb") as fp:
                root = _parse_root(fp)  # reuse the existing NIMD parser
                self._export_entry(root, fp, out_dir)
            QMessageBox.information(
                self, "Export finished", f"All contents written to:\n{out_dir}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    # recursive helper (private)
    def _export_entry(self, entry, fp, base):
        """
        Walk the parsed tree and write each node to disk,
        preserving the sub-folder hierarchy.
        """
        path = os.path.join(base, entry.name) if entry.name else base

        if entry.is_list:
            os.makedirs(path, exist_ok=True)
            for sub in entry.entries:
                self._export_entry(sub, fp, path)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fp.seek(entry.data_offset)
            with open(path, "wb") as out:
                out.write(fp.read(entry.data_size))

    # ─── Restore all slots ────────────────────────────────────────────
    def restore_all(self):
        if not self.tables_path:
            return
        confirm = QMessageBox.question(
            self,
            "Restore All",
            "Are you sure you want to restore every wavetable to its original state?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        try:
            with open(self.backup_path, "rb") as bak, open(
                self.tables_path, "r+b"
            ) as cur:
                for info in self.index.values():
                    bak.seek(info["byte_offset"])
                    original = bak.read(info["pcm_size"])
                    cur.seek(info["byte_offset"])
                    cur.write(original)
                    info["overridden"] = False
            self.populate_table()
            QMessageBox.information(self, "Restored", "All wavetables restored.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ─── table fill ──────────────────────────────────────────────────
    def populate_table(self):
        headers = [
            "Type",
            "Label",
            "Filename",
            "Samples",
            "Overridden",
            "Replace…",
            "Restore",
        ]

        # Block repaints & signals for snappier rebuilds ----------------
        self.table.setUpdatesEnabled(False)
        try:
            self.table.clear()
            self.table.setColumnCount(len(headers))
            self.table.setHorizontalHeaderLabels(headers)
            self.table.verticalHeader().setVisible(False)

            # prepare sorted items list
            items = list(self.index.items())
            if self.sort_key == "Type":
                items.sort(key=lambda kv: kv[0].split("/")[0] if "/" in kv[0] else "")
            elif self.sort_key == "Label":
                items.sort(key=lambda kv: LABEL_MAP.get(os.path.basename(kv[0]), ""))
            elif self.sort_key == "Samples":
                items.sort(key=lambda kv: kv[1]["samples"])
            # None: keep original order

            if self.sort_desc:
                items.reverse()

            self.table.setRowCount(len(items))

            for row, (fname, info) in enumerate(items):
                base = os.path.basename(fname)
                label = LABEL_MAP.get(base, "")
                wtype = fname.split("/")[0] if "/" in fname else ""

                cols = [
                    QTableWidgetItem(wtype),
                    QTableWidgetItem(label),
                    QTableWidgetItem(fname),
                    QTableWidgetItem(str(info["samples"])),
                ]
                for c in cols:
                    c.setFlags(Qt.ItemIsEnabled)
                    c.setTextAlignment(Qt.AlignCenter)

                for col_idx, item in enumerate(cols):
                    self.table.setItem(row, col_idx, item)

                self.table.setItem(row, 4, self._tick(info["overridden"]))

                rep = QPushButton("Replace")
                rep.setFocusPolicy(Qt.NoFocus)
                rep.clicked.connect(lambda _, f=fname, r=row: self.replace_wt(f, r))
                res = QPushButton("Restore")
                res.setFocusPolicy(Qt.NoFocus)
                res.clicked.connect(lambda _, f=fname, r=row: self.restore_wt(f, r))
                self.table.setCellWidget(row, 5, rep)
                self.table.setCellWidget(row, 6, res)

            self.table.resizeColumnsToContents()
        finally:
            self.table.setUpdatesEnabled(True)

    # ─── tick helper ────────────────────────────────────────────────
    @staticmethod
    def _tick(state: bool) -> QTableWidgetItem:
        item = QTableWidgetItem("✅" if state else "❌")
        item.setFlags(Qt.ItemIsEnabled)
        item.setTextAlignment(Qt.AlignCenter)
        return item

    # ─── WAV utilities ──────────────────────────────────────────────
    @staticmethod
    def _stereo_to_mono(pcm: bytes) -> bytes:
        """Average L+R 16-bit little-endian samples to mono."""
        samples = np.frombuffer(pcm, dtype="<i2").reshape(-1, 2).astype(np.int32)
        mono = ((samples[:, 0] + samples[:, 1]) // 2).astype("<i2")
        return mono.tobytes()

    @staticmethod
    def pcm_from_wav(path: str) -> bytes:
        """
        Return 16-bit little-endian mono PCM from a WAV file.
        Supports:
            • PCM   : 16/24/32-bit
            • Float : 32/64-bit (IEEE float)
        """
        with open(path, "rb") as fh:
            data = fh.read()

        # ---- locate 'fmt ' chunk -------------------------------------------------
        p_fmt = data.find(b"fmt ")
        if p_fmt == -1:
            raise ValueError("Malformed WAV: no 'fmt ' chunk")
        fmt_size = struct.unpack_from("<I", data, p_fmt + 4)[0]
        audio_fmt, n_channels, sample_rate, _, _, bits = struct.unpack_from(
            "<HHIIHH", data, p_fmt + 8
        )

        # ---- locate 'data' chunk -------------------------------------------------
        p_data = data.find(b"data", p_fmt + 8 + fmt_size)
        if p_data == -1:
            raise ValueError("Malformed WAV: no 'data' chunk")
        pcm_size = struct.unpack_from("<I", data, p_data + 4)[0]
        pcm = data[p_data + 8 : p_data + 8 + pcm_size]

        # helper to scale/clamp → int16
        def to_int16(arr_float: np.ndarray) -> np.ndarray:
            arr = np.clip(arr_float, -1.0, 1.0)  # avoid wrap
            return (arr * 32767.0).astype("<i2")

        if audio_fmt == 1:  # PCM integers
            if bits == 16:
                pcm_i16 = np.frombuffer(pcm, dtype="<i2")
            elif bits == 24:
                # 24-bit packed little endian → int32, then shift
                raw = np.frombuffer(pcm, dtype=np.uint8).reshape(-1, 3)
                pcm_i32 = (
                    raw[:, 0].astype(np.int32)
                    | (raw[:, 1].astype(np.int32) << 8)
                    | (raw[:, 2].astype(np.int32) << 16)
                )
                pcm_i32 = pcm_i32.astype(np.int32, copy=False)
                pcm_i32 = np.where(pcm_i32 & 0x800000, pcm_i32 - 0x1000000, pcm_i32)
                pcm_i16 = (pcm_i32 >> 8).astype("<i2")  # down-shift to 16-bit
            elif bits == 32:
                pcm_i32 = np.frombuffer(pcm, dtype="<i4")
                pcm_i16 = (pcm_i32 >> 16).astype("<i2")
            else:
                raise ValueError(f"Unsupported PCM bit-depth: {bits}")
        elif audio_fmt == 3:  # IEEE float
            if bits == 32:
                pcm_f32 = np.frombuffer(pcm, dtype="<f4")
                pcm_i16 = to_int16(pcm_f32)
            elif bits == 64:
                pcm_f64 = np.frombuffer(pcm, dtype="<f8")
                pcm_i16 = to_int16(pcm_f64.astype(np.float32))
            else:
                raise ValueError(f"Unsupported float bit-depth: {bits}")
        else:
            raise ValueError(f"Unsupported WAV audio format: {audio_fmt}")

        # ---- stereo/ multi → mono (average) --------------------------------------
        if n_channels > 1:
            pcm_i16 = pcm_i16.reshape(-1, n_channels).mean(axis=1).astype("<i2")

        return pcm_i16.tobytes()

    # ─── size-mismatch helpers ──────────────────────────────────────
    def _loop_or_cut(self, pcm: bytes, needed_bytes: int) -> bytes:
        """Trim if too long, loop-repeat if too short."""
        if len(pcm) >= needed_bytes:
            return pcm[:needed_bytes]
        reps, rem = divmod(needed_bytes, len(pcm))
        return pcm * reps + pcm[:rem]

    def _stretch(self, pcm: bytes, needed_bytes: int) -> bytes:
        """Resample (linear) to exactly needed_bytes (16-bit)."""
        src = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        dst_samples = needed_bytes // 2
        dst_idx = np.linspace(0, len(src) - 1, dst_samples, endpoint=True)
        dst = np.interp(dst_idx, np.arange(len(src)), src).astype(np.int16)
        return dst.tobytes()

    def _handle_mismatch(self, pcm: bytes, needed_bytes: int) -> bytes | None:
        new_samples = len(pcm) // 2
        want_samples = needed_bytes // 2
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Size mismatch")
        msg.setText(
            f"The selected WAV has {new_samples} samples, but the slot "
            f"requires {want_samples}.\n\nChoose how to proceed:"
        )
        loop_btn = msg.addButton("Loop / Cut", QMessageBox.AcceptRole)
        stretch_btn = msg.addButton("Pitch / Stretch", QMessageBox.AcceptRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
        msg.exec()

        if msg.clickedButton() is loop_btn:
            return self._loop_or_cut(pcm, needed_bytes)
        if msg.clickedButton() is stretch_btn:
            try:
                return self._stretch(pcm, needed_bytes)
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Stretch failed: {exc}")
                return None
        return None  # cancel

    # ─── replace / restore ─────────────────────────────────────────
    def replace_wt(self, fname: str, row: int):
        wav_path, _ = QFileDialog.getOpenFileName(
            self, "Select WAV", filter="WAV files (*.wav)"
        )
        if not wav_path:
            return
        try:
            pcm = self.pcm_from_wav(wav_path)
            info = self.index[fname]
            if len(pcm) != info["pcm_size"]:
                pcm = self._handle_mismatch(pcm, info["pcm_size"])
                if pcm is None:
                    return
            # write new PCM block
            with open(self.tables_path, "r+b") as fp:
                fp.seek(info["byte_offset"])
                fp.write(pcm)
            info["overridden"] = True
            self.table.setItem(row, 4, self._tick(True))
            QMessageBox.information(self, "Success", "Wavetable replaced!")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def restore_wt(self, fname: str, row: int):
        info = self.index[fname]
        try:
            with open(self.backup_path, "rb") as bak:
                bak.seek(info["byte_offset"])
                original = bak.read(info["pcm_size"])
            with open(self.tables_path, "r+b") as cur:
                cur.seek(info["byte_offset"])
                cur.write(original)
            info["overridden"] = False
            self.table.setItem(row, 4, self._tick(False))
            QMessageBox.information(self, "Restored", "Original wavetable restored.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    initial = sys.argv[1] if len(sys.argv) == 2 else None
    try:
        app = QApplication(sys.argv)
        win = WavetableInjector(initial)
        win.show()
        sys.exit(app.exec())
    except Exception as exc:
        traceback.print_exc()
        print("\nAn error occurred:", exc)
        input("Press ENTER to exit…")
