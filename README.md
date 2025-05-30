# Massive Wavetable Injector 🎛️  
**By Dion Timmer**

## Notice

⚠️ This tool is **not affiliated with Native Instruments** in any way. Massive and all related assets are trademarks of **Native Instruments GmbH**. This is strictly a fan-made utility intended for educational and creative experimentation.

---

## Features
- 🔍 View and sort the internal wavetables by type, label, or sample count
- 💾 Export the full contents of `TABLES.DAT` to WAVs
- 🔄 Replace individual wavetables with your own
- 🛠️ Automatically converts non-PCM and float WAVs to 16-bit mono
- 🧼 Restore any individual wavetable or all at once from backup
- 🗃️ Keeps a backup of `TABLES_ORIGINAL.DAT` for safety

---

## Requirements
- ✅ **Windows** only
- 🧰 Python 3.9+ (with `PySide6`, `numpy`) if building from source
- 📂 Admin access required (Massive's system files are usually in a protected directory)
- 💽 Must have **Massive** installed — this tool looks for `TABLES.DAT` under:
  ```
  C:\Program Files (x86)\Common Files\Native Instruments\Massive
  ```

---

## Input File Guidelines
When replacing a wavetable, it is **strongly recommended** that your input WAVs follow these specs:

- 📈 **Same sample size** as the slot you're targeting (you’ll be prompted if not)
- 🎚️ **Mono**
- 🎧 **16-bit**
- ⏱️ **44.1kHz sample rate**

If your WAV doesn't match the slot size, the tool can:
- Loop or trim it to fit
- Stretch it via linear resampling (changes pitch)

---

## How to Use

### 🟢 Option 1: Download the Binary
You can download a ready-to-run Windows executable from the Releases section.  
Just unzip it, run it as Administrator, and you’re good to go.

### 🔧 Option 2: Run from Source
1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the tool:
   ```bash
   python massive_injector.py
   ```

> You'll need to run as **Administrator** if `TABLES.DAT` is in Program Files.

---

## Creating a Standalone EXE (Optional)
You can bundle everything into a single Windows executable using **PyInstaller**:

1. Install PyInstaller:
   ```bash
   pip install pyinstaller
   ```

2. Build with:
   ```bash
   pyinstaller --noconsole --onefile --icon=dtico3.ico main.py
   ```

This will produce a standalone `main.exe` in the `dist` folder.

---

## Disclaimer
This tool modifies internal files of Massive. While it keeps a backup (`TABLES_ORIGINAL.DAT`) and has been tested to be safe, **use it at your own risk**. Always keep your own backups of important plugin data.
