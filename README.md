# 🎙 Morse Code Audio Decoder

A Python desktop application that decodes Morse code from audio files using dual signal processing engines. Features a clean dark-themed GUI with drag-and-drop support, auto tone-frequency detection, and export to Word document.

---

## ✨ Features

- **Drag & Drop** — drop an audio file directly onto the app window, or use the **Browse** button to locate a file manually
- **Dual Decoder Engines** — runs both a custom signal-processing decoder and the `morse-audio-decoder` library decoder in parallel, then picks the best result
- **Auto Frequency Detection** — automatically detects the Morse carrier tone frequency (300–4000 Hz range) without any manual configuration
- **Dots & Dashes Output** — displays the raw Morse code symbols (e.g. `... --- ...`) alongside the decoded plain text
- **Export to Word** — save results as a formatted `.docx` file
- **Dark Theme UI** — clean Catppuccin-inspired dark interface built with Tkinter

---

## 🎵 Supported Audio Formats

| Format | Extension |
|--------|-----------|
| WAV | `.wav` |
| MP3 | `.mp3` |
| OGG Vorbis | `.ogg` |
| FLAC | `.flac` |
| AAC | `.aac` |
| M4A | `.m4a` |
| AIFF | `.aiff` |
| MOV | `.mov` |
| MP4 | `.mp4` |

> **Note:** Non-WAV formats require `pydub` and [FFmpeg](https://ffmpeg.org/download.html) to be installed on your system.

---

## 🔧 How It Works

### Signal Processing Decoder
1. Loads the audio and converts it to mono 22 050 Hz WAV
2. **Auto-detects the carrier tone** by averaging FFT windows across the file to find the dominant frequency peak between 300–4000 Hz
3. Applies a bandpass filter (±250 Hz around the peak) and computes short-time energy with a 5 ms hop
4. Binarises the energy envelope (ON/OFF) using Otsu-like thresholding
5. Measures dot/dash durations and auto-classifies symbols, characters, and word gaps
6. Translates the symbol sequence to plain text

### Library Decoder
Uses the `morse-audio-decoder` Python library as a second independent pass. Both results are compared and the cleanest output is returned. If both produce valid text, both are shown.

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/morse-audio-decoder.git
cd morse-audio-decoder
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg (for non-WAV audio)
- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to your PATH
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

### 4. (Optional) Enable native drag-and-drop
```bash
pip install tkinterdnd2
```
Without this, clicking the drop zone opens the file browser instead.

### 5. Run the app
```bash
python morse_decoder.py
```

> **Note:** On first run, missing packages are auto-installed via `pip`.

---

## 📋 Requirements

See [`requirements.txt`](requirements.txt) for the full list. Core dependencies:

- `pydub` — audio format conversion
- `numpy` — signal processing
- `python-docx` — Word document export
- `morse-audio-decoder` — library decoder engine
- `tkinterdnd2` *(optional)* — native drag-and-drop support

Python **3.8+** recommended.

---

## 🖥 Usage

1. Launch the app with `python morse_decoder.py`
2. **Load a file** — drag and drop an audio file onto the drop zone, or click **Browse…**
3. Click **Decode Audio** — the app analyses the file and displays:
   - The Morse code in dots and dashes (e.g. `... --- ...`)
   - The decoded plain text (e.g. `SOS`)
4. Optionally click **Save as .docx** to export the results
5. Click **Clear** to reset and decode another file

---

## 📸 Screenshot

> *Coming soon*

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
