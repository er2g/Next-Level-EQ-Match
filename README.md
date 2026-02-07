# EQ Matcher 🎚️

A professional audio equalization matching tool built with **Tauri**, **Rust**, and **React**.

EQ Matcher analyzes the frequency spectrum of a reference audio track and generates an EQ curve to match your input audio to that reference. It uses intelligent psychoacoustic weighting and smoothing to create natural-sounding corrections.

![EQ Matcher Screenshot](https://via.placeholder.com/800x450?text=EQ+Matcher+Screenshot)

## ✨ Features

*   **Intelligent Matching:** Matches EQ curves using psychoacoustic principles (ISO 226:2003).
*   **Professional Analysis:** High-resolution FFT analysis with Blackman-Harris windowing.
*   **Visual Feedback:** Interactive frequency response charts and real-time visualization.
*   **Customizable Control:** Adjust match intensity, smoothing, and maximum correction limits.
*   **Multi-Format Export:** Export settings as:
    *   Reaper FX Chain (`.RfxChain`)
    *   JSON Data
    *   Human-readable Text
*   **Format Support:** Supports MP3, WAV, FLAC, OGG, AAC, and more.

## 🛠️ Tech Stack

*   **Frontend:** React, TypeScript, Chart.js, Vite
*   **Backend:** Rust, Tauri
*   **Audio Processing:** Symphonia (decoding), RustFFT (analysis), Biquad (filtering)

## 🚀 Getting Started

### Prerequisites

*   [Node.js](https://nodejs.org/) (v16 or later)
*   [Rust](https://www.rust-lang.org/tools/install) (latest stable)
*   [Tauri Prerequisites](https://tauri.app/v1/guides/getting-started/prerequisites) (C++ Build Tools, WebView2 on Windows)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/eq-matcher.git
    cd eq-matcher
    ```

2.  **Install frontend dependencies:**
    ```bash
    npm install
    ```

3.  **Run in Development Mode:**
    ```bash
    npm run tauri dev
    ```

## ⚡ Performance Benchmark (Headless)

If you only want to benchmark the audio pipeline (no Tauri/GUI build), you can run the Rust benchmark binary with Tauri disabled:

```bash
cd src-tauri
mkdir -p ../bench
curl -L -o ../bench/sample.mp3 https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3
cargo run --release --no-default-features --bin bench -- ../bench/sample.mp3
```

To emulate the previous behavior (always resample to 48kHz), use:

```bash
cargo run --release --no-default-features --bin bench -- ../bench/sample.mp3 --preset legacy
```

### Building for Production

To create a standalone executable:

```bash
npm run tauri build
```

The executable will be located in `src-tauri/target/release/bundle/`.

## 📖 Usage

1.  **Upload Reference:** Drag & drop the audio file you want to sound like (e.g., a pro mix).
2.  **Upload Input:** Drag & drop your own audio recording.
3.  **Analyze & Match:** Click "Calculate EQ Match".
4.  **Fine-tune:** Use the control panel to adjust:
    *   **Intensity:** How strong the EQ match should be.
    *   **Smoothing:** How smooth the curve transitions are.
    *   **Max Correction:** Limit the maximum boost/cut in dB.
5.  **Export:** Click "Export Settings" and choose your preferred format (e.g., for Reaper).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

*   [Tauri Team](https://tauri.app/) for the amazing framework.
*   [Chart.js](https://www.chartjs.org/) for visualization.
*   Rust Audio community for crates like `symphonia` and `rustfft`.
