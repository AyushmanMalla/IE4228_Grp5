#!/usr/bin/env python3
"""Entry point for the live, hardware-accelerated Classical Face Recognition system.

Usage:
    python scripts/run_live.py [--gallery-dir data/team-photos] [--camera 0]
"""

import argparse
from pathlib import Path

from PySide6.QtWidgets import QApplication

from facerec_classical.gui_pyside import MainWindow


def main() -> None:
    parser = argparse.ArgumentParser(description="Live PySide6 Classical Face Recognition Demo")
    parser.add_argument(
        "--gallery-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "team-photos",
        help="Path to the training gallery database",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    args = parser.parse_args()

    # The Qt Application
    app = QApplication([])
    window = MainWindow(gallery_dir=args.gallery_dir, camera_index=args.camera)
    window.show()
    
    print("\n--- Classical Face Recognition System Started ---")
    print(f"Gallery directory: {args.gallery_dir.resolve()}")
    print("Press Ctrl+C in the terminal or close the window to exit.")
    
    app.exec()


if __name__ == "__main__":
    main()
