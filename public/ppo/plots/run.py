"""Remove epoch_001.png through epoch_009.png from all subfolders."""
import os

PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))
EPOCHS_TO_REMOVE = range(1, 10)  # 1..9

removed = 0
for name in os.listdir(PLOTS_DIR):
    subdir = os.path.join(PLOTS_DIR, name)
    if not os.path.isdir(subdir):
        continue
    for i in EPOCHS_TO_REMOVE:
        path = os.path.join(subdir, f"epoch_{i:03d}.png")
        if os.path.isfile(path):
            os.remove(path)
            removed += 1
            print(f"Removed: {path}")

print(f"Done. Removed {removed} files.")
