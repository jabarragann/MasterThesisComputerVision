from pathlib import Path

if __name__ == "__main__":

    root_path = Path(r"C:\Users\asus\OneDrive - purdue.edu\ThesisDataset")

    for f in root_path.rglob("*_color.avi"):
        print(f.parent)