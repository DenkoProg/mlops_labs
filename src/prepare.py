from pathlib import Path
from typing import Annotated

import numpy as np
from PIL import Image
from tqdm import tqdm
import typer


app = typer.Typer()

CLASSES = ["NORMAL", "PNEUMONIA"]


def load_images(directory: Path, img_size: int) -> tuple[np.ndarray, np.ndarray]:
    images, labels = [], []
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = directory / class_name
        if not class_dir.exists():
            continue
        paths = list(class_dir.glob("*"))
        for p in tqdm(paths, desc=f"Loading {class_name}"):
            try:
                img = Image.open(p).convert("L").resize((img_size, img_size))
                images.append(np.array(img).flatten())
                labels.append(label_idx)
            except Exception:
                continue
    return np.array(images), np.array(labels)


@app.command()
def main(
    data_dir: Annotated[Path, typer.Argument(help="Path to raw dataset")],
    output_dir: Annotated[Path, typer.Argument(help="Path to output prepared data")],
    img_size: Annotated[int, typer.Option(help="Resize images to img_size x img_size")] = 64,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Preparing training data from {data_dir / 'train'}...")
    X_train, y_train = load_images(data_dir / "train", img_size)

    typer.echo(f"Preparing test data from {data_dir / 'test'}...")
    X_test, y_test = load_images(data_dir / "test", img_size)

    # Normalize
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    typer.echo(f"Train: {X_train.shape}, Test: {X_test.shape}")
    typer.echo(f"Train distribution: NORMAL={np.sum(y_train == 0)}, PNEUMONIA={np.sum(y_train == 1)}")

    # Save to numpy arrays
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)

    typer.echo(f"Prepared data saved to {output_dir}")


if __name__ == "__main__":
    app()
