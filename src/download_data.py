from pathlib import Path
import shutil

import kagglehub
import typer


def main(output_dir: Path = Path("data/raw")):
    if output_dir.exists():
        typer.echo(f"Dataset already exists at {output_dir}")
        return

    typer.echo("Downloading Chest X-Ray dataset from Kaggle...")
    cache_path = Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia"))
    shutil.copytree(cache_path, output_dir)
    typer.echo(f"Dataset copied to: {output_dir}")


if __name__ == "__main__":
    typer.run(main)
