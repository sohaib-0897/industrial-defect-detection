"""Download the official archive atomically and inspect paths before extraction."""

import argparse
import subprocess
import zipfile
from pathlib import Path

URL = "https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip"


def download(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / "KolektorSDD2.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".zip.part")
        subprocess.run(
            [
                "curl.exe" if __import__("os").name == "nt" else "curl",
                "--noproxy",
                "*",
                "-L",
                "--fail",
                "--retry",
                "3",
                URL,
                "-o",
                str(temporary),
            ],
            check=True,
        )
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as stream:
        for member in stream.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive path")
        print(f"Archive members: {len(stream.infolist())}; first paths: {stream.namelist()[:12]}")
        stream.extractall(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/KolektorSDD2"))
    download(parser.parse_args().output)
