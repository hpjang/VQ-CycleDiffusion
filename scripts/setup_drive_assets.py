"""Link the separately distributed Drive assets into this checkout."""

import argparse
from pathlib import Path


DATA_DIRS = (
    "VCTK_2F2M",
    "VCTK_2F2M_train",
    "VCTK_2F2M_valid",
    "VCTK_2F2M_test",
)


def link_asset(source, destination):
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return False
        raise FileExistsError(f"Different link already exists: {destination}")
    if destination.exists():
        raise FileExistsError(f"Path already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    return True


def setup(asset_root, repo_root):
    asset_root = Path(asset_root).expanduser().resolve()
    repo_root = Path(repo_root).expanduser().resolve()
    if not (repo_root / "params.py").is_file():
        raise ValueError(f"Not a VQ-CycleDiffusion checkout: {repo_root}")
    if asset_root == repo_root or repo_root in asset_root.parents:
        raise ValueError("Download assets outside the Git checkout")

    assets = {
        "checkpts": asset_root / "checkpts",
        "codebooks": asset_root / "log" / "codebook_stock_255_exclude",
        "vocoder": asset_root / "hifi-gan" / "generator_universal.pth",
        "backbone": asset_root / "vc_255.pt",
    }
    assets.update({name: asset_root / name for name in DATA_DIRS})
    missing = [str(path) for path in assets.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing downloaded assets:\n" + "\n".join(missing))
    if not (assets["checkpts"] / "spk_encoder" / "pretrained.pt").is_file():
        raise FileNotFoundError(assets["checkpts"] / "spk_encoder" / "pretrained.pt")
    for name in DATA_DIRS:
        if not (assets[name] / "wavs").is_dir():
            raise FileNotFoundError(assets[name] / "wavs")
    codebooks = sorted(assets["codebooks"].rglob("*.pt"))
    if not codebooks:
        raise FileNotFoundError(f"No codebooks found in {assets['codebooks']}")
    codebook_dir = repo_root / "log" / "codebook_stock_255_exclude"
    if codebook_dir.is_symlink():
        raise FileExistsError(
            f"Replace the existing directory link before setup: {codebook_dir}"
        )

    links = [(assets[name], repo_root / name) for name in DATA_DIRS]
    links += [
        (assets["checkpts"], repo_root / "checkpts"),
        (assets["backbone"], repo_root / "log" / "log_Gunhee" / "vc_255.pt"),
        (assets["vocoder"], repo_root / "hifi-gan" / "generator_universal.pth"),
    ]
    links += [
        (
            codebook,
            codebook_dir / codebook.relative_to(assets["codebooks"]),
        )
        for codebook in codebooks
    ]
    for source, destination in links:
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise FileExistsError(f"Different link already exists: {destination}")
        elif destination.exists():
            raise FileExistsError(f"Path already exists: {destination}")
    for source, destination in links:
        action = "linked" if link_asset(source, destination) else "already linked"
        print(f"{action}: {destination} -> {source}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    setup(args.asset_root, args.repo_root)


if __name__ == "__main__":
    main()
