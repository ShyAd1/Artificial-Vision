#!/usr/bin/env python3
"""Aumentar_dataset.py

Lee las imágenes en la carpeta `FIGURAS` (por defecto en el mismo directorio
del script) y genera aumentos: rotaciones 90/180/270 y reflejos horizontal
y vertical. Guarda las imágenes aumentadas en la misma carpeta con sufijos
indicativos para evitar procesarlas de nuevo.

Uso:
    python Aumentar_dataset.py [--dir PATH] [--overwrite]

"""
from __future__ import annotations

import argparse
import os
from glob import glob
from typing import List

try:
    from PIL import Image
except Exception:  # pragma: no cover - user will see install instruction
    raise SystemExit("Pillow no está instalado. Instálalo con: pip install pillow")


SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
AUG_SUFFIXES = ["_rot90", "_rot180", "_rot270", "_flipH", "_flipV"]


def is_augmented(filename: str) -> bool:
    """Devuelve True si el nombre del archivo ya contiene un sufijo de aumento."""
    base = os.path.splitext(os.path.basename(filename))[0]
    return any(base.endswith(s) for s in AUG_SUFFIXES)


def gather_images(folder: str) -> List[str]:
    files: List[str] = []
    for ext in SUPPORTED_EXTS:
        files.extend(glob(os.path.join(folder, f"*{ext}"), recursive=False))
        files.extend(glob(os.path.join(folder, f"*{ext.upper()}"), recursive=False))
    return sorted(files)


def augment_image(path: str, overwrite: bool = False) -> int:
    """Genera y guarda las aumentaciones para una imagen.

    Retorna el número de imágenes nuevas guardadas (0-5).
    """
    saved = 0
    base_name = os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1]

    if is_augmented(path):
        print(f"Omitiendo (ya aumentado): {path}")
        return 0

    try:
        with Image.open(path) as img:
            img.load()

            # Rotaciones
            ops = [
                (img.rotate(90, expand=True), "_rot90"),
                (img.rotate(180, expand=True), "_rot180"),
                (img.rotate(270, expand=True), "_rot270"),
                (img.transpose(Image.FLIP_LEFT_RIGHT), "_flipH"),
                (img.transpose(Image.FLIP_TOP_BOTTOM), "_flipV"),
            ]

            for out_img, suffix in ops:
                out_name = f"{base_name}{suffix}{ext}"
                out_path = os.path.join(os.path.dirname(path), out_name)
                if os.path.exists(out_path) and not overwrite:
                    # No sobrescribimos por defecto
                    print(f"Ya existe, omitiendo: {out_name}")
                    continue
                # Intentar preservar el modo/compresión: si es PNG/JPEG usamos ext
                try:
                    out_img.save(out_path)
                    saved += 1
                except Exception as e:
                    print(f"Error guardando {out_path}: {e}")
    except Exception as e:
        print(f"Error abriendo {path}: {e}")

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Aumentar dataset en carpeta FIGURAS")
    parser.add_argument(
        "--dir",
        "-d",
        default=None,
        help="Carpeta que contiene las imágenes (por defecto 'FIGURAS' al lado del script)",
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="Sobrescribir archivos aumentados si ya existen",
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = args.dir if args.dir else os.path.join(script_dir, "FIGURAS")

    if not os.path.isdir(target_dir):
        raise SystemExit(f"Carpeta no encontrada: {target_dir}")

    images = gather_images(target_dir)
    if not images:
        print(f"No se encontraron imágenes en: {target_dir}")
        return

    total_saved = 0
    total_candidates = 0
    for img_path in images:
        if is_augmented(img_path):
            continue
        total_candidates += 1
        saved = augment_image(img_path, overwrite=args.overwrite)
        total_saved += saved

    print(
        f"Procesadas: {total_candidates} archivos fuente. Nuevas imágenes guardadas: {total_saved}"
    )


if __name__ == "__main__":
    main()
