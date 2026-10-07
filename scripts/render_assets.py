#!/usr/bin/env python3
"""Rasterize the app icon and redraw the cover art.

The icon SVG is the source for the PNG, ICO, and ICNS files. The cover is a
redraw of the attached dashboard image: same palette, title, table, suspicion
bars, review grid, and the tag-and-magnifier mark.
"""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
WEB = ROOT / "labelnoiseaudit" / "web"
FONT = Path("/usr/share/fonts/truetype/macos")

BG = (27, 36, 49)
CARD = (36, 49, 64)
BORDER = (51, 65, 85)
TEXT = (248, 250, 252)
MUTED = (148, 163, 184)
ORANGE = (245, 130, 32)
ORANGE_2 = (255, 138, 31)
GREEN = (61, 220, 132)
GREEN_BG = (20, 83, 45)
CHECK_BG = (58, 45, 36)
DEEP = (21, 28, 39)


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT / name
    if not path.is_file():
        path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size)


def rounded(draw: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def rasterize_svg(size: int) -> Image.Image:
    svg = ASSETS / "icon.svg"
    out = ASSETS / f".icon-{size}.png"
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            (
                "import cairosvg,sys;"
                f"cairosvg.svg2png(url={str(svg)!r}, write_to={str(out)!r}, "
                f"output_width={size}, output_height={size})"
            ),
        ]
    )
    image = Image.open(out).convert("RGBA")
    out.unlink(missing_ok=True)
    return image


def write_icns(path: Path, pngs: dict[str, bytes]) -> None:
    chunks = []
    for ostype, blob in pngs.items():
        chunks.append(ostype.encode("ascii") + struct.pack(">I", len(blob) + 8) + blob)
    body = b"".join(chunks)
    path.write_bytes(b"icns" + struct.pack(">I", len(body) + 8) + body)


def png_bytes(image: Image.Image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def draw_cat(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), (214, 196, 176))
    draw = ImageDraw.Draw(image)
    draw.polygon([(18, 28), (34, 8), (48, 30)], fill=(160, 130, 100))
    draw.polygon([(size - 48, 30), (size - 34, 8), (size - 18, 28)], fill=(160, 130, 100))
    draw.ellipse((16, 22, size - 16, size - 10), fill=(196, 166, 130))
    draw.ellipse((28, 40, 52, 62), fill=(40, 32, 28))
    draw.ellipse((size - 52, 40, size - 28, 62), fill=(40, 32, 28))
    draw.polygon([(size // 2, 58), (size // 2 - 8, 74), (size // 2 + 8, 74)], fill=(232, 170, 150))
    return image


def draw_dog(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), (186, 206, 170))
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 18, 36, 78), fill=(122, 78, 42))
    draw.ellipse((size - 36, 18, size - 8, 78), fill=(122, 78, 42))
    draw.ellipse((18, 20, size - 18, size - 12), fill=(176, 122, 70))
    draw.ellipse((30, 40, 50, 58), fill=(40, 28, 18))
    draw.ellipse((size - 50, 40, size - 30, 58), fill=(40, 28, 18))
    draw.ellipse((size // 2 - 14, 58, size // 2 + 14, 82), fill=(90, 56, 32))
    return image


def draw_car(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), (176, 196, 214))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 40, size - 8, 78), radius=10, fill=(48, 58, 72))
    draw.polygon([(24, 42), (40, 22), (size - 36, 22), (size - 18, 42)], fill=(70, 86, 104))
    draw.ellipse((18, 64, 42, 88), fill=(24, 24, 28))
    draw.ellipse((size - 42, 64, size - 18, 88), fill=(24, 24, 28))
    return image


def draw_bird(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), (198, 214, 230))
    draw = ImageDraw.Draw(image)
    draw.ellipse((16, 28, size - 10, size - 22), fill=(46, 74, 120))
    draw.polygon([(size - 28, 40), (size - 6, 32), (size - 22, 54)], fill=(232, 168, 60))
    draw.ellipse((28, 40, 46, 58), fill=(250, 250, 250))
    draw.ellipse((34, 46, 42, 54), fill=(20, 20, 20))
    return image


def draw_tree(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), (190, 214, 196))
    draw = ImageDraw.Draw(image)
    draw.rectangle((size // 2 - 6, 52, size // 2 + 6, size - 12), fill=(110, 74, 42))
    draw.ellipse((14, 16, size - 14, 70), fill=(46, 130, 72))
    return image


def draw_chair(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), (214, 206, 196))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((22, 16, size - 22, 48), radius=6, fill=(150, 96, 54))
    draw.rounded_rectangle((18, 46, size - 18, 66), radius=4, fill=(176, 122, 74))
    draw.rectangle((24, 64, 34, size - 12), fill=(120, 78, 42))
    draw.rectangle((size - 34, 64, size - 24, size - 12), fill=(120, 78, 42))
    return image


def draw_boat(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), (168, 198, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, size // 2 + 8, size, size), fill=(46, 110, 160))
    draw.polygon([(16, 58), (size - 16, 58), (size - 28, 78), (28, 78)], fill=(236, 236, 232))
    draw.polygon([(size // 2, 16), (size // 2 + 8, 58), (size // 2 - 22, 58)], fill=(220, 80, 60))
    return image


def draw_truck(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), (206, 196, 176))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 28, size - 34, 70), radius=4, fill=(230, 230, 226))
    draw.rounded_rectangle((size - 38, 40, size - 8, 70), radius=4, fill=(60, 78, 96))
    draw.ellipse((16, 62, 36, 82), fill=(30, 30, 30))
    draw.ellipse((size - 36, 62, size - 16, 82), fill=(30, 30, 30))
    return image


THUMBS = {
    "cat": draw_cat,
    "dog": draw_dog,
    "car": draw_car,
    "bird": draw_bird,
    "tree": draw_tree,
    "chair": draw_chair,
    "boat": draw_boat,
    "truck": draw_truck,
}


def paste_thumb(canvas: Image.Image, kind: str, box):
    size = box[2] - box[0]
    thumb = THUMBS[kind](160).resize((size, size), Image.Resampling.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=8, fill=255)
    canvas.paste(thumb, box[:2], mask)


def draw_cover() -> Image.Image:
    width, height = 1600, 900
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    title_font = font("Inter-Bold.ttf", 72)
    label_font = font("Inter-SemiBold.ttf", 18)
    body = font("Inter-Medium.ttf", 20)
    small = font("Inter-Regular.ttf", 16)
    tiny = font("Inter-Medium.ttf", 15)

    # Title
    label = "Label"
    noise = "Noise"
    audit = "Audit"
    label_w = draw.textlength(label, font=title_font)
    noise_w = draw.textlength(noise, font=title_font)
    audit_w = draw.textlength(audit, font=title_font)
    total = label_w + noise_w + audit_w
    x = (width - total) / 2
    y = 28
    draw.text((x, y), label, font=title_font, fill=TEXT)
    draw.text((x + label_w, y), noise, font=title_font, fill=ORANGE)
    draw.text((x + label_w + noise_w, y), audit, font=title_font, fill=TEXT)

    rounded(draw, (36, 124, 1036, 560), 22, CARD, BORDER, 2)
    rounded(draw, (1056, 124, 1564, 560), 22, CARD, BORDER, 2)
    rounded(draw, (36, 580, 1036, 864), 22, CARD, BORDER, 2)
    rounded(draw, (1228, 620, 1564, 864), 28, CARD, BORDER, 2)

    draw.text((60, 142), "DATA TABLE (SAMPLE)", font=label_font, fill=MUTED)
    headers = ["ID", "Image", "Label", "Predicted", "Confidence", "Status"]
    header_x = [60, 150, 250, 420, 620, 820]
    for text, hx in zip(headers, header_x, strict=True):
        draw.text((hx, 180), text, font=small, fill=MUTED)

    rows = [
        ("0047", "cat", "cat", "cat", "0.92", "Clean", False),
        ("0123", "cat", "dog", "cat", "0.31", "Check", True),
        ("0189", "car", "truck", "car", "0.28", "Check", True),
        ("0256", "bird", "pigeon", "crow", "0.19", "Check", True),
        ("0301", "tree", "tree", "tree", "0.88", "Clean", False),
        ("0364", "chair", "chair", "chair", "0.79", "Clean", False),
        ("0410", "boat", "boat", "boat", "0.91", "Clean", False),
    ]
    top = 214
    row_h = 46
    for index, (row_id, kind, given, predicted, confidence, status, flagged) in enumerate(rows):
        y0 = top + index * row_h
        if flagged:
            draw.rounded_rectangle((48, y0 - 4, 1016, y0 + 40), radius=8, fill=CHECK_BG)
        draw.text((60, y0 + 6), row_id, font=body, fill=TEXT)
        paste_thumb(canvas, kind, (150, y0, 186, y0 + 36))
        draw.text((250, y0 + 6), given, font=body, fill=TEXT)
        draw.text((420, y0 + 6), predicted, font=body, fill=TEXT)
        draw.text((640, y0 + 6), confidence, font=body, fill=TEXT)
        if status == "Clean":
            draw.rounded_rectangle((820, y0 + 4, 910, y0 + 32), radius=12, fill=GREEN_BG)
            draw.text((838, y0 + 6), "Clean", font=tiny, fill=GREEN)
        else:
            draw.rounded_rectangle((820, y0 + 4, 920, y0 + 32), radius=12, outline=ORANGE, width=2)
            draw.text((836, y0 + 6), "Check", font=tiny, fill=ORANGE_2)

    draw.text((1080, 142), "SUSPICION SCORE (TOP RANKED)", font=label_font, fill=MUTED)
    # question mark
    draw.ellipse((1496, 138, 1536, 178), outline=ORANGE, width=2)
    draw.text((1506, 144), "?", font=label_font, fill=ORANGE)

    bars = [
        ("1", "0189.jpg", 0.87),
        ("2", "0123.jpg", 0.74),
        ("3", "0256.jpg", 0.61),
        ("4", "0478.jpg", 0.53),
        ("5", "0332.jpg", 0.42),
    ]
    for index, (rank, name, score) in enumerate(bars):
        y0 = 196 + index * 64
        draw.text((1080, y0), rank, font=font("Inter-Bold.ttf", 26), fill=TEXT)
        draw.ellipse((1124, y0 + 2, 1156, y0 + 34), outline=ORANGE, width=2)
        draw.text((1134, y0 + 4), "?", font=small, fill=ORANGE)
        draw.text((1170, y0 + 4), name, font=body, fill=TEXT)
        bar_w = int(250 * score)
        draw.rounded_rectangle((1170, y0 + 36, 1170 + 250, y0 + 50), radius=7, fill=DEEP)
        draw.rounded_rectangle((1170, y0 + 36, 1170 + bar_w, y0 + 50), radius=7, fill=ORANGE)
        draw.text((1450, y0 + 8), f"{score:.2f}", font=font("Inter-Bold.ttf", 22), fill=TEXT)
    draw.text(
        (1080, 520),
        "Lower confidence + label mismatch = higher suspicion",
        font=small,
        fill=MUTED,
    )

    draw.text((60, 598), "IMAGE REVIEW GRID", font=label_font, fill=MUTED)
    tiles = [
        ("cat", "cat", True),
        ("cat", "dog", False),
        ("dog", "dog", True),
        ("car", "car", True),
        ("truck", "truck", False),
        ("bird", "bird", True),
        ("tree", "tree", True),
        ("chair", "chair", True),
    ]
    for index, (kind, caption, ok) in enumerate(tiles):
        col, row = index % 4, index // 4
        x0 = 60 + col * 240
        y0 = 640 + row * 110
        if not ok and kind == "cat":
            draw.rounded_rectangle((x0 - 4, y0 - 4, x0 + 204, y0 + 100), radius=12, outline=ORANGE, width=3)
        paste_thumb(canvas, kind, (x0, y0, x0 + 72, y0 + 72))
        draw.text((x0 + 84, y0 + 16), caption, font=body, fill=TEXT)
        mark = "✓" if ok else "×"
        color = GREEN if ok else (240, 113, 120)
        draw.text((x0 + 84, y0 + 44), mark, font=font("Inter-Bold.ttf", 22), fill=color)
    # magnifier over the mislabeled cat
    mag = ImageDraw.Draw(canvas)
    mag.ellipse((250, 650, 330, 730), outline=ORANGE, width=8)
    mag.line((318, 718, 360, 760), fill=ORANGE, width=8)

    icon = rasterize_svg(220).resize((250, 250), Image.Resampling.LANCZOS)
    canvas.paste(icon, (1270, 630), icon)
    return canvas


def main() -> None:
    try:
        import cairosvg  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cairosvg"])

    sizes = {
        "icp4": 16,
        "icp5": 32,
        "icp6": 64,
        "ic07": 128,
        "ic08": 256,
        "ic09": 512,
        "ic10": 1024,
        "ic11": 32,
        "ic12": 64,
        "ic13": 256,
        "ic14": 512,
    }
    rendered = {name: rasterize_svg(size) for name, size in sizes.items()}
    master = rendered["ic10"]
    master.save(ASSETS / "icon-1024.png")
    rendered["ic09"].save(ASSETS / "icon-512.png")
    master.save(ASSETS / "icon.png")
    ico = master.copy()
    ico.save(
        ASSETS / "icon.ico",
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    write_icns(ASSETS / "icon.icns", {name: png_bytes(image) for name, image in rendered.items()})
    ico.save(WEB / "favicon.ico", format="ICO", sizes=[(16, 16), (32, 32), (48, 48)])
    rendered["ic08"].save(WEB / "icon.png")

    cover = draw_cover()
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    cover.save(docs / "cover.jpg", quality=92, optimize=True)
    print(f"Wrote {ASSETS / 'icon.png'} and {docs / 'cover.jpg'}")


if __name__ == "__main__":
    main()
