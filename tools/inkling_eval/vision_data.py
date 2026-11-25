# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""The synthetic vision set: 88 deterministic, judge-free image tasks.

Every image is drawn here with a tiny pure-Python canvas and PNG writer; there
are no photographs or downloaded datasets.  The set is sent to the model's
``/v1/chat/completions`` endpoint as OpenAI ``image_url`` content parts carrying
base64 PNG data URLs.  The Inkling chat template renders each part as
``<role><|content_image|><|unused_200054|><|end_message|>`` and the Inkling-2
processor expands the pad to one token per 40x40 pixel cell.

Families: OCR-style codes/numbers (24), bar charts (10), line charts (10),
shape/colour counts (20), table row reads (16) and eight text-only controls.
The scorer sees only the committed answer channel.
"""

import base64
import hashlib
import json
import random
import struct
import zlib

from tools.inkling_eval import stats, tasks

WIDTH, HEIGHT = 480, 240
SYSTEM = {
    "role": "system",
    "content": (
        "You are a precise vision assistant. Read only what is visible in the "
        "image. Answer with the exact requested value and nothing else."
    ),
}
CONTROL_SYSTEM = {
    "role": "system",
    "content": "You are a precise assistant. Answer the question directly and concisely.",
}

# 5x7 bitmap font, one string per row.  Pure Python keeps the PNG bytes
# identical across machines and Pillow versions.
FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01110"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ":": ("00000", "00100", "00100", "00000", "00100", "00100", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00100", "00100"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
}

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (190, 190, 190)
RED = (220, 40, 40)
BLUE = (40, 70, 220)
GREEN = (30, 150, 70)
ORANGE = (240, 140, 0)
PURPLE = (130, 60, 180)
COLORS = {"red": RED, "blue": BLUE, "green": GREEN, "orange": ORANGE, "purple": PURPLE}


class Canvas:
    def __init__(self, width: int = WIDTH, height: int = HEIGHT, background=WHITE):
        self.width = width
        self.height = height
        self.rows = [bytearray(background * width) for _ in range(height)]

    def pixel(self, x: int, y: int, color) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.rows[y][x * 3 : x * 3 + 3] = bytes(color)

    def rect(self, x0: int, y0: int, x1: int, y1: int, color) -> None:
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        for y in range(max(0, y0), min(self.height, y1)):
            row = self.rows[y]
            for x in range(max(0, x0), min(self.width, x1)):
                row[x * 3 : x * 3 + 3] = bytes(color)

    def line(self, x0: int, y0: int, x1: int, y1: int, color, width: int = 1) -> None:
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx - dy
        half = width // 2
        while True:
            self.rect(x0 - half, y0 - half, x0 + half + 1, y0 + half + 1, color)
            if x0 == x1 and y0 == y1:
                break
            twice = 2 * error
            if twice > -dy:
                error -= dy
                x0 += sx
            if twice < dx:
                error += dx
                y0 += sy

    def circle(self, cx: int, cy: int, radius: int, color) -> None:
        r2 = radius * radius
        for y in range(cy - radius, cy + radius + 1):
            span = int((r2 - (y - cy) ** 2) ** 0.5)
            self.rect(cx - span, y, cx + span + 1, y + 1, color)

    def text(self, x: int, y: int, value: str, color=BLACK, scale: int = 2) -> int:
        cursor = x
        for char in value.upper():
            glyph = FONT.get(char)
            if glyph is None:
                cursor += 6 * scale
                continue
            for row_index, row in enumerate(glyph):
                for col_index, bit in enumerate(row):
                    if bit == "1":
                        self.rect(
                            cursor + col_index * scale,
                            y + row_index * scale,
                            cursor + (col_index + 1) * scale,
                            y + (row_index + 1) * scale,
                            color,
                        )
            cursor += 6 * scale
        return cursor - x

    def centered(self, y: int, value: str, color=BLACK, scale: int = 2) -> None:
        width = _text_width(value, scale)
        self.text(max(0, (self.width - width) // 2), y, value, color, scale)

    def png(self) -> bytes:
        # Indexed colour plus stored DEFLATE keeps the bytes independent of the
        # host zlib version; a compressed PNG would make the plan hash depend on it.
        colors = sorted(
            {
                bytes(row[x * 3 : x * 3 + 3])
                for row in self.rows
                for x in range(self.width)
            }
        )
        index = {color: position for position, color in enumerate(colors)}
        raw = b"".join(
            b"\x00" + bytes(index[bytes(row[x * 3 : x * 3 + 3])] for x in range(self.width))
            for row in self.rows
        )

        def chunk(tag: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + tag
                + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
            )

        def stored_zlib(data: bytes) -> bytes:
            out = bytearray(b"\x78\x01")
            position = 0
            while True:
                block = data[position : position + 65535]
                final = position + len(block) >= len(data)
                out.append(1 if final else 0)
                out += struct.pack("<HH", len(block), 0xFFFF - len(block))
                out += block
                position += len(block)
                if final:
                    break
            out += struct.pack(">I", zlib.adler32(data) & 0xFFFFFFFF)
            return bytes(out)

        header = struct.pack(">IIBBBBB", self.width, self.height, 8, 3, 0, 0, 0)
        palette = b"".join(colors)
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"PLTE", palette)
            + chunk(b"IDAT", stored_zlib(raw))
            + chunk(b"IEND", b"")
        )


def _text_width(value: str, scale: int) -> int:
    return max(0, len(value) * 6 * scale - scale)


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _cells(png_width: int = WIDTH, png_height: int = HEIGHT) -> int:
    return -(-png_height // 40) * -(-png_width // 40)


def _frame(canvas: Canvas) -> None:
    canvas.rect(0, 0, canvas.width, 3, BLACK)
    canvas.rect(0, canvas.height - 3, canvas.width, canvas.height, BLACK)
    canvas.rect(0, 0, 3, canvas.height, BLACK)
    canvas.rect(canvas.width - 3, 0, canvas.width, canvas.height, BLACK)


def _image_item(
    *,
    item_id: str,
    subcategory: str,
    prompt: str,
    png: bytes,
    score_name: str,
    expected: dict,
    gold: str,
    family: str,
    detail: dict | None = None,
) -> dict:
    url = _data_url(png)
    messages = [
        SYSTEM,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        },
    ]
    prompt_payload = json.dumps(messages, sort_keys=True, separators=(",", ":"))
    text_tokens = tasks.estimate_tokens(prompt)
    return {
        "id": item_id,
        "category": "vision",
        "subcategory": subcategory,
        "scorer": score_name,
        "max_tokens": 768,
        "prompt": prompt_payload,
        "approx_prompt_tokens": text_tokens + _cells() + 20,
        "messages": messages,
        "transport": "chat",
        "image_parts": 1,
        "image_sha256": _sha256(png),
        "expected": expected,
        "gold": gold,
        "meta": {"family": family, "synthetic": True, **(detail or {})},
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- OCR


OCR_VALUES = [
    "AB1234",
    "X7K2",
    "Q4821",
    "ZX90",
    "M4N8",
    "P731",
    "K49Q",
    "B208",
    "T5V7",
    "R614",
    "W93L",
    "C805",
    "4821",
    "73905",
    "2064",
    "91837",
    "5560",
    "30471",
    "8812",
    "6473",
    "12096",
    "735",
    "48015",
    "9268",
]


def _ocr_items() -> list[dict]:
    items = []
    for index, value in enumerate(OCR_VALUES):
        canvas = Canvas()
        _frame(canvas)
        label = "CODE" if any(ch.isalpha() for ch in value) else "NUMBER"
        canvas.centered(30, label, BLACK, 3)
        scale = 14 if len(value) <= 4 else 10
        canvas.centered(100, value, BLACK, scale)
        prompt = (
            f"Read the {label.lower()} in the image. "
            "Answer with the exact characters only."
        )
        items.append(
            _image_item(
                item_id=f"vision-ocr-{index:02d}",
                subcategory="ocr",
                prompt=prompt,
                png=canvas.png(),
                score_name="qa_word",
                expected={"answers": [value]},
                gold=value,
                family="ocr",
                detail={"value": value, "label": label},
            )
        )
    return items


# --------------------------------------------------------------------------- charts


def _chart_canvas(
    *,
    labels: list[str],
    values: list[int],
    line: bool,
    seed: int,
) -> Canvas:
    canvas = Canvas()
    _frame(canvas)
    left, right, top, bottom = 70, 455, 35, 205
    maximum = 60
    for tick in (0, 20, 40, 60):
        y = bottom - int(tick / maximum * (bottom - top))
        canvas.line(left, y, right, y, GRAY, 1)
        canvas.text(left - 35, y - 7, str(tick), BLACK, 2)
    canvas.line(left, top, left, bottom, BLACK, 2)
    canvas.line(left, bottom, right, bottom, BLACK, 2)

    if line:
        step = (right - left - 20) // (len(values) - 1)
        points = []
        for index, value in enumerate(values):
            x = left + 10 + index * step
            y = bottom - int(value / maximum * (bottom - top))
            points.append((x, y))
        for first, second in zip(points, points[1:]):
            canvas.line(first[0], first[1], second[0], second[1], BLUE, 3)
        for index, (x, y) in enumerate(points):
            canvas.circle(x, y, 5, BLUE)
            canvas.text(x - 6, bottom + 10, labels[index], BLACK, 2)
    else:
        slot = (right - left) // len(values)
        for index, value in enumerate(values):
            x0 = left + index * slot + slot // 4
            x1 = left + index * slot + 3 * slot // 4
            y = bottom - int(value / maximum * (bottom - top))
            color = BLUE if index != seed % len(values) else GREEN
            canvas.rect(x0, y, x1, bottom, color)
            canvas.text((x0 + x1) // 2 - 6, bottom + 10, labels[index], BLACK, 2)
    return canvas


BAR_VALUES = [
    [20, 50, 30, 60, 10],
    [40, 10, 50, 20, 60],
    [30, 60, 20, 40, 10],
    [50, 20, 60, 10, 40],
    [10, 40, 30, 50, 20],
    [60, 30, 10, 40, 50],
    [20, 30, 60, 50, 10],
    [40, 60, 20, 30, 50],
    [30, 10, 50, 60, 20],
    [50, 40, 10, 20, 60],
]
LINE_VALUES = [
    [10, 30, 20, 50, 40],
    [40, 10, 50, 20, 60],
    [20, 60, 30, 10, 40],
    [50, 30, 60, 20, 10],
    [30, 50, 10, 40, 20],
    [60, 20, 40, 50, 30],
    [10, 50, 60, 30, 20],
    [40, 20, 30, 60, 10],
    [20, 40, 10, 60, 50],
    [50, 60, 20, 10, 30],
]
LABELS = ["A", "B", "C", "D", "E"]


def _chart_items() -> list[dict]:
    items = []
    for index in range(20):
        line = index >= 10
        variant = index % 10
        values = (LINE_VALUES if line else BAR_VALUES)[variant]
        canvas = _chart_canvas(labels=LABELS, values=values, line=line, seed=variant)
        subcategory = "chart_line" if line else "chart_bar"
        if variant % 2 == 0:
            highest = LABELS[values.index(max(values))]
            prompt = "Which point has the highest value? Answer with its label."
            if not line:
                prompt = "Which bar is tallest? Answer with its label."
            expected = {"answers": [highest]}
            score_name = "qa_word"
            gold = highest
            detail = {"kind": "highest_label", "values": values}
        else:
            target = "C"
            value = values[LABELS.index(target)]
            prompt = f"What value does {target} show? Answer with the number."
            if not line:
                prompt = f"What is the value of bar {target}? Answer with the number."
            expected = {"answer": str(value), "tolerance": 2.0}
            score_name = "vision_chart"
            gold = str(value)
            detail = {"kind": "label_value", "label": target, "value": value}
        items.append(
            _image_item(
                item_id=f"vision-{subcategory.replace('_', '-')}-{variant:02d}",
                subcategory=subcategory,
                prompt=prompt,
                png=canvas.png(),
                score_name=score_name,
                expected=expected,
                gold=gold,
                family="charts",
                detail=detail,
            )
        )
    return items


# --------------------------------------------------------------------------- counts


COUNT_SPECS = [
    # (total shapes, target shape, target colour, distractor)
    (6, "circle", "red", "square"),
    (7, "square", "blue", "circle"),
    (5, "circle", "green", "square"),
    (8, "square", "red", "circle"),
    (6, "circle", "orange", "square"),
    (7, "square", "green", "circle"),
    (5, "circle", "blue", "square"),
    (8, "square", "orange", "circle"),
    (6, "circle", "purple", "square"),
    (7, "square", "purple", "circle"),
    (5, "circle", "red", "square"),
    (8, "square", "blue", "circle"),
    (6, "circle", "green", "square"),
    (7, "square", "red", "circle"),
    (5, "circle", "orange", "square"),
    (8, "square", "green", "circle"),
    (6, "circle", "blue", "square"),
    (7, "square", "orange", "circle"),
    (5, "circle", "purple", "square"),
    (8, "square", "purple", "circle"),
]


def _count_items() -> list[dict]:
    items = []
    for index, (total, target_shape, target_color, other_shape) in enumerate(COUNT_SPECS):
        rng = random.Random(index)
        positions = [(70 + (i % 4) * 100, 70 + (i // 4) * 70) for i in range(total)]
        rng.shuffle(positions)
        shapes = []
        target_count = 1 + index % 4
        for slot in range(total):
            if slot < target_count:
                shape, color = target_shape, target_color
            else:
                shape, color = other_shape, list(COLORS)[(index + slot) % len(COLORS)]
                if color == target_color:
                    color = "purple" if target_color != "purple" else "green"
            shapes.append((positions[slot], shape, color))
        canvas = Canvas()
        _frame(canvas)
        for (x, y), shape, color in shapes:
            if shape == "circle":
                canvas.circle(x, y, 28, COLORS[color])
            else:
                canvas.rect(x - 28, y - 28, x + 28, y + 28, COLORS[color])
        ask_total = index % 4 == 3
        if ask_total:
            prompt = f"How many {other_shape}s are in the image? Answer with a number."
            count = sum(1 for _, shape, _ in shapes if shape == other_shape)
            expected = {"answer": str(count)}
            gold = str(count)
            detail = {"kind": "count_shape", "shape": other_shape}
        else:
            prompt = (
                f"How many {target_color} {target_shape}s are in the image? "
                "Answer with a number."
            )
            count = target_count
            expected = {"answer": str(count)}
            gold = str(count)
            detail = {"kind": "count_color_shape", "shape": target_shape, "color": target_color}
        items.append(
            _image_item(
                item_id=f"vision-count-{index:02d}",
                subcategory="count",
                prompt=prompt,
                png=canvas.png(),
                score_name="qa_numeric",
                expected=expected,
                gold=gold,
                family="count",
                detail={**detail, "count": count, "total": total},
            )
        )
    return items


# --------------------------------------------------------------------------- table


TABLE_SETS = [
    [
        ["A1", "OSAKA", "2026-12-03", "2"],
        ["B2", "LIMA", "2027-01-07", "5"],
        ["C3", "PRAGUE", "2026-12-17", "3"],
        ["D4", "TORONTO", "2027-01-14", "4"],
    ],
    [
        ["E5", "HELSINKI", "2026-11-30", "1"],
        ["F6", "NAIROBI", "2027-02-04", "6"],
        ["G7", "AUCKLAND", "2026-12-21", "2"],
        ["H8", "BERLIN", "2027-01-21", "3"],
    ],
    [
        ["J9", "LISBON", "2026-11-26", "4"],
        ["K1", "SEOUL", "2027-02-11", "2"],
        ["L2", "LIMA", "2026-12-24", "5"],
        ["M3", "PRAGUE", "2027-01-28", "1"],
    ],
    [
        ["N4", "OSAKA", "2026-12-31", "3"],
        ["P5", "TORONTO", "2027-02-18", "2"],
        ["Q6", "BERLIN", "2026-12-08", "6"],
        ["R7", "HELSINKI", "2027-02-25", "1"],
    ],
]
TABLE_HEADER = ["CODE", "CITY", "DATE", "NIGHTS"]
TABLE_X = [20, 110, 230, 395]


def _table_canvas(rows: list[list[str]]) -> Canvas:
    canvas = Canvas()
    _frame(canvas)
    for x in TABLE_X:
        canvas.text(x, 20, TABLE_HEADER[TABLE_X.index(x)], BLACK, 2)
    canvas.line(15, 48, 465, 48, BLACK, 2)
    for row_index, row in enumerate(rows):
        y = 62 + row_index * 40
        for x, value in zip(TABLE_X, row):
            canvas.text(x, y, value, BLACK, 2)
        canvas.line(15, y + 30, 465, y + 30, GRAY, 1)
    return canvas


def _table_items() -> list[dict]:
    items = []
    fields = (
        ("CITY", 1, "qa_word"),
        ("DATE", 2, "qa_word"),
        ("NIGHTS", 3, "qa_numeric"),
        ("CODE", 0, "qa_word"),
    )
    for index in range(16):
        rows = [row[:] for row in TABLE_SETS[index // 4]]
        shift = index % len(rows)
        rows = rows[shift:] + rows[:shift]
        canvas = _table_canvas(rows)
        row = rows[index % len(rows)]
        field, value_index, score_name = fields[index % len(fields)]
        value = row[value_index]
        if field == "CODE":
            prompt = (
                f"What is the code of the row for {row[1]}? "
                "Answer with the value only."
            )
        else:
            prompt = (
                f"In the row with code {row[0]}, what is the {field.lower()}? "
                "Answer with the value only."
            )
        items.append(
            _image_item(
                item_id=f"vision-table-{index:02d}",
                subcategory="table",
                prompt=prompt,
                png=canvas.png(),
                score_name=score_name,
                expected=(
                    {"answers": [value]} if score_name == "qa_word" else {"answer": value}
                ),
                gold=value,
                family="table",
                detail={"row": row[0], "field": field.lower(), "value": value},
            )
        )
    return items


# --------------------------------------------------------------------------- controls


CONTROL_NOTES = [
    ("LIMA", "2027-01-07", "2", "3"),
    ("OSAKA", "2026-12-03", "5", "1"),
    ("PRAGUE", "2026-12-17", "3", "2"),
    ("TORONTO", "2027-01-14", "4", "4"),
]


def _control_items() -> list[dict]:
    items = []
    questions = [
        ("CITY", "What city is in the note? Answer with one word."),
        ("DATE", "What check-in date is in the note? Answer YYYY-MM-DD."),
        ("NIGHTS", "How many nights are in the note? Answer with a number."),
        ("GUESTS", "How many guests are in the note? Answer with a number."),
    ]
    for index in range(8):
        city, check_in, nights, guests = CONTROL_NOTES[index % len(CONTROL_NOTES)]
        note = (
            f"Synthetic note {index + 1}: city {city}; check-in {check_in}; "
            f"{nights} nights; {guests} guests. No image is attached."
        )
        field, question = questions[index % len(questions)]
        value = {"CITY": city, "DATE": check_in, "NIGHTS": nights, "GUESTS": guests}[field]
        messages = [
            CONTROL_SYSTEM,
            {"role": "user", "content": [{"type": "text", "text": note + " " + question}]},
        ]
        items.append(
            {
                "id": f"vision-control-{index:02d}",
                "category": "vision",
                "subcategory": "control",
                "scorer": "qa_word" if field in ("CITY", "DATE") else "qa_numeric",
                "max_tokens": 768,
                "prompt": json.dumps(messages, sort_keys=True, separators=(",", ":")),
                "approx_prompt_tokens": tasks.estimate_tokens(note + " " + question) + 20,
                "messages": messages,
                "transport": "chat",
                "image_parts": 0,
                "expected": (
                    {"answers": [value]} if field in ("CITY", "DATE") else {"answer": value}
                ),
                "gold": value,
                "meta": {"family": "control", "synthetic": True, "field": field.lower()},
            }
        )
    return items


# --------------------------------------------------------------------------- plan


def build_plan(*, seed: int = 0, reasoning_effort: float = 0.0, max_model_len: int | None = None) -> dict:
    """The 88-item vision set, with its own hash and chat transport."""
    items = (
        _ocr_items()
        + _chart_items()
        + _count_items()
        + _table_items()
        + _control_items()
    )
    mdd = stats.mdd(len(items), 0.10)
    return tasks.finalize_plan(
        items,
        seed=seed,
        wanted=("vision",),
        usable=(),
        max_model_len=max_model_len,
        reasoning_effort=reasoning_effort,
        set_name="vision",
        extra_meta={
            "transport": "/v1/chat/completions",
            "thinking_effort": "none",
            "vision": {
                "items": len(items),
                "image_items": sum(1 for item in items if item.get("image_parts")),
                "text_controls": sum(1 for item in items if not item.get("image_parts")),
                "families": {
                    "ocr": 24,
                    "charts": 20,
                    "count": 20,
                    "table": 16,
                    "control": 8,
                },
                "assumed_discordance": 0.10,
                "mdd_pp": round(100 * mdd, 1) if mdd is not None else None,
                "note": "MDE at n=88 and 10% discordance; all images are synthetic.",
            },
        },
    )
