import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

TIME_PATTERN = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}[\.,]\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}[\.,]\d{3})"
)


@dataclass
class SubtitleEntry:
    start_time: str
    end_time: str
    start_seconds: float
    end_seconds: float
    text: str


def parse_timecode(value: str) -> float:
    hours, minutes, rest = value.replace(",", ".").split(":")
    seconds = float(rest)
    return int(hours) * 3600 + int(minutes) * 60 + seconds


def clean_text(lines: Iterable[str]) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_subtitle_file(path: Path) -> List[SubtitleEntry]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".vtt":
        raw = re.sub(r"^WEBVTT.*\n", "", raw, flags=re.IGNORECASE)
    blocks = re.split(r"\n\s*\n", raw.strip())
    entries: List[SubtitleEntry] = []

    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        time_line_index = 0
        if re.match(r"^\d+$", lines[0]):
            time_line_index = 1
        if time_line_index >= len(lines):
            continue
        match = TIME_PATTERN.search(lines[time_line_index])
        if not match:
            continue
        start_time = match.group("start")
        end_time = match.group("end")
        text_lines = lines[time_line_index + 1 :]
        text = clean_text(text_lines)
        if not text:
            continue
        entries.append(
            SubtitleEntry(
                start_time=start_time.replace(",", "."),
                end_time=end_time.replace(",", "."),
                start_seconds=parse_timecode(start_time),
                end_seconds=parse_timecode(end_time),
                text=text,
            )
        )
    return entries
