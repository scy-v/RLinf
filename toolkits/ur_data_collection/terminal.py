# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Terminal input for the standalone keyboard collector."""

import logging
import os
import select
import sys
import termios
import unicodedata
from contextlib import contextmanager
from typing import Any, Iterator

from rlinf.utils.logging import get_logger

logger = get_logger()


class RolloutConsoleFormatter(logging.Formatter):
    """Show routine messages directly while retaining warning/error levels."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if record.levelno >= logging.WARNING:
            return f"[{record.levelname}] {message}"
        return message


def configure_console_logging() -> None:
    """Configure RLinf console output for the standalone rollout entry point."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(RolloutConsoleFormatter("%(message)s"))
    logger.handlers = [handler]
    logger.propagate = False


_TITLE_COLORS = {
    "采集": 117,
    "采集统计": 121,
    "结果判定": 222,
    "复位": 147,
    "等待": 153,
    "保留": 121,
    "退出": 218,
    "异常退出": 210,
    "回收站": 215,
}


def format_collection_summary(counts: list[list[int]]) -> str:
    """Format autonomous/intervened success/failure counts in terminal columns."""
    rows = [
        ["类型", "成功", "失败", "合计"],
        ["自主 rollout", *map(str, counts[0]), str(sum(counts[0]))],
        ["人工干预", *map(str, counts[1]), str(sum(counts[1]))],
        [
            "总计",
            str(counts[0][0] + counts[1][0]),
            str(counts[0][1] + counts[1][1]),
            str(sum(map(sum, counts))),
        ],
    ]

    def width(text: str) -> int:
        return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)

    widths = [max(width(row[i]) for row in rows) for i in range(4)]

    def border(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * (w + 2) for w in widths) + right

    lines = [color_status("[采集统计] 当前数据集已保存轨迹"), border("┌", "┬", "┐")]
    for index, row in enumerate(rows):
        cells = []
        for col, value in enumerate(row):
            padding = " " * (widths[col] - width(value))
            cells.append(value + padding if col == 0 else padding + value)
        lines.append("│ " + " │ ".join(cells) + " │")
        if index in (0, 2):
            lines.append(border("├", "┼", "┤"))
    lines.append(border("└", "┴", "┘"))
    return "\n".join(lines)


def color_status(message: str) -> str:
    """Color a known status title on interactive terminals only."""
    if not sys.stderr.isatty() or "NO_COLOR" in os.environ:
        return message
    for title, color in _TITLE_COLORS.items():
        tag = f"[{title}]"
        if tag in message:
            return message.replace(tag, f"\033[38;5;{color}m{tag}\033[0m", 1)
    return message


@contextmanager
def quiet_keyboard(*, echo: bool = False) -> Iterator[None]:
    """Set keyboard echo for teleop or prompts, restoring settings on exit."""
    if not sys.stdin.isatty():
        yield
        return
    original = termios.tcgetattr(sys.stdin)
    quiet = original.copy()
    if echo:
        quiet[3] |= termios.ECHO | termios.ECHONL
    else:
        quiet[3] &= ~(termios.ECHO | termios.ECHONL)
    termios.tcsetattr(sys.stdin, termios.TCSANOW, quiet)
    try:
        yield
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSANOW, original)


def wait_for_enter(prompt: str, controls: Any) -> bool:
    """Wait for a fresh Enter, or return False when Escape requests exit."""
    logger.info(
        color_status("---------------- [等待] %s（Esc 退出） ----------------"), prompt
    )
    if sys.stdin.isatty():
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    while not controls.snapshot().stop:
        readable, _, _ = select.select([sys.stdin], [], [], 0.05)
        if not readable:
            continue
        line = sys.stdin.readline()
        if line == "":
            raise EOFError("Terminal input closed while waiting for Enter.")
        if line.strip() == "":
            return not controls.snapshot().stop
        logger.info("请仅按回车继续。")
    return False
