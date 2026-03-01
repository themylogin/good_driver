"""
Dense binary encoding for per-pixel segmentation masks at model resolution.

Each pixel is classified into one of four classes (2 bits):

    0b00 (0) = Nothing          — not driveable, not a lane separator
    0b01 (1) = Driveable area   — driveable surface, no lane separator
    0b10 (2) = Lane separator (on driveable area)
    0b11 (3) = Lane separator (no driveable area beneath)

Encoding layout
---------------
Only rows that contain at least one non-zero pixel are stored.  All pixels
in those rows (including leading/trailing zeros) are packed so that four
pixels occupy one byte, MSB-first:

    byte = (px0 << 6) | (px1 << 4) | (px2 << 2) | px3

For a 640-wide image this gives exactly 160 bytes per row.

Wire format (JSON-friendly dict)
---------------------------------
{
    "start_row":  int,       # first model row with any content
    "row_count":  int,       # number of consecutive rows stored
    "width":      int,       # model-resolution width (e.g. 640)
    "offsets":    list[int], # per stored row: column index of first non-zero pixel
    "data":       str,       # base64-encoded packed bytes (row_count * width/4)
}

Usage example
-------------
    from good_driver.segmentation_mask import encode_mask, decode_mask

    # Encode from two model-resolution binary masks
    payload = encode_mask(da_mask, lane_mask, width=640)

    # Decode back into a 2-D class array
    classes = decode_mask(payload)
    # classes.shape == (row_count, width), dtype uint8, values 0-3
"""

from __future__ import annotations

import base64
from typing import TypedDict

import numpy as np


class MaskPayload(TypedDict):
    start_row: int
    row_count: int
    width: int
    offsets: list[int]
    data: str


# Pixel class constants
NOTHING = 0
DRIVEABLE = 1
LANE_ON_DRIVEABLE = 2
LANE_NO_DRIVEABLE = 3


def encode_mask(
    da_mask: np.ndarray,
    lane_mask: np.ndarray,
    width: int,
) -> MaskPayload | None:
    """Encode two binary masks into a dense base64 payload.

    Parameters
    ----------
    da_mask : np.ndarray
        Driveable-area mask at model resolution (H×W), nonzero = driveable.
    lane_mask : np.ndarray
        Lane-separator mask at model resolution (H×W), nonzero = lane line.
    width : int
        Model-resolution width (must match mask columns and be divisible by 4).

    Returns
    -------
    MaskPayload or None
        Encoded payload dict, or None if the masks are entirely empty.
    """
    assert width % 4 == 0, f"width must be divisible by 4, got {width}"
    assert da_mask.shape[1] == width and lane_mask.shape[1] == width

    # Build per-pixel class array: 0/1/2/3
    da = da_mask > 0
    ll = lane_mask > 0
    classes = np.zeros(da.shape, dtype=np.uint8)
    classes[da & ~ll] = DRIVEABLE           # driveable only
    classes[da & ll] = LANE_ON_DRIVEABLE    # lane separator on driveable area
    classes[~da & ll] = LANE_NO_DRIVEABLE   # lane separator, no driveable area

    # Find row range with content
    row_has_content = classes.any(axis=1)
    active_rows = np.where(row_has_content)[0]
    if len(active_rows) == 0:
        return None

    start_row = int(active_rows[0])
    end_row = int(active_rows[-1])
    row_count = end_row - start_row + 1

    region = classes[start_row : start_row + row_count]  # (row_count, width)

    # Compute per-row first non-zero column
    offsets: list[int] = []
    for r in range(row_count):
        nz = np.nonzero(region[r])[0]
        offsets.append(int(nz[0]) if len(nz) > 0 else width)

    # Pack 4 pixels per byte, MSB-first
    packed = _pack_2bit(region, width)

    return {
        "start_row": start_row,
        "row_count": row_count,
        "width": width,
        "offsets": offsets,
        "data": base64.b64encode(packed).decode("ascii"),
    }


def decode_mask(payload: MaskPayload) -> np.ndarray:
    """Decode a MaskPayload back into a 2-D class array.

    Returns
    -------
    np.ndarray
        Shape (row_count, width), dtype uint8, values in {0, 1, 2, 3}.
    """
    width = payload["width"]
    row_count = payload["row_count"]
    raw = base64.b64decode(payload["data"])
    return _unpack_2bit(raw, row_count, width)


def _pack_2bit(classes: np.ndarray, width: int) -> bytes:
    """Pack a (rows, width) uint8 array of 0-3 values into bytes, 4 pixels per byte."""
    rows = classes.shape[0]
    cols_per_byte = 4
    bytes_per_row = width // cols_per_byte
    flat = classes.reshape(rows, bytes_per_row, cols_per_byte)
    packed = (
        (flat[:, :, 0].astype(np.uint8) << 6)
        | (flat[:, :, 1].astype(np.uint8) << 4)
        | (flat[:, :, 2].astype(np.uint8) << 2)
        | flat[:, :, 3].astype(np.uint8)
    )
    return packed.tobytes()


def _unpack_2bit(raw: bytes, rows: int, width: int) -> np.ndarray:
    """Unpack bytes into a (rows, width) uint8 array of 0-3 values."""
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, width // 4)
    out = np.empty((rows, width), dtype=np.uint8)
    out[:, 0::4] = (arr >> 6) & 0x03
    out[:, 1::4] = (arr >> 4) & 0x03
    out[:, 2::4] = (arr >> 2) & 0x03
    out[:, 3::4] = arr & 0x03
    return out
