"""Extract embedded GPS data from Novatek MP4 files.

Based on https://github.com/Cynobs-repo/pydashcamviewer/blob/main/pydashcam/nvtk_mp42gpx.py
(Author: Sergei Franco, License: GPL3)
"""
from __future__ import annotations

import struct
from pathlib import Path


def _get_atom_info(eight_bytes: bytes) -> tuple[int, str]:
    try:
        atom_size, atom_type = struct.unpack(">I4s", eight_bytes)
    except struct.error:
        return 0, ""
    try:
        a_t = atom_type.decode()
    except UnicodeDecodeError:
        a_t = "UNKNOWN"
    return int(atom_size), a_t


def _get_gps_atom_info(eight_bytes: bytes) -> tuple[int, int]:
    atom_pos, atom_size = struct.unpack(">II", eight_bytes)
    return int(atom_pos), int(atom_size)


def _get_gps_offset(data: bytes) -> int:
    pointer = len(data) - 20
    beginning = 0
    while pointer > beginning:
        active, lon_hemi, lat_hemi = struct.unpack_from("<sss", data, pointer)
        try:
            active = active.decode()
            lon_hemi = lon_hemi.decode()
            lat_hemi = lat_hemi.decode()
        except UnicodeDecodeError:
            pointer -= 1
            continue
        if active == "A" and lon_hemi in ("N", "S") and lat_hemi in ("E", "W"):
            return pointer - 24
        pointer -= 1
    return -1


def _fix_coordinates(hemisphere: str, coordinate: float) -> float:
    minutes = coordinate % 100.0
    degrees = coordinate - minutes
    coordinate = degrees / 100.0 + (minutes / 60.0)
    if hemisphere in ("S", "W"):
        return -coordinate
    return coordinate


def _parse_gps_payload(data: bytes) -> dict | None:
    offset = _get_gps_offset(data)
    if offset < 0:
        return None

    hour, minute, second = struct.unpack_from("<III", data, offset)
    offset += 12
    year, month, day = struct.unpack_from("<III", data, offset)
    offset += 12
    active, lat_hemi, lon_hemi = struct.unpack_from("<sss", data, offset)
    offset += 4
    lat_raw, lon_raw = struct.unpack_from("<ff", data, offset)
    offset += 8
    speed_knots, _bearing = struct.unpack_from("<ff", data, offset)

    try:
        active = active.decode()
        lat_hemi = lat_hemi.decode()
        lon_hemi = lon_hemi.decode()
    except UnicodeDecodeError:
        return None

    if active != "A":
        return None

    return {
        "lat": _fix_coordinates(lat_hemi, lat_raw),
        "lon": _fix_coordinates(lon_hemi, lon_raw),
        "datetime": "%d-%02d-%02dT%02d:%02d:%02dZ" % (
            year + 2000, int(month), int(day),
            int(hour), int(minute), int(second),
        ),
        "speed_kmh": round(speed_knots * 1.852, 1),
    }


def _parse_gps_atom(atom_pos: int, atom_size: int, fh) -> dict | None:
    if atom_size == 0 or atom_pos == 0:
        return None
    fh.seek(atom_pos)
    data = fh.read(atom_size)
    if len(data) < 12:
        return None
    atom_size1, atom_type, magic = struct.unpack_from(">I4s4s", data)
    try:
        atom_type = atom_type.decode()
        magic = magic.decode()
        if atom_size != atom_size1 or atom_type != "free" or magic != "GPS ":
            return None
    except UnicodeDecodeError:
        return None
    return _parse_gps_payload(data[12:])


def extract_gps(mp4_path: Path) -> list[dict | None]:
    """Extract GPS data from a Novatek MP4 file.

    Returns a list with one entry per second of video.  Entries with a valid
    GPS fix are dicts ``{"lat", "lon", "datetime", "speed_kmh"}``.  Entries
    without a fix are ``None``.  The list length matches the video duration
    in seconds so that index *i* always corresponds to second *i*.
    """
    results: list[dict | None] = []
    with open(mp4_path, "rb") as fh:
        offset = 0
        while True:
            atom_size, atom_type = _get_atom_info(fh.read(8))
            if atom_size == 0:
                break
            if atom_type == "moov":
                sub_offset = offset + 8
                while sub_offset < offset + atom_size:
                    sub_atom_size, sub_atom_type = _get_atom_info(fh.read(8))
                    if sub_atom_type == "gps ":
                        gps_offset = 16 + sub_offset
                        fh.seek(gps_offset)
                        while gps_offset < sub_offset + sub_atom_size:
                            pos, size = _get_gps_atom_info(fh.read(8))
                            entry = _parse_gps_atom(pos, size, fh)
                            results.append(entry)
                            gps_offset += 8
                            fh.seek(gps_offset)
                    sub_offset += sub_atom_size
                    fh.seek(sub_offset)
            offset += atom_size
            fh.seek(offset)
    return results
