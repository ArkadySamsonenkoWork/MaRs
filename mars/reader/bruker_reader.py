import pathlib

import typing as tp
import numpy as np

import re


def _parse_key_value(line: str) -> tuple[str, str]:
    """Extracts key and value from a line, handling multiple spaces or tabs."""
    parts = re.split(r"\s{2,}|\t", line, maxsplit=1)
    key = parts[0].strip()
    value = parts[1].strip() if len(parts) > 1 else ""
    return key, value


def _parse_comma_separated_values(value):
    """Parses comma-separated values into a list of numbers."""
    values = value.replace("\n", "").split(",")
    parsed_values = []

    for v in values:
        v = v.strip()
        if v.replace(".", "", 1).isdigit():
            parsed_values.append(float(v) if "." in v else int(v))
        else:
            parsed_values.append(v)
    return parsed_values


def _parse_numeric_value(value):
    """Extracts numbers and units from a single value."""
    match = re.match(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Z/%]*)", value)
    if match:
        num, unit = match.groups()
        value = float(num) if "." in num or "e" in num.lower() else int(num)
        return {"value": value, "unit": unit} if unit else value
    return value


def _handle_dvc_lines(key, value):
    """Handles special `.DVC` metadata lines by restructuring the key."""
    key_parts = key.split()
    if len(key_parts) > 1:
        key = f"{key_parts[0]}_{key_parts[1]}"
        value = key_parts[2] if len(key_parts) > 2 else ""
    return key, value


def read_dsc(filename):
    """Reads and cleans a Bruker DSC file, returning a structured
    dictionary."""
    metadata = {}
    current_key = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("*") or line.startswith("#"):
                continue
            if current_key and re.match(r"^\d+(,\d+)*$", line):
                metadata[current_key].extend(_parse_comma_separated_values(line))
                continue

            key, value = _parse_key_value(line)
            if key.startswith(".DVC"):
                key, value = _handle_dvc_lines(key, value)
            if key.startswith("Psd") and "," in value:
                metadata[key] = _parse_comma_separated_values(value)
                current_key = key
            else:
                metadata[key] = _parse_numeric_value(value)
                current_key = None

    return metadata


def read_dta(filepath, metadata):
    """Read Bruker .DTA file with complex data for EPR spectroscopy.

    Parameters:
    - x_values: numpy array of x-axis values
    """
    endian = ">" if metadata["BSEQ"] == "BIG" else "<"
    if metadata["IRFMT"] == "D":
        dtype = np.dtype(endian + "f8")
    else:
        dtype = np.dtype(endian + "f4")
    raw_data = np.fromfile(filepath, dtype=dtype)
    if metadata["IKKF"] == "CPLX":
        raw_data = raw_data.reshape(-1, 2)
        data = raw_data[:, 0] + 1j * raw_data[:, 1]
    else:
        data = raw_data
    x_axis = np.linspace(metadata["XMIN"], metadata["XMIN"] + metadata["XWID"], metadata["XPTS"])
    return {
        "x_values": x_axis,
        "y_values": data,
    }


def _parse_par_value(value_str: str) -> tp.Any:
    """Parse values from .par files handling numbers with units."""
    value_str = value_str.strip()
    if re.fullmatch(r"[-+]?\d+", value_str):
        return int(value_str)
    if re.fullmatch(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_str):
        return float(value_str)

    match = re.match(r"^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Z%°]*)$", value_str)
    if match:
        num_str, unit = match.groups()
        num_str = num_str.strip()
        unit = unit.strip()
        try:
            num_val = float(num_str) if "." in num_str or "e" in num_str.lower() else int(num_str)
            return {"value": num_val, "unit": unit} if unit else num_val
        except ValueError:
            return {"value": value_str, "unit": unit} if unit else value_str
    return value_str


def read_par(filename: tp.Union[str, pathlib.Path]) -> dict:
    """Parse Bruker .par parameter files."""
    metadata = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("*") or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            key, value_str = parts
            metadata[key] = _parse_par_value(value_str)
    return metadata


def read_spc(filepath: tp.Union[str, pathlib.Path], metadata: dict) -> dict:
    """Read Bruker .spc binary data files."""
    data = np.fromfile(filepath, dtype="<f4")
    if "GST" in metadata and "HSW" in metadata:
        start_field = metadata["GST"]
        sweep_width = metadata["HSW"]
        x_axis = np.linspace(start_field, start_field + sweep_width, len(data))
    elif 'HCF' in metadata and 'HSW' in metadata:
        center_field = metadata['HCF']
        sweep_width = metadata['HSW']
        x_axis = np.linspace(center_field - sweep_width / 2,
                             center_field + sweep_width / 2,
                             len(data))
    else:
        x_axis = np.arange(len(data))
    return {'x_values': x_axis, 'y_values': data}


def read_bruker_spc_par_data(path: tp.Union[str, pathlib.Path]) -> tuple[dict[str, tp.Any], dict[str, np.array]]:
    """Read Bruker .par/.spc file pairs."""
    path = pathlib.Path(path)
    path_spc = path.with_suffix('.spc')
    path_par = path.with_suffix('.par')

    if not path_par.exists() or not path_spc.exists():
        raise FileNotFoundError(f"Missing .par or .spc files for base path: {path}")

    metadata = read_par(path_par)
    data = read_spc(path_spc, metadata)
    return metadata, data


def read_bruker_dsc_dta_data(path: tp.Union[str, pathlib.Path]) -> tuple[dict[str, tp.Any], dict[str, np.array]]:
    """
    :param path: path to bruker file.

    :return: metadata and the results
    """
    path = pathlib.Path(path)
    path_dta = path.with_suffix(".dta")
    path_dsc = path.with_suffix(".dsc")
    metadata = read_dsc(path_dsc)
    data = read_dta(path_dta, metadata=metadata)
    return metadata, data


def read_bruker_data(path: tp.Union[str, pathlib.Path]) -> tuple[dict[str, tp.Any], dict[str, np.array]]:
    """Read data from Bruker spectrometer using dta/dsc of pra/spc files."""
    path = pathlib.Path(path)

    if (path.with_suffix(".dsc").exists() and path.with_suffix(".dta").exists()):
        return read_bruker_dsc_dta_data(path)

    if (path.with_suffix(".DSC").exists() and path.with_suffix(".DTA").exists()):
        return read_bruker_dsc_dta_data(path)

    if (path.with_suffix(".par").exists() and path.with_suffix(".spc").exists()):
        return read_bruker_spc_par_data(path)

    if (path.with_suffix(".PAR").exists() and path.with_suffix(".SPC").exists()):
        return read_bruker_spc_par_data(path)

    raise ValueError(f"Unsupported Bruker format for path: {path}. "
                     "Missing either (.dsc + .dta) or (.par + .spc) file pairs.")

