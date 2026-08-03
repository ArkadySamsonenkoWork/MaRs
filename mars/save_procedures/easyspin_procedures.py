import itertools
import math
import pathlib
from dataclasses import dataclass

import typing as tp
import numpy as np
import torch
import scipy

from .. import particles, utils
from ..spin_model import BaseSample, SpinSystem, Interaction, MultiOrientedSample
from ..spectra_manager import BaseResSpectra, StationarySpectra

from ..serialization import serialization


class EasySpinSaverSampleDict:
    hz_to_MHz = 1e-6
    T_to_mT = 1e3
    g_easy_spin_strain_converter = 1.0

    def get_dict(self, sample: BaseSample):
        out = self._serialize_sample(sample)
        return out

    def _serialize_sample(self, sample: BaseSample) -> tp.Dict[str, tp.Any]:
        spin_system = sample.base_spin_system

        lorentz = sample.lorentz.detach().cpu().item()
        gauss = sample.gauss.detach().cpu().item()
        ham_strain = self._convert_tensor(sample.base_ham_strain)

        sys_dict = self._serialize_spin_system(spin_system)
        sys_dict["HStrain"] = ham_strain * self.hz_to_MHz
        sys_dict["lw"] = np.array([gauss, lorentz]) * self.T_to_mT

        return sys_dict

    def _serialize_spin_system(self, spin_system: SpinSystem) -> tp.Dict[str, tp.Any]:
        electrons = spin_system.electrons
        g_tensors = spin_system.g_tensors
        nuclei = spin_system.nuclei
        electron_nuclei = spin_system.electron_nuclei
        electron_electron = spin_system.electron_electron
        nuclei_nuclei = spin_system.nuclei_nuclei
        result = {
            **self._serialize_electrons(electrons),
            **self._serialize_nuclei(nuclei),
            **self._serialize_g_tensors(g_tensors),
            **self._serialize_electron_nuclei(electrons, nuclei, electron_nuclei),
            **self._serialize_electron_electron(electrons, electron_electron),
            **self._serialize_nuclei_nuclei(nuclei, nuclei_nuclei),
        }

        return result

    def _serialize_electrons(self, electrons: list[particles.Electron]):
        return {"S": [electron.spin for electron in electrons]}

    def _serialize_nuclei(self, nuclei: list[particles.Nucleus]):
        if nuclei:

            return {"Nucs": ",".join([nucleus.nucleus_str for nucleus in nuclei])}
        else:
            return {}

    def _convert_tensor(self, tensor: torch.Tensor):
        return tensor.detach().cpu().numpy().astype(np.float64)

    def _serialize_g_tensors(self, g_interactions: list[Interaction]):
        g_tensors = []
        g_frames = []
        g_strains = []
        for g_interaction in g_interactions:
            g_tensors.append(self._convert_tensor(g_interaction.components))

            frame = g_interaction.frame
            frame = self._convert_tensor(frame) if frame is not None else np.array([0.0, 0.0, 0.0])
            g_frames.append(frame)

            strain = g_interaction.strain
            g_strain = self._convert_tensor(strain) if strain is not None else np.array([0.0, 0.0, 0.0])
            g_strains.append(g_strain)

        return {"g": np.array(g_tensors),
                "gStrain": np.array(g_strains) / self.g_easy_spin_strain_converter, "gFrame": np.array(g_frames)}

    def _serialize_electron_nuclei(self,
                                   electrons,
                                   nuclei: list[particles.Nucleus],
                                   electron_nuclei: list[tuple[int, int, Interaction]]
    ):

        num_electrons = len(electrons)
        num_nuclei = len(nuclei)

        tensors = np.zeros((num_nuclei, num_electrons * 3), dtype=np.float64)
        strains = np.zeros((num_nuclei, num_electrons * 3), dtype=np.float64)
        frames = np.zeros((num_nuclei, num_electrons * 3), dtype=np.float64)

        if electron_nuclei:

            interaction_dict = {}
            for el_idx, nuc_idx, interaction in electron_nuclei:
                interaction_dict[(el_idx, nuc_idx)] = interaction

            for el_idx in range(num_electrons):
                for nuc_idx in range(num_nuclei):

                    start_pos = el_idx * 3

                    if (el_idx, nuc_idx) in interaction_dict:
                        interaction = interaction_dict[(el_idx, nuc_idx)]

                        strain = interaction.strain
                        strain = self._convert_tensor(strain) if strain else [0, 0, 0]

                        tensors[nuc_idx, start_pos:start_pos + 3] = self._convert_tensor(interaction.components)
                        frames[nuc_idx, start_pos:start_pos + 3] = self._convert_tensor(interaction.frame)
                        strains[nuc_idx, start_pos:start_pos + 3] = strain
            return {"A": np.array(tensors) * self.hz_to_MHz,
                    # "AStrain": np.array(frames),    I didn't understand the AStrain logic in Easyspin.
                    # So I disolved it. It should be one or many or what.!!
                    "AFrame": np.array(strains) * self.hz_to_MHz}
        else:
            return {}

    def _serialize_electron_electron(self,
                                     electrons,
                                     electron_electron: list[tuple[int, int, Interaction]]
                                     ):

        zfs_flag = False
        num_electrons = len(electrons)
        J_tensor = np.zeros(int(num_electrons * (num_electrons - 1) / 2), dtype=np.float64)
        dipole_tensor = np.zeros((int(num_electrons * (num_electrons - 1) / 2), 3), dtype=np.float64)
        dipole_frame_tensor = np.zeros((int(num_electrons * (num_electrons - 1) / 2), 3), dtype=np.float64)

        zfs_array = np.zeros((int(num_electrons), 2), dtype=np.float64)
        zfs_frame = np.zeros((int(num_electrons), 3), dtype=np.float64)
        zfz_strain = np.zeros((int(num_electrons), 2), dtype=np.float64)

        coupling_dict = {}
        zero_field = {}
        for el_idx_1, el_idx_2, interaction in electron_electron:
            if el_idx_1 != el_idx_2:
                coupling_dict[(min(el_idx_1, el_idx_2), max(el_idx_1, el_idx_2))] = interaction
            else:
                zero_field[(el_idx_1, el_idx_2)] = interaction
        position_zfs = 0
        position_dip_dip = 0
        for el_idx_1 in range(num_electrons):
            for el_idx_2 in range(el_idx_1, num_electrons):

                if (el_idx_1, el_idx_2) in coupling_dict:
                    interaction = coupling_dict[(el_idx_1, el_idx_2)]

                    tensor = self._convert_tensor(interaction.components)
                    J = np.mean(tensor)
                    dip = tensor - J

                    J_tensor[position_dip_dip] = J
                    dipole_tensor[position_dip_dip] = dip
                    dipole_frame_tensor[position_dip_dip] = self._convert_tensor(interaction.frame)

                if (el_idx_1, el_idx_2) in zero_field:
                    zfs_flag = True
                    interaction = zero_field[(el_idx_1, el_idx_2)]

                    tensor = self._convert_tensor(interaction.components)
                    frame = self._convert_tensor(interaction.frame)

                    strain = interaction.strain
                    strain = self._convert_tensor(strain) if strain is not None else [0, 0]

                    D = 3 * tensor[-1] / 2
                    E = (tensor[0] - tensor[1]) / 2

                    zfs_array[position_zfs] = np.array([D, E])
                    zfs_frame[position_zfs] = frame

                    D_str = strain[0]
                    E_str = strain[1]

                    zfz_strain[position_zfs] = np.array([D_str, E_str])

                if el_idx_1 == el_idx_2:
                    position_zfs += 1
                else:
                    position_dip_dip += 1

        out_dict = {}
        if zfs_flag:
            out_dict = {"D": np.array(zfs_array) * self.hz_to_MHz, "DFrame": np.array(zfs_frame),
                        "DStrain": np.array(zfz_strain) * self.hz_to_MHz}

        dipole_tensor = np.array(dipole_tensor)

        out_dict["dip"] = np.array(dipole_tensor) * self.hz_to_MHz
        out_dict["J"] = np.array(J_tensor) * self.hz_to_MHz
        out_dict["eeFrame"] = np.array(dipole_frame_tensor)

        return out_dict

    def _serialize_nuclei_nuclei(self, nuclei, nuclei_nuclei: list[tuple[int, int, Interaction]]):

        out_dict = {}
        num_nuclei = len(nuclei)
        if num_nuclei and nuclei_nuclei:
            Q_array = np.zeros(int(num_nuclei * (num_nuclei - 1) / 2), dtype=np.float64)
            frame_array = np.zeros((int(num_nuclei * (num_nuclei - 1) / 2), 3), dtype=np.float64)

            coupling_dict = {}
            for nuc_idx_1, nuc_idx_2, interaction in nuclei_nuclei:
                if nuc_idx_1 != nuc_idx_2:
                    coupling_dict[(nuc_idx_1, nuc_idx_2)] = interaction

            position = 0
            for nuc_idx_1 in range(num_nuclei):
                for nuc_idx_2 in range(nuc_idx_1 + 1, num_nuclei):

                    if (nuc_idx_1, nuc_idx_2) in coupling_dict:
                        interaction = coupling_dict[(nuc_idx_1, nuc_idx_2)]

                        Q_array[position] = self._convert_tensor(interaction.components)
                        frame_array[position] = self._convert_tensor(interaction.frame)

                    position += 1
            out_dict["Q"] = np.array(Q_array) * self.hz_to_MHz
            out_dict["QFrame"] = np.array(frame_array)

        return out_dict


class EasySpinCreatorDict:
    """Converts a SpectraCreator to an EasySpin-compatible Exp dictionary."""
    hz_to_ghz = 1e-9
    def get_dict(self, creator: BaseResSpectra) -> tp.Dict[str, tp.Any]:
        """Convert creator to EasySpin Exp dictionary.

        :param creator: The spectra creator instance.
        :return: Dictionary containing EasySpin Exp parameters.
        """
        temperature = creator.intensity_calculator.temperature
        temperature = temperature if temperature is None else np.array(temperature).astype(np.float64)
        frequency = creator.resonance_parameter.detach().cpu().numpy().astype(np.float64)
        return {"Temperature": temperature, "mwFreq": frequency * self.hz_to_ghz}


def parse_field(field: tp.Optional[torch.Tensor], freq: tp.Optional[torch.Tensor] = None) ->\
        tp.Optional[serialization.ExperimentalParameters]:
    """Parse a magnetic field tensor into FieldParameters.

    :param field: The magnetic field tensor in Tesla. Shape (N,).
    :param freq: The resonance frequency tensor.
    :return: A FieldParameters instance or None if field is None.
    """
    if field is None:
        return None

    return serialization.ExperimentalParameters(
        min_pos=torch.min(field),
        max_pos=torch.max(field),
        num_points=int(field.shape[0]),
        resonance_parameter=freq
    )


def save_easyspin(filepath: str, sample: tp.Optional[BaseSample], spectra_creator: tp.Optional[BaseResSpectra],
                  field: tp.Optional[torch.Tensor]):
    """Save data in EasySpin-compatible MATLAB format.

    :param filepath: The file path to save the data.
    :param sample: BaseSample instance.
    :param spectra_creator: SpectraCreator instance.
    :param field: The magnetic field tensor in Tesla units.
    """
    T_to_mT = 1e3
    sample_dict = EasySpinSaverSampleDict().get_dict(sample) if sample is not None else {}
    creator_dict = EasySpinCreatorDict().get_dict(spectra_creator) if spectra_creator is not None else {}
    out = {"Sys": sample_dict, "Exp": creator_dict}
    field_params = parse_field(field)

    if field_params:
        out["Exp"]["Range"] = np.array(
            [field_params.min_pos.detach().cpu().numpy(),
             field_params.max_pos.detach().cpu().numpy()],
             dtype=np.float64) * T_to_mT
        out["Exp"]["nPoints"] = field_params.num_points
    scipy.io.savemat(filepath, out, oned_as="row")


class EasySpinLoaderSampleDict:
    """Reconstructs BaseSample objects from EasySpin dictionary representations."""
    g_easy_spin_strain_converter = 1.0

    def load_easy_spin(self, sample_dict: dict[str, tp.Any],
                       device: torch.device = torch.device("cpu"),
                       dtype: torch.dtype = torch.float32,
                       ) -> BaseSample:
        MHz_to_hz = 1e6
        mT_to_T = 1e-3
        lorentz_conversion = math.sqrt(2 * math.log(2))
        gauss_conversion = math.sqrt(3)
        gauss = None
        lorentz = None

        if "lw" in sample_dict:
            lw = sample_dict.get("lw", np.array([[0.0, 0.0]]))
            gauss = torch.tensor(lw[0][0] * mT_to_T, dtype=dtype, device=device)
            lorentz = torch.tensor(lw[0][1] * mT_to_T, dtype=dtype, device=device)
        elif "lwpp" in sample_dict:
            lwpp = sample_dict.get("lwpp", np.array([[0.0, 0.0]]))
            gauss = torch.tensor(lwpp[0][0] * mT_to_T, dtype=dtype, device=device) * gauss_conversion
            lorentz = torch.tensor(lwpp[0][1] * mT_to_T, dtype=dtype, device=device) * lorentz_conversion

        spin_system = self._deserialize_easyspin_spin_system(sample_dict, device=device, dtype=dtype)

        if "HStrain" in sample_dict:
            ham_strain = torch.tensor(sample_dict["HStrain"] * MHz_to_hz, dtype=dtype, device=device)[0]
        else:
            ham_strain = None

        return MultiOrientedSample(
            base_spin_system=spin_system, gauss=gauss, lorentz=lorentz, ham_strain=ham_strain,
            dtype=dtype, device=device
        )

    def _deserialize_easyspin_spin_system(
            self, sys_dict: dict[str, tp.Any],
            device: torch.device = torch.device("cpu"),
            dtype: torch.dtype = torch.float32,
    ) -> SpinSystem:
        complex_dtype = utils.float_to_complex_dtype(dtype)

        MHz_to_hz = 1e6
        S_list = sys_dict.get("S", [[]])
        electrons = [
            particles.Electron(spin=s, device=device, complex_dtype=complex_dtype) for s in itertools.chain(*S_list)
        ]

        nuclei = []
        if "Nucs" in sys_dict:
            for nuc_str in sys_dict["Nucs"].tolist()[0].split(","):
                nuclei.append(particles.Nucleus(nuc_str, device=device, complex_dtype=complex_dtype))

        g_tensors = []
        if "g" in sys_dict:
            g_components = sys_dict["g"]
            g_strains = sys_dict.get("gStrain", np.zeros_like(g_components)) * self.g_easy_spin_strain_converter
            g_frames = sys_dict.get("gFrame", np.zeros_like(g_components))

            for i in range(len(g_components)):
                components = torch.tensor(g_components[i], dtype=dtype, device=device)
                strain = torch.tensor(g_strains[i], dtype=dtype, device=device) if np.any(g_strains[i]) else None
                frame = torch.tensor(g_frames[i], dtype=dtype, device=device) if np.any(g_frames[i]) else None
                g_tensors.append(Interaction(components=components, strain=strain, frame=frame, dtype=dtype, device=device))

        spin_system = SpinSystem(electrons=electrons, nuclei=nuclei, g_tensors=g_tensors, dtype=dtype, device=device)

        if "A" in sys_dict:
            A_tensor = sys_dict["A"] * MHz_to_hz
            A_strains = sys_dict.get("AStrain", np.zeros_like(A_tensor))
            A_frames = sys_dict.get("AFrame", np.zeros_like(A_tensor))
            electron_nuclei = []
            for el_idx in range(len(electrons)):
                for nuc_idx in range(len(nuclei)):
                    start_pos = nuc_idx * 3
                    components = A_tensor[nuc_idx, start_pos:start_pos + 3]
                    if np.any(components):
                        strain_vals = A_strains[nuc_idx, start_pos:start_pos + 3] * MHz_to_hz
                        frame_vals = A_frames[nuc_idx, start_pos:start_pos + 3]
                        strain = torch.tensor(strain_vals, dtype=dtype, device=device) if np.any(strain_vals) else None
                        frame = torch.tensor(frame_vals, dtype=dtype, device=device) if np.any(frame_vals) else None
                        interaction = Interaction(components=torch.tensor(components, dtype=dtype, device=device),
                                                  strain=strain, frame=frame, dtype=dtype, device=device)
                        electron_nuclei.append((el_idx, nuc_idx, interaction))
            spin_system.electron_nuclei = electron_nuclei

        electron_electron = []
        if "J" in sys_dict or "dip" in sys_dict or "D" in sys_dict:
            self._add_easyspin_electron_electron(sys_dict, spin_system, electron_electron, device, dtype)

        if "Q" in sys_dict:
            Q_tensor = sys_dict["Q"] * MHz_to_hz
            Q_frames = sys_dict.get("QFrame", np.zeros((len(Q_tensor), 3)))
            nuclei_nuclei = []
            position = 0
            for nuc_idx_1 in range(len(nuclei)):
                for nuc_idx_2 in range(nuc_idx_1 + 1, len(nuclei)):
                    if position < len(Q_tensor) and Q_tensor[position] != 0:
                        components = torch.tensor(Q_tensor[position], dtype=dtype, device=device)
                        frame = torch.tensor(Q_frames[position], dtype=dtype, device=device) if np.any(
                            Q_frames[position]) else None
                        interaction = Interaction(components=components, frame=frame, dtype=dtype, device=device)
                        nuclei_nuclei.append((nuc_idx_1, nuc_idx_2, interaction))
                    position += 1
            spin_system.nuclei_nuclei = nuclei_nuclei

        return spin_system

    def _add_easyspin_electron_electron(self, sys_dict: dict[str, tp.Any], spin_system: SpinSystem,
                                        electron_electron: list[tuple[int, int, Interaction]],
                                        device: torch.device = torch.device("cpu"),
                                        dtype: torch.dtype = torch.float32,
                                        ):
        MHz_to_hz = 1e6
        num_electrons = len(spin_system.electrons)
        J_tensor = sys_dict.get("J", 0.0)
        dip_tensor = sys_dict.get("dip", np.zeros((len(J_tensor), 3)) if len(J_tensor) > 0 else np.array([]))
        ee_frames = sys_dict.get("eeFrame", np.zeros_like(dip_tensor))
        D_tensor = sys_dict.get("D", np.array([]))
        D_frames = sys_dict.get("DFrame", np.zeros((len(D_tensor), 3)) if len(D_tensor) > 0 else np.array([]))
        D_strains = sys_dict.get("DStrain", np.zeros_like(D_tensor))

        position = 0
        for el_idx_1 in range(num_electrons):
            for el_idx_2 in range(el_idx_1 + 1, num_electrons):
                if position < len(J_tensor):
                    J = J_tensor[position] * MHz_to_hz
                    dip = dip_tensor[position] * MHz_to_hz if position < len(dip_tensor) else np.zeros(3)
                    frame = ee_frames[position] if position < len(ee_frames) else np.zeros(3)
                    components = torch.tensor(J + dip, device=device, dtype=dtype)
                    frame_tensor = torch.tensor(frame, device=device, dtype=dtype) if np.any(frame) else None
                    interaction = Interaction(
                        components=components, frame=frame_tensor, device=device, dtype=dtype)
                    electron_electron.append((el_idx_1, el_idx_2, interaction))

                if position < len(D_tensor) and np.any(D_tensor[position]):
                    D_val = D_tensor[position] * MHz_to_hz
                    D_frame = D_frames[position] if position < len(D_frames) else np.zeros(3)
                    D_strain = D_strains[position] * MHz_to_hz if position < len(D_strains) else np.zeros(3)
                    components = torch.tensor(D_val, device=device, dtype=dtype)
                    frame_tensor = torch.tensor(D_frame, device=device, dtype=dtype) if np.any(D_frame) else None
                    strain_tensor = torch.tensor(D_strain, device=device, dtype=dtype) if np.any(D_strain) else None
                    interaction = Interaction(
                        components=components, frame=frame_tensor, strain=strain_tensor, device=device, dtype=dtype)
                    electron_electron.append((el_idx_1, el_idx_1, interaction))
                position += 1
        spin_system.electron_electron = electron_electron


def load_mat_file(filepath: tp.Union[str, pathlib.Path]):
    """Load and properly parse MATLAB .mat file with structured arrays.

    :param filepath: Path to the .mat file.
    :return: Dictionary containing parsed MATLAB structures.
    """
    data = scipy.io.loadmat(filepath)
    clean_data = {k: v for k, v in data.items() if not k.startswith("__")}
    result = {}
    for key, value in clean_data.items():
        if hasattr(value, "dtype") and value.dtype.names:
            structured_dict = {}
            if value.size > 0:
                first_element = value.flat[0]
                for field_name in value.dtype.names:
                    field_data = first_element[field_name]
                    structured_dict[field_name] = field_data
            result[key] = structured_dict
        else:
            result[key] = value
    return result


def load_easyspin_creator(sample: MultiOrientedSample,
                          temperature: tp.Union[float, torch.Tensor],
                          freq: tp.Union[float, torch.Tensor],
                          device: torch.device = torch.device("cpu"),
                          dtype: torch.dtype = torch.float32
                          ) -> StationarySpectra:
    """Load creator from EasySpin format."""
    return StationarySpectra(sample=sample, temperature=temperature,
                             freq=torch.tensor(freq, dtype=dtype, device=device))


@dataclass
class _EasySpinParsedData:
    """Internal helper to hold parsed EasySpin data before converting to Session."""
    sample: BaseSample
    experimental_parameters: tp.Optional[serialization.ExperimentalParameters]
    polarization: tp.Optional[serialization.PolarizationParameters]


def _parse_easyspin_core(filepath: tp.Union[str, pathlib.Path],
                         device: torch.device,
                         dtype: torch.dtype) -> _EasySpinParsedData:
    """Core parsing logic for EasySpin MATLAB files."""
    data = load_mat_file(filepath)
    ghz_to_hz = 1e9

    if "Sys" not in data:
        raise KeyError("EasySpin file must contain a 'Sys' structure.")

    sample_loader = EasySpinLoaderSampleDict()
    sample = sample_loader.load_easy_spin(data["Sys"], device=device, dtype=dtype)

    exp_parameters = None
    polarization = None

    if "Exp" in data:
        exp_data = data["Exp"]

        if "Range" in exp_data and "nPoints" in exp_data:
            mT_to_T = 1e-3
            field_range = exp_data["Range"] * mT_to_T
            n_points = int(exp_data["nPoints"][0][0])

            mw_freq_raw = exp_data.get("mwFreq", None)
            if mw_freq_raw is not None and hasattr(mw_freq_raw, "size") and mw_freq_raw.size == 0:
                mw_freq_raw = None
            mw_freq_tensor = torch.as_tensor(mw_freq_raw, device=device,
                                             dtype=dtype) * ghz_to_hz if mw_freq_raw is not None else None

            exp_parameters = serialization.ExperimentalParameters(
                min_pos=torch.as_tensor(field_range[0][0], device=device, dtype=dtype),
                max_pos=torch.as_tensor(field_range[0][1], device=device, dtype=dtype),
                num_points=n_points,
                resonance_parameter=mw_freq_tensor[0][0]
            )

        temp_raw = exp_data.get("Temperature", None)
        if temp_raw is not None and hasattr(temp_raw, "size") and temp_raw.size == 0:
            temp_raw = None
        if temp_raw is not None:
            temp_tensor = torch.as_tensor(temp_raw, device=device, dtype=dtype)
            polarization = serialization.PolarizationParameters(temperature=temp_tensor)

    return _EasySpinParsedData(
        sample=sample,
        experimental_parameters=exp_parameters,
        polarization=polarization
    )


def load_easyspin_serialized(filepath: tp.Union[str, pathlib.Path],
                             device: torch.device = torch.device("cpu"),
                             dtype: torch.dtype = torch.float32) -> serialization.SerializedMarsSession:
    """Load serialized data from an EasySpin-compatible MATLAB file.

    :param filepath: The file path to load data from.
    :param device: Torch device to place the tensors on.
    :param dtype: Torch data type for the tensors.
    :return: A SerializedMarsSession dataclass.
    """
    parsed_data = _parse_easyspin_core(filepath, device=device, dtype=dtype)
    return serialization.SerializedMarsSession(
        sample=serialization.SerializedSample.from_mars_sample(parsed_data.sample),
        experimental_parameters=parsed_data.experimental_parameters,
        polarization=parsed_data.polarization
    )


def load_easyspin(filepath: tp.Union[str, pathlib.Path],
                  device: torch.device = torch.device("cpu"),
                  dtype: torch.dtype = torch.float32) -> serialization.MarsSession:
    """Load data from an EasySpin-compatible MATLAB file.

    :param filepath: The file path to load data from.
    :param device: Torch device to place the tensors on.
    :param dtype: Torch data type for the tensors.
    :return: A fully instantiated MarsSession.
    """
    serialized_session = load_easyspin_serialized(filepath, device=device, dtype=dtype)
    return serialized_session.to_experiment(device=device, dtype=dtype)
