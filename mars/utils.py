import typing as tp
import math

import torch

from . import constants


_SUPPORTED_EULER_CONVENTIONS = {"zyz", "xyz", "xzy", "yxz", "yzx", "zxy", "zyx"}


def apply_expanded_rotations(R: torch.Tensor, T: torch.Tensor):
    """Transform rank-2 tensors according to T_target = R T_source R^T.

    ``R`` is assumed to transform vector coordinates from a source frame to a
    target frame,

        v_target = R @ v_source.

    The corresponding representation of a rank-2 tensor transforms as

        T_target = R @ T_source @ R.T.

    In this interpretation the physical tensor is not rotated; the same tensor
    is being expressed in another coordinate frame. This is a passive coordinate
    transformation.

    This function expands independent batch dimensions of ``R`` and ``T`` so
    that every requested rotation is applied to every input tensor.

    :param R: Rotation matrices mapping source-frame vector coordinates to
        target-frame coordinates. Shape ``[*rotation_dims, 3, 3]``.
    :param T: Rank-2 tensors expressed in the source frame.
        Shape ``[..., 3, 3]``.
    :return: Tensors expressed in the target frame, with shape
        ``[..., *rotation_dims, 3, 3]``.
    """
    R_batch_shape = R.shape[:-2]
    T_batch_shape = T.shape[:-2]

    R_expanded = R.view(*([1] * len(T_batch_shape)), *R_batch_shape, 3, 3)
    T_expanded = T.view(*T_batch_shape, *([1] * len(R_batch_shape)), 3, 3)
    RT = torch.matmul(R_expanded, T_expanded)

    return torch.matmul(RT, R_expanded.transpose(-1, -2))


def apply_single_rotation(R: torch.Tensor, T: torch.Tensor):
    """Transform rank-2 tensors according to T_target = R T_source R^T.

    ``R`` is assumed to transform vector coordinates from a source frame to a
    target frame,
        v_target = R @ v_source.

    Therefore a rank-2 tensor transforms as
        T_target = R @ T_source @ R.T.
    For rotation matrices produced by ``CrystalMesh``,
        v_lab = R @ v_mol
    and consequently
        T_lab = R @ T_mol @ R.T.

    Here ``T_mol`` and ``T_lab`` represent the same physical tensor in
    molecular/crystal and laboratory coordinates, respectively. In this use
    case the operation is a passive change of coordinates.

    :param R: The rotation matrices with shape [..., 3, 3].
    :param T: The tensor to be rotated with shape [..., 3, 3].
    :return: The rotated tensors with shape [..., 3, 3].
    """
    RT = torch.matmul(R, T)
    rotated_T = torch.matmul(RT, R.transpose(-2, -1))

    return rotated_T


def calculate_deriv_max(g_tensors_el: torch.Tensor, g_factors_nuc: torch.Tensor,
                        el_numbers: torch.Tensor, nuc_numbers: torch.Tensor) -> torch.Tensor:
    """Calculate the maximum value of the energy derivatives with respect to
    magnetic field.

    It is assumed that B has direction along z-axis
    :param g_tensors_el: g-tensors of electron spins. The shape is [..., 3, 3]
    :param g_factors_nuc: g-factors of the nuclei spins. The shape is [...]
    :param el_numbers: electron spin quantum numbers
    :param nuc_numbers: nuclei spins quantum numbers
    :return: the maximum value of the energy derivatives with respect to magnetic field
    """
    electron_contrib = (constants.BOHR / constants.PLANCK) * g_tensors_el[..., :, 0].sum(dim=-1) * el_numbers
    nuclear_contrib = (constants.NUCLEAR_MAGNETRON / constants.PLANCK) * g_factors_nuc * nuc_numbers
    return nuclear_contrib + electron_contrib


def euler_angles_to_matrix(angles: torch.Tensor, convention: str = "zyz"):
    """Convert Euler angles to rotation matrices.

    The convention string is written with lowercase letters because it specifies
    intrinsic rotations about the moving molecular axes. For
    ``convention="abc"``, the angles ``[alpha, beta, gamma]`` describe the
    physical rotation sequence

    ``a(alpha) -> b'(beta) -> c''(gamma)``.

    Here ``a`` is an axis of the initial molecular frame, ``b'`` is the
    corresponding molecular axis after the first rotation, and ``c''`` is the
    molecular axis after the first two rotations.

    The resulting rotation matrix is

    ``R = R_a(alpha) @ R_b(beta) @ R_c(gamma)``.

    The same final orientation can be obtained using extrinsic rotations about
    the fixed laboratory axes. These axes are written with capital letters.
    The equivalent physical sequence is reversed:

    ``C(gamma) -> B(beta) -> A(alpha)``.

    Note the distinction between physical rotation order and matrix-factor
    order. With column vectors, a new rotation about a fixed laboratory axis is
    multiplied from the left. Therefore the extrinsic physical sequence

    ``C(gamma) -> B(beta) -> A(alpha)``

    also gives

    ``R = R_A(alpha) @ R_B(beta) @ R_C(gamma)``.

    In contrast, intrinsic rotations about moving molecular axes are
    post-multiplied, so their physical sequence

    ``a(alpha) -> b'(beta) -> c''(gamma)``

    directly produces the same written matrix product.

    For the default ``"zyz"`` convention:

    intrinsic molecular rotations:
        ``z(alpha) -> y'(beta) -> z''(gamma)``

    equivalent extrinsic rotations about fixed laboratory axes:
        ``Z(gamma) -> Y(beta) -> Z(alpha)``

    both give:
        ``R = R_Z(alpha) @ R_Y(beta) @ R_Z(gamma)``.

    The returned matrix transforms coordinates from the molecular frame to the
    laboratory frame according to

    ``v_lab = R @ v_mol``.

    The angle order is important: the fixed-laboratory rotations occur in the
    reverse physical order.
    Supported conventions and their equivalent fixed-laboratory sequences are.

    - ``"zyz"``:
      molecular:   ``z(alpha) -> y'(beta) -> z''(gamma)``
      laboratory:  ``Z(gamma) -> Y(beta) -> Z(alpha)``
      matrix:      ``R_Z(alpha) @ R_Y(beta) @ R_Z(gamma)``

    - ``"xyz"``:
      molecular:   ``x(alpha) -> y'(beta) -> z''(gamma)``
      laboratory:  ``Z(gamma) -> Y(beta) -> X(alpha)``
      matrix:      ``R_X(alpha) @ R_Y(beta) @ R_Z(gamma)``

    - ``"xzy"``:
      molecular:   ``x(alpha) -> z'(beta) -> y''(gamma)``
      laboratory:  ``Y(gamma) -> Z(beta) -> X(alpha)``
      matrix:      ``R_X(alpha) @ R_Z(beta) @ R_Y(gamma)``

    - ``"yxz"``:
      molecular:   ``y(alpha) -> x'(beta) -> z''(gamma)``
      laboratory:  ``Z(gamma) -> X(beta) -> Y(alpha)``
      matrix:      ``R_Y(alpha) @ R_X(beta) @ R_Z(gamma)``

    - ``"yzx"``:
      molecular:   ``y(alpha) -> z'(beta) -> x''(gamma)``
      laboratory:  ``X(gamma) -> Z(beta) -> Y(alpha)``
      matrix:      ``R_Y(alpha) @ R_Z(beta) @ R_X(gamma)``

    - ``"zxy"``:
      molecular:   ``z(alpha) -> x'(beta) -> y''(gamma)``
      laboratory:  ``Y(gamma) -> X(beta) -> Z(alpha)``
      matrix:      ``R_Z(alpha) @ R_X(beta) @ R_Y(gamma)``

    - ``"zyx"``:
      molecular:   ``z(alpha) -> y'(beta) -> x''(gamma)``
      laboratory:  ``X(gamma) -> Y(beta) -> Z(alpha)``
      matrix:      ``R_Z(alpha) @ R_Y(beta) @ R_X(gamma)``.

    :param angles: Euler angles in radians with shape ``[..., 3]``.
        The last dimension contains ``[alpha, beta, gamma]``. ``alpha`` belongs
        to the first axis in ``convention``, ``beta`` to the second moving axis,
        and ``gamma`` to the third moving axis.
    :param convention: Intrinsic molecular-axis rotation convention. Default is
        ``"zyz"``.
    :return: Rotation matrices with shape ``[..., 3, 3]``.
    """
    if not isinstance(angles, torch.Tensor):
        angles = torch.tensor(angles, dtype=torch.float32)

    if angles.shape[-1] != 3:
        raise ValueError(
            f"Expected angles with shape [..., 3], got {angles.shape}."
        )

    convention = convention.lower()

    if convention not in _SUPPORTED_EULER_CONVENTIONS:
        raise ValueError(
            f"Unsupported convention: {convention!r}. "
            f"Supported conventions are {sorted(_SUPPORTED_EULER_CONVENTIONS)}."
        )

    batch_shape = angles.shape[:-1]
    angles = angles.reshape(-1, 3)

    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    ca, cb, cg = cos_angles[:, 0], cos_angles[:, 1], cos_angles[:, 2]
    sa, sb, sg = sin_angles[:, 0], sin_angles[:, 1], sin_angles[:, 2]

    R = torch.zeros(
        angles.shape[0],
        3,
        3,
        device=angles.device,
        dtype=angles.dtype,
    )

    if convention == "zyz":
        # Intrinsic molecular:
        # z(alpha) -> y'(beta) -> z''(gamma)
        #
        # Equivalent fixed laboratory:
        # Z(gamma) -> Y(beta) -> Z(alpha)
        #
        # Matrix:
        # R = R_Z(alpha) @ R_Y(beta) @ R_Z(gamma)

        R[:, 0, 0] = ca * cb * cg - sa * sg
        R[:, 0, 1] = -ca * cb * sg - sa * cg
        R[:, 0, 2] = ca * sb

        R[:, 1, 0] = sa * cb * cg + ca * sg
        R[:, 1, 1] = -sa * cb * sg + ca * cg
        R[:, 1, 2] = sa * sb

        R[:, 2, 0] = -sb * cg
        R[:, 2, 1] = sb * sg
        R[:, 2, 2] = cb

    elif convention == "xyz":
        # Intrinsic molecular:
        # x(alpha) -> y'(beta) -> z''(gamma)
        #
        # Equivalent fixed laboratory:
        # Z(gamma) -> Y(beta) -> X(alpha)
        #
        # Matrix:
        # R = R_X(alpha) @ R_Y(beta) @ R_Z(gamma)

        R[:, 0, 0] = cb * cg
        R[:, 0, 1] = -cb * sg
        R[:, 0, 2] = sb

        R[:, 1, 0] = sa * sb * cg + ca * sg
        R[:, 1, 1] = -sa * sb * sg + ca * cg
        R[:, 1, 2] = -sa * cb

        R[:, 2, 0] = sa * sg - ca * sb * cg
        R[:, 2, 1] = sa * cg + ca * sb * sg
        R[:, 2, 2] = ca * cb

    elif convention == "xzy":
        # Intrinsic molecular:
        # x(alpha) -> z'(beta) -> y''(gamma)
        #
        # Equivalent fixed laboratory:
        # Y(gamma) -> Z(beta) -> X(alpha)
        #
        # Matrix:
        # R = R_X(alpha) @ R_Z(beta) @ R_Y(gamma)

        R[:, 0, 0] = cb * cg
        R[:, 0, 1] = -sb
        R[:, 0, 2] = cb * sg

        R[:, 1, 0] = sa * sg + ca * sb * cg
        R[:, 1, 1] = ca * cb
        R[:, 1, 2] = -sa * cg + ca * sb * sg

        R[:, 2, 0] = sa * sb * cg - ca * sg
        R[:, 2, 1] = sa * cb
        R[:, 2, 2] = sa * sb * sg + ca * cg

    elif convention == "yxz":
        # Intrinsic molecular:
        # y(alpha) -> x'(beta) -> z''(gamma)
        #
        # Equivalent fixed laboratory:
        # Z(gamma) -> X(beta) -> Y(alpha)
        #
        # Matrix:
        # R = R_Y(alpha) @ R_X(beta) @ R_Z(gamma)

        R[:, 0, 0] = ca * cg + sa * sb * sg
        R[:, 0, 1] = sa * sb * cg - ca * sg
        R[:, 0, 2] = sa * cb

        R[:, 1, 0] = cb * sg
        R[:, 1, 1] = cb * cg
        R[:, 1, 2] = -sb

        R[:, 2, 0] = -sa * cg + ca * sb * sg
        R[:, 2, 1] = sa * sg + ca * sb * cg
        R[:, 2, 2] = ca * cb

    elif convention == "yzx":
        # Intrinsic molecular:
        # y(alpha) -> z'(beta) -> x''(gamma)
        #
        # Equivalent fixed laboratory:
        # X(gamma) -> Z(beta) -> Y(alpha)
        #
        # Matrix:
        # R = R_Y(alpha) @ R_Z(beta) @ R_X(gamma)

        R[:, 0, 0] = ca * cb
        R[:, 0, 1] = sa * sg - ca * sb * cg
        R[:, 0, 2] = sa * cg + ca * sb * sg

        R[:, 1, 0] = sb
        R[:, 1, 1] = cb * cg
        R[:, 1, 2] = -cb * sg

        R[:, 2, 0] = -sa * cb
        R[:, 2, 1] = sa * sb * cg + ca * sg
        R[:, 2, 2] = -sa * sb * sg + ca * cg

    elif convention == "zxy":
        # Intrinsic molecular:
        # z(alpha) -> x'(beta) -> y''(gamma)
        #
        # Equivalent fixed laboratory:
        # Y(gamma) -> X(beta) -> Z(alpha)
        #
        # Matrix:
        # R = R_Z(alpha) @ R_X(beta) @ R_Y(gamma)

        R[:, 0, 0] = ca * cg - sa * sb * sg
        R[:, 0, 1] = -sa * cb
        R[:, 0, 2] = ca * sg + sa * sb * cg

        R[:, 1, 0] = sa * cg + ca * sb * sg
        R[:, 1, 1] = ca * cb
        R[:, 1, 2] = sa * sg - ca * sb * cg

        R[:, 2, 0] = -cb * sg
        R[:, 2, 1] = sb
        R[:, 2, 2] = cb * cg

    elif convention == "zyx":
        # Intrinsic molecular:
        # z(alpha) -> y'(beta) -> x''(gamma)
        #
        # Equivalent fixed laboratory:
        # X(gamma) -> Y(beta) -> Z(alpha)
        #
        # Matrix:
        # R = R_Z(alpha) @ R_Y(beta) @ R_X(gamma)

        R[:, 0, 0] = ca * cb
        R[:, 0, 1] = -sa * cg + ca * sb * sg
        R[:, 0, 2] = sa * sg + ca * sb * cg

        R[:, 1, 0] = sa * cb
        R[:, 1, 1] = sa * sb * sg + ca * cg
        R[:, 1, 2] = sa * sb * cg - ca * sg

        R[:, 2, 0] = -sb
        R[:, 2, 1] = cb * sg
        R[:, 2, 2] = cb * cg

    return R.view(*batch_shape, 3, 3)


def rotation_matrix_to_euler_angles(
    R: torch.Tensor,
    convention: str = "zyz",
) -> torch.Tensor:
    """Convert rotation matrices to Euler angles.

    The returned angles follow the same convention as
    :func:`euler_angles_to_matrix`. The convention string is written with
    lowercase letters because it specifies rotations about the *moving molecular
    axes*. For ``convention="abc"`` the returned vector is
    ``[alpha, beta, gamma]`` and means the intrinsic sequence

    ``a(alpha) -> b'(beta) -> c''(gamma)``.

    Here ``a`` is an axis of the initial molecular frame, ``b'`` is the molecular
    axis after the first rotation, and ``c''`` is the molecular axis after the
    first two rotations. The corresponding rotation matrix is

    ``R = R_a(alpha) @ R_b(beta) @ R_c(gamma)``.

    The same final orientation can be described using extrinsic rotations about
    the fixed laboratory axes, written with capital letters. Because fixed-axis
    rotations act right-to-left in the matrix product, the equivalent physical
    sequence is

    ``C(gamma) -> B(beta) -> A(alpha)``.

    Thus the intrinsic molecular and extrinsic laboratory descriptions use the
    same three angles, but in opposite physical order.

    For the default ``"zyz"`` convention this gives the usual equivalence

    ``zy'z'' intrinsic:  z(alpha) -> y'(beta) -> z''(gamma)``

    ``ZYZ extrinsic:     Z(gamma) -> Y(beta) -> Z(alpha)``

    with

    ``R = R_Z(alpha) @ R_Y(beta) @ R_Z(gamma)``.

    In this ZYZ case, ``alpha`` fixes the azimuth of the molecular ``z`` axis in
    the laboratory ``XY`` plane, ``beta`` is the polar tilt of molecular ``z``
    away from laboratory ``Z``, and ``gamma`` is the final twist about molecular
    ``z''``. In the equivalent fixed-axis description, ``gamma`` is applied
    first about laboratory ``Z`` and ``alpha`` is applied last about laboratory
    ``Z``.

    The angle order is important: the fixed-laboratory rotations occur in the
    reverse physical order.
    Supported conventions and their equivalent fixed-laboratory sequences are.

    - ``"zyz"``:
      molecular:   ``z(alpha) -> y'(beta) -> z''(gamma)``
      laboratory:  ``Z(gamma) -> Y(beta) -> Z(alpha)``
      matrix:      ``R_Z(alpha) @ R_Y(beta) @ R_Z(gamma)``

    - ``"xyz"``:
      molecular:   ``x(alpha) -> y'(beta) -> z''(gamma)``
      laboratory:  ``Z(gamma) -> Y(beta) -> X(alpha)``
      matrix:      ``R_X(alpha) @ R_Y(beta) @ R_Z(gamma)``

    - ``"xzy"``:
      molecular:   ``x(alpha) -> z'(beta) -> y''(gamma)``
      laboratory:  ``Y(gamma) -> Z(beta) -> X(alpha)``
      matrix:      ``R_X(alpha) @ R_Z(beta) @ R_Y(gamma)``

    - ``"yxz"``:
      molecular:   ``y(alpha) -> x'(beta) -> z''(gamma)``
      laboratory:  ``Z(gamma) -> X(beta) -> Y(alpha)``
      matrix:      ``R_Y(alpha) @ R_X(beta) @ R_Z(gamma)``

    - ``"yzx"``:
      molecular:   ``y(alpha) -> z'(beta) -> x''(gamma)``
      laboratory:  ``X(gamma) -> Z(beta) -> Y(alpha)``
      matrix:      ``R_Y(alpha) @ R_Z(beta) @ R_X(gamma)``

    - ``"zxy"``:
      molecular:   ``z(alpha) -> x'(beta) -> y''(gamma)``
      laboratory:  ``Y(gamma) -> X(beta) -> Z(alpha)``
      matrix:      ``R_Z(alpha) @ R_X(beta) @ R_Y(gamma)``

    - ``"zyx"``:
      molecular:   ``z(alpha) -> y'(beta) -> x''(gamma)``
      laboratory:  ``X(gamma) -> Y(beta) -> Z(alpha)``
      matrix:      ``R_Z(alpha) @ R_Y(beta) @ R_X(gamma)``.

    Euler-angle representations are not unique. This function returns canonical
    angles. For ``"zyz"``, ``beta`` is in ``[0, pi]``. For euler
    conventions, ``beta`` is in ``[-pi/2, pi/2]``. ``alpha`` and ``gamma`` are
    returned in the principal ``atan2`` range. At a singular orientation
    (gimbal lock), the first and third angles cannot be determined separately;
    this implementation sets ``gamma = 0`` and places the observable combined
    rotation into ``alpha``.

    ``R`` is assumed to be a proper rotation matrix whose columns are the
    molecular basis vectors expressed in the laboratory frame. Therefore a
    vector expressed in molecular coordinates transforms as ``v_lab = R @ v_mol``.

    :param R: Rotation matrix or batch of rotation matrices with shape
        ``[..., 3, 3]``.
    :param convention: Intrinsic molecular-axis convention. Supported values are
        ``"zyz"``, ``"xyz"``, ``"xzy"``, ``"yxz"``, ``"yzx"``, ``"zxy"``,
        and ``"zyx"``. Default is ``"zyz"``.
    :return: Euler/Tait-Bryan angles ``[alpha, beta, gamma]`` in radians with
        shape ``[..., 3]``.
    """
    if not isinstance(R, torch.Tensor):
        raise TypeError("R must be a torch.Tensor.")
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R must have shape [..., 3, 3], got {tuple(R.shape)}.")
    if not torch.is_floating_point(R):
        R = R.to(dtype=torch.get_default_dtype())

    if convention not in _SUPPORTED_EULER_CONVENTIONS:
        raise ValueError(
            f"Unsupported convention: {convention!r}. "
            f"Supported conventions are {sorted(_SUPPORTED_EULER_CONVENTIONS)}."
        )

    r00, r01, r02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    r10, r11, r12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    r20, r21, r22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    zero = torch.zeros_like(r00)
    eps = 1e-6

    if convention == "zyz":
        sin_beta_abs = torch.sqrt(torch.clamp(r02 * r02 + r12 * r12, min=0.0))
        beta = torch.atan2(sin_beta_abs, r22)
        nonsingular = sin_beta_abs > eps

        alpha_general = torch.atan2(r12, r02)
        gamma_general = torch.atan2(r21, -r20)
        alpha_beta0 = torch.atan2(r10, r00)

        alpha_betapi = torch.atan2(-r10, r11)
        near_beta0 = beta < (math.pi / 2.0)
        alpha_singular = torch.where(near_beta0, alpha_beta0, alpha_betapi)

        alpha = torch.where(nonsingular, alpha_general, alpha_singular)
        gamma = torch.where(nonsingular, gamma_general, zero)

    else:
        if convention == "xyz":
            sin_beta = r02
            cos_beta_abs = torch.sqrt(torch.clamp(r00 * r00 + r01 * r01, min=0.0))
            alpha_general = torch.atan2(-r12, r22)
            gamma_general = torch.atan2(-r01, r00)
            alpha_pos_lock = torch.atan2(r10, r11)
            alpha_neg_lock = torch.atan2(-r10, r11)

        elif convention == "xzy":
            sin_beta = -r01
            cos_beta_abs = torch.sqrt(torch.clamp(r00 * r00 + r02 * r02, min=0.0))
            alpha_general = torch.atan2(r21, r11)
            gamma_general = torch.atan2(r02, r00)
            alpha_pos_lock = torch.atan2(r20, r22)
            alpha_neg_lock = torch.atan2(-r20, r22)

        elif convention == "yxz":
            sin_beta = -r12
            cos_beta_abs = torch.sqrt(torch.clamp(r10 * r10 + r11 * r11, min=0.0))
            alpha_general = torch.atan2(r02, r22)
            gamma_general = torch.atan2(r10, r11)
            alpha_pos_lock = torch.atan2(-r20, r00)
            alpha_neg_lock = alpha_pos_lock

        elif convention == "yzx":
            sin_beta = r10
            cos_beta_abs = torch.sqrt(torch.clamp(r11 * r11 + r12 * r12, min=0.0))
            alpha_general = torch.atan2(-r20, r00)
            gamma_general = torch.atan2(-r12, r11)
            alpha_pos_lock = torch.atan2(r02, r22)
            alpha_neg_lock = alpha_pos_lock

        elif convention == "zxy":
            sin_beta = r21
            cos_beta_abs = torch.sqrt(torch.clamp(r20 * r20 + r22 * r22, min=0.0))
            alpha_general = torch.atan2(-r01, r11)
            gamma_general = torch.atan2(-r20, r22)
            alpha_pos_lock = torch.atan2(r10, r00)
            alpha_neg_lock = alpha_pos_lock

        else:  # convention == "zyx"
            sin_beta = -r20
            cos_beta_abs = torch.sqrt(torch.clamp(r00 * r00 + r10 * r10, min=0.0))
            alpha_general = torch.atan2(r10, r00)
            gamma_general = torch.atan2(r21, r22)
            alpha_pos_lock = torch.atan2(r12, r11)
            alpha_neg_lock = torch.atan2(-r12, r11)

        beta = torch.atan2(sin_beta, cos_beta_abs)
        nonsingular = cos_beta_abs > eps

        alpha_singular = torch.where(
            beta >= 0.0,
            alpha_pos_lock,
            alpha_neg_lock,
        )
        alpha = torch.where(nonsingular, alpha_general, alpha_singular)
        gamma = torch.where(nonsingular, gamma_general, zero)

    return torch.stack((alpha, beta, gamma), dim=-1)


def mean_rotation_svd(Rs: torch.Tensor):
    """Compute mean rotation matrix as SVD projection of mean value of rotation
    matrices.

    :param Rs: rotation matrices with shape [..., n, 3, 3], where n is number for mean computation.
    :return: R_mean - mean rotation matrix with shape [..., 3, 3]
    """
    M = Rs.sum(dim=-3)
    U, S, Vh = torch.linalg.svd(M)
    R = U @ Vh
    detR = torch.det(R)
    neg_mask = detR < 0
    if neg_mask.any():
        U_alt = U.clone()
        U_alt[..., :, -1] *= -1.0
        R_alt = U_alt @ Vh
        mask_mat = neg_mask.unsqueeze(-1).unsqueeze(-1)
        R = torch.where(mask_mat, R_alt, R)
    return R


def get_canonical_orientations(angles: torch.Tensor):
    """Compute Canonical angles for set of angles using SVD mean projection.

    :param angles: euler angles in convention zyz. The shape is [..., n, 3], where n is set size
    :return: Canonical angles
    """
    Rs = euler_angles_to_matrix(angles)
    R_mean = mean_rotation_svd(Rs)
    R_align = R_mean.transpose(-2, -1)

    n = Rs.shape[-3]
    batch_shape = R_align.shape[:-2]
    expand_shape = tuple(batch_shape) + (n, 3, 3)
    R_align_expanded = R_align.unsqueeze(-3).expand(expand_shape)
    new_Rs = torch.matmul(R_align_expanded, Rs)

    return rotation_matrix_to_euler_angles(new_Rs)


def get_canonical_orientations_(angles: torch.Tensor) -> torch.Tensor:
    """
    Compute Canonical angles for set of angles using SVD mean projection in-place.
    """
    Rs = euler_angles_to_matrix(angles)
    R_mean = mean_rotation_svd(Rs)
    R_mean.transpose_(-2, -1)
    n = Rs.shape[-3]
    batch_shape = R_mean.shape[:-2]
    expand_shape = batch_shape + (n, 3, 3)
    R_align_expanded = R_mean.unsqueeze(-3).expand(expand_shape)
    torch.matmul(R_align_expanded, Rs, out=Rs)
    angles.copy_(rotation_matrix_to_euler_angles(Rs))
    return angles


def float_to_complex_dtype(dtype: torch.dtype):
    if dtype is torch.float16:
        return torch.complex32
    elif dtype is torch.float32:
        return torch.complex64
    elif dtype is torch.float64:
        return torch.complex128
    else:
        raise NotImplementedError("dtype must be float")


def are_optional_tensors_close(first_tensor: tp.Optional[torch.Tensor],
                               second_tensor: tp.Optional[torch.Tensor],
                               rtol: float = 1e-5, atol: float = 1e-6) -> bool:
    """
    Compare two optional tensors for numerical equality, handling ``None`` values gracefully.

    This method safely checks if two tensors are equivalent. It first verifies that both
    tensors share the same ``None`` state. If both are tensors, it evaluates their numerical
    closeness

    :param first_tensor: The first tensor to compare. Can be ``None``.
    :param second_tensor: The second tensor to compare. Can be ``None``.
    :return: ``True`` if both tensors are ``None``, or if both are tensors and their values
             are close
             Returns ``False`` if one is ``None`` and the other is not, or if the tensors
             differ.
    """
    if (first_tensor is None) != (second_tensor is None):
        return False
    if first_tensor is not None:
        if not torch.allclose(first_tensor, second_tensor, rtol=rtol, atol=atol):
            return False
    return True


def _phase_to_positive(z: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Return unit-modulus phase factors that make ``z`` real and positive.

    :param z:
        Complex tensor.

    :param eps:
        Threshold below which the phase is treated as undefined.

    :return:
        Complex tensor with the same shape as ``z`` containing unit-modulus
        phase factors.
    """
    abs_z = z.abs()

    return torch.where(
        abs_z > eps,
        z.conj() / abs_z.clamp_min(eps),
        torch.ones_like(z),
    )


def align_eigenvector_phases(
    V: torch.Tensor,
    eps: tp.Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Phase-align eigenvectors on an ``[..., R, K, N, N]`` grid.

    Eigenvectors must be stored in columns:

    ``V[..., r, k, :, n]``

    is eigenvector ``n``.

    The algorithm treats every K position independently:

    1. At ``R = 0``, each eigenvector phase is fixed using its component
       with the largest absolute value.
    2. That largest component is made real and positive.
    3. For every K independently, the phase is transported along R using
       overlaps between neighboring eigenvectors.
    4. All leading ``...`` dimensions are treated independently.

    The phase at ``R = 0`` is therefore not aligned between different K
    values. Each K chain has its own independent phase reference.

    The implementation is fully vectorized and does not use Python loops
    over R or K.

    :param V:
        Eigenvector tensor with shape ``[..., R, K, N, N]``.

        The second-to-last dimension contains vector components in the
        original basis and the last dimension enumerates eigenvectors.

    :param eps:
        Numerical threshold used when determining phases. If ``None``,
        a dtype-dependent value is used.

    :return:
        Tuple ``(V_aligned, gauge)``.

        ``V_aligned``:
            Phase-aligned eigenvectors with the same shape as ``V``.

        ``gauge``:
            Complex phase factors with shape ``[..., R, K, N]``.
            The relation is
            ``V_aligned = V * gauge.unsqueeze(-2)``.

    .. note::
        This function performs phase alignment only. It assumes that
        eigenvector index ``n`` corresponds to the same physical state
        at neighboring R positions.

        At eigenvalue crossings or degeneracies, eigenvector matching or
        degenerate-subspace alignment may be required before phase fixing.
    """
    if V.ndim < 4:
        raise ValueError(
            "Expected V with shape [..., R, K, N, N]."
        )

    if eps is None:
        eps = 100.0 * torch.finfo(V.real.dtype).eps

    V_r0 = V[..., 0, :, :, :]
    indices = V_r0.abs().argmax(
        dim=-2,
        keepdim=True,
    )
    dominant = V_r0.gather(
        dim=-2,
        index=indices,
    ).squeeze(-2)

    gauge_r0 = _phase_to_positive(
        dominant,
        eps,
    )

    overlap = (
        V[..., :-1, :, :, :].conj()
        * V[..., 1:, :, :, :]
    ).sum(dim=-2)

    relative_phase = _phase_to_positive(
        overlap,
        eps,
    )

    accumulated = torch.cumprod(
        relative_phase,
        dim=-3,
    )

    gauge = torch.cat(
        (
            gauge_r0.unsqueeze(-3),
            gauge_r0.unsqueeze(-3) * accumulated,
        ),
        dim=-3,
    )

    V_aligned = V * gauge.unsqueeze(-2)

    return V_aligned, gauge
