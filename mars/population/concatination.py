import typing as tp
import warnings
import copy

import torch
import numpy as np

from .contexts import Context, SummedContext
from . import transform
from .relaxation_channels.redfield import RedfieldRelaxationChannel
from .relaxation_channels.base_couling_channels import CouplingChannelManager, combine_coupling_managers


def concat_coupling_managers(contexts: tp.Sequence[Context]) -> CouplingChannelManager:
    """
    Constructs a composite Coupling (Redfield, Lindblad)
    manager by expanding individual managers via Kronecker products.

    This function iterates through a sequence of contexts, expands each associated redfield manager
    into the full Hilbert space of the composite system using the dimensions of the other spin
    system.

    :param contexts: A sequence of Context objects, each containing a spin system dimension and
        an associated redfield manager.
    :return: A combined CouplingChannelManager object representing the relaxation dynamics of the
        entire composite system.
    """
    dims = [context.spin_system_dim for context in contexts]
    managers = [copy.deepcopy(context.coupling_manager) for context in contexts]
    for idx, manager in enumerate(managers):
        if manager is None:
            continue
        left_dim = np.sum(dims[:idx]) if idx > 0 else 0
        right_dim = np.sum(dims[idx + 1:]) if idx < len(dims) - 1 else 0
        manager.expand_zeros(left_dim, right_dim)
    return combine_coupling_managers(managers)


def _normalize_to_components(
        contexts: tp.Sequence[tp.Union[Context, SummedContext]]
) -> tp.Tuple[tp.List[tp.List[Context]], tp.List[int], tp.List[int], tp.List[int], torch.device, torch.dtype]:
    """
    Normalize input contexts into uniform component lists with metadata.

    Converts all inputs to lists of atomic Context objects while preserving:
    - Component structure (flattening SummedContexts)
    - Basis presence metadata
    - System properties (dimension, device, dtype)

    :param contexts: List of Context or SummedContext objects to normalize
    :return:
        component_lists: List of component lists for each input context
        num_with_basis: Count of basis-equipped components per input context
        num_none_basis: Count of Non-basis components per input context
        spin_system_dimension: Count of size of spin system
        device:
        dtype:
    """
    dtype, device = None, None
    for ctx in contexts:
        ref_ctx = ctx
        dtype, device = ref_ctx.dtype, ref_ctx.device
        if dtype is not None:
            break

    for ctx in contexts[1:]:
        if ctx.dtype is not None:
            if ctx.dtype != dtype or ctx.device != device:
                raise ValueError("All contexts must share the same dtype and device")

    component_lists = []
    num_with_basis = []
    spin_system_dim = []
    num_none_basis = []

    for ctx in contexts:
        if isinstance(ctx, Context):
            components = [ctx]
            n_basis = 1 if ctx.basis is not None else 0
            spin_system_dim.append(ctx.spin_system_dim)
        elif isinstance(ctx, SummedContext):
            components = list(ctx.component_contexts)
            n_basis = ctx.num_contexts_with_basis
            spin_system_dim.append(ctx.spin_system_dim)
        else:
            raise TypeError(f"Unsupported context type: {type(ctx)}")

        if not components:
            continue

        component_lists.append(components)
        num_with_basis.append(n_basis)
        num_none_basis.append(len(components) - n_basis)

    return component_lists, num_with_basis, num_none_basis, spin_system_dim, device, dtype


def _create_zero_context(
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
        has_basis: bool
) -> Context:
    """
    Create a zero-initialized context with specified properties.

    :param dim: Dimension of the spin system
    :param device: Device to place tensors on
    :param dtype: Floating-point dtype for real components
    :param has_basis: Whether to include an identity basis matrix
    :return: Minimal context object with zero-initialized properties
    """
    if has_basis:
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        basis = torch.eye(dim, device=device, dtype=complex_dtype)
    else:
        basis = None

    context = Context(
        basis=basis,
        device=device,
        dtype=dtype
    )
    context.spin_system_dim = dim

    return context  


def _group_components_by_position(
        component_lists: tp.List[tp.List[Context]],
        num_with_basis: tp.List[int],
        num_none_basis: tp.List[int],
        spin_system_dim: tp.List[int],
        device: torch.device,
        dtype: torch.dtype,
) -> tp.List[tp.Sequence[Context]]:
    """
    Group components by position with minimal zero-padding.

    For each component position:
    1. Collect existing components from all contexts
    2. Pad missing positions with eigen-type (no basis) zero contexts
    3. Split into homogeneous basis-type groups when necessary

    Guarantees minimal groups by:
    - Creating one group when all components share basis type
    - Creating two groups (non-eigen + eigen) when types are mixed

    :param component_lists: Normalized component lists per context
    :param num_with_basis: Basis counts per original context
    :param num_none_basis: Basis None counts per original context
    :param spin_system_dim: Count the size of spin systems
    :param device:
    :param dtype:
    :return: List of context groups ready for concatenation
    """
    if not component_lists:
        return []

    max_components = max(len(lst) for lst in component_lists)
    max_num_basis = max(num_with_basis) if num_with_basis else 0
    max_none_basis = max(num_none_basis)if num_none_basis else 0

    if max_none_basis:
        warnings.warn(
            "Concatenation with context with the default (eigen or None) basis"
            "may lead to unexpected behavior. \n"
            "In this configuration, relaxation parameters for subsystems with "
            "basis=None` are interpreted directly in the final (field-dependent eigen) basis"
            "they will NOT be transformed from any other representation.  \n"
            "This is only physically meaningful if: \n"
            "(1) there is no coupling between subsystems, OR  \n"
            "(2) you intentionally define relaxation rates in the eigenbasis of the full Hamiltonian. \n"
            "To avoid ambiguity, consider explicitly specifying a consistent basis"
            "(e.g., 'zfs', 'xyz', or 'multiplet') for all subsystems."
        )

    padded_component_lists = []
    for i, component_group in enumerate(component_lists):
        dim = spin_system_dim[i]
        leading_zeros = max_num_basis - num_with_basis[i]
        trailing_zeros = max_none_basis - num_none_basis[i]
        padded_group = (
                [_create_zero_context(dim, device, dtype, has_basis=True)] * leading_zeros
                + component_group
                + [_create_zero_context(dim, device, dtype, has_basis=False)] * trailing_zeros
        )
        padded_component_lists.append(padded_group)
    return list(zip(*padded_component_lists))


def organize_contexts_for_concatenation(
        contexts: tp.Sequence[tp.Union[Context, SummedContext]]
) -> tp.Tuple[tp.List[tp.Sequence[Context]], torch.device, torch.dtype]:
    """Organize contexts into homogeneous basis groups with minimal zero padding.

    Groups contexts by position and basis type to enable efficient concatenation:
    1. Normalizes all inputs to component lists
    2. For each component position:
        - Creates groups of contexts with the same basis type
        - Adds minimal zero contexts only where necessary
    3. Returns groups ready for concat_sametype_contexts processing

    Example:
        Input: [(A1+A2), (B1)]
        Output: [
            [A1, B1],  # Position 0, non-eigen group
            [A2, zero] # Position 1, non-eigen group (assuming same basis type)
        ]

    Zero context strategy:
    - For positions < num_contexts_with_basis: create contexts with identity basis
    - For positions >= num_contexts_with_basis: create contexts with None basis
    - Only creates zero contexts when absolutely necessary

    :param contexts: List of Context or SummedContext objects to organize.

    :return:
        List of context groups, where each group contains contexts with:
        - Same component position
        - Same basis type (all None or all non-None)
        - Minimal zero padding for alignment
        device
        dtype
    """
    component_lists, num_with_basis, num_none_basis, spin_system_dim, device, dtype = _normalize_to_components(contexts)
    return _group_components_by_position(
        component_lists, num_with_basis, num_none_basis, spin_system_dim, device, dtype), device, dtype


def concat_contexts(contexts: tp.Sequence[tp.Union[Context, SummedContext]],
                    mode: str = "direct_sum") -> tp.Union[Context, SummedContext]:
    """Concatenates multiple contexts (either Context or SummedContext objects) into a single Composite context.

    Concatenates multiple contexts into a single context representing the direct sum of spin systems.

    Creates a new context where:
    - Hilbert space dimension is the sum of individual system dimensions
    - Parameters are combined block-diagonally (matrices) or by concatenation (vectors)
    - None values are replaced with zeros (vectors/matrices) or identity (basis)
    - Callables (time-dependent parameters) are not supported

    Physical interpretation:
    The resulting context describes multiple independent levels of the same system evolving without mutual interaction.
    Initial states are block-diagonal, transition rates only occur within subsystems.

    This function handles a heterogeneous collection of contexts by:
    1. Flattening all contexts (extracting components from SummedContexts)
    2. Grouping contexts by basis type (None vs non-None basis)
    3. Concatenating each homogeneous group using concat_sametype_contexts
    4. Summing the results to form the final composite context

    The grouping strategy ensures proper handling of basis transformations:
    - Non-eigenbasis contexts (explicit basis) are concatenated together
    - Eigenbasis contexts (basis=None) are concatenated together
    - The two resulting contexts are combined via summation (+)

    Example:
        concat_contexts([A+B+C, D+E]) produces (A con D) + (B con E) + (C con 0)
        where C con 0 is C concatenated with a zero initialized context

    :param contexts: List of Context or SummedContext objects to concatenate.

    :return:
        A single Context object if all subsystems share the same basis type,
        or a SummedContext containing two contexts (non-eigenbasis and eigenbasis groups).
    """
    if mode != "direct_sum":
        raise NotImplementedError("Current version of 'concat_contexts'"
                                  "supports only direct_sum concatenation")
    groups, device, dtype = organize_contexts_for_concatenation(contexts)
    time_dimension = groups[0][0].time_dimension
    results = [_concat_homogeneous_contexts(group, device, dtype, time_dimension) for group in groups]
    return results[0] if len(results) == 1 else SummedContext(results)


def _concat_sametype_contexts(contexts: tp.Sequence[Context]) -> tp.Union[Context, SummedContext]:
    """Concatenates multiple contexts into a single context representing the direct sum of spin systems.

    Creates a new context with special handling for basis specifications:
    - If ALL contexts have None basis OR ALL have non-None basis: creates one context with block-diagonal parameters
    - If MIXED basis types: creates two contexts and combines them via sum (+):
        1) Context for non-None basis subsystems (other subsystems have zero parameters)
        2) Context for None basis subsystems (other subsystems have zero parameters)

    Batch dimensions are preserved through broadcasting and expansion for all parameters.

    Parameter handling rules:
    - Homogeneous case:
        * If ALL contexts have None for a parameter then Resulting context has None
        * If ANY context has non-None parameter then Missing values replaced with:
            - Zeros for vectors (init_populations, decay_rates, dephasing)
            - Zero matrices for square matrices (thermal_rates, driven_rates)
    - Mixed basis case:
        * Non-eigenbasis context: Only non-None basis subsystems have non-zero parameters
        * Eigenbasis context: Only None basis subsystems have non-zero parameters

    Limitations:
    - Requires all contexts to share the same dtype and device
    - Does not support time-dependent (callable) parameters
    - Incompatible batch shapes will raise errors

    :param contexts: List of Context objects to concatenate.
    :return: New Context object representing the compiste (direct sum) system.

    Raises:
        ValueError: If contexts list is empty, contains incompatible dtypes/devices,
                    or has incompatible batch shapes
        TypeError: If any parameter is callable (time-dependent)
    """
    dtype, device = None, None
    for ctx in contexts:
        ref_ctx = ctx
        dtype, device = ref_ctx.dtype, ref_ctx.device
        if dtype is not None:
            break

    for ctx in contexts[1:]:
        if ctx.dtype is not None:
            if ctx.dtype != dtype or ctx.device != device:
                raise ValueError("All contexts must share the same dtype and device")

    basis_status = [ctx.basis is not None for ctx in contexts]
    all_none = all(not status for status in basis_status)
    all_non_none = all(basis_status)

    if not all_non_none:
        warnings.warn(
            "Concatenation with context with the default (eigen or None) basis"
            "may lead to unexpected behavior. \n"
            "In this configuration, relaxation parameters for subsystems with "
            "basis=None` are interpreted directly in the final (field-dependent eigen) basis"
            "they will NOT be transformed from any other representation.  \n"
            "This is only physically meaningful if: \n"
            "(1) there is no coupling between subsystems, OR  \n"
            "(2) you intentionally define relaxation rates in the eigenbasis of the full Hamiltonian. \n"
            "To avoid ambiguity, consider explicitly specifying a consistent basis"
            "(e.g., 'zfs', 'xyz', or 'multiplet') for all subsystems."
        )

    if all_none or all_non_none:
        return _concat_homogeneous_contexts(contexts, device, dtype, ref_ctx.time_dimension)
    else:
        complex_dtype = torch.complex64 if (dtype == torch.float32) else torch.complex128
        non_eigen_contexts = [
            ctx if ctx.basis is not None else Context(
                basis=torch.eye(ctx.dim, device=device, dtype=complex_dtype), device=device, dtype=dtype)
            for ctx in contexts]
        eigen_contexts = [ctx if ctx.basis is None else Context(device=device, dtype=dtype) for ctx in contexts]

        ctx_non_eigen = _concat_homogeneous_contexts(
            non_eigen_contexts, device, dtype, ref_ctx.time_dimension
        )
        ctx_eigen = _concat_homogeneous_contexts(
            eigen_contexts, device, dtype, ref_ctx.time_dimension
        )
        return ctx_non_eigen + ctx_eigen


def _concat_homogeneous_contexts(
    contexts: tp.Sequence[Context],
    device: torch.device,
    dtype: torch.dtype,
    time_dimension: int
) -> Context:
    """Concatenates multiple contexts into a single context representing the direct sum of spin systems.

    Creates a new context where:
    - Hilbert space dimension is the sum of individual system dimensions
    - Parameters are combined block-diagonally (matrices) or by concatenation (vectors)
    - None values are replaced with zeros (vectors/matrices) or identity (basis)
    - Callables (time-dependent parameters) are not supported
    - Requires that all contexts either have their own basis or none at all

    Physical interpretation:
    The resulting context describes multiple independent levels of the same system evolving without mutual interaction.
    Initial states are block-diagonal, transition rates only occur within subsystems, and dephasing is
    confined to individual subspaces.

    Parameter handling rules:
    - If ALL contexts have None for a parameter then Resulting context has None
    - If ANY context has non-None parameter then Missing values replaced with:
        * Zeros for vectors (init_populations, decay_rates, dephasing)
        * Zero matrices for square matrices (thermal_rates, driven_rates)
        * Identity matrices for basis transformations
    - profiles are always set to None in concatenated context
    - Batch dimensions are determined by broadcasting all non-None parameters

    Limitations:
    - Requires all contexts to share the same dtype and device
    - Does not support time-dependent (callable) parameters

    :param contexts:
        contexts: List of Context objects to concatenate.

    :return:
        New Context object representing the direct-sum system.

    Raises:
        ValueError: If contexts list is empty or contains incompatible dtypes/devices
        TypeError: If any parameter is callable (time-dependent)
    """

    dims = [ctx.spin_system_dim for ctx in contexts]
    total_dim = sum(dims)
    def _get_batch_shapes(values: tp.List[tp.Optional[torch.Tensor]], param_name: str, matrix=False):
        """Extract and validate batch shapes from a list of parameter values.

        Checks for unsupported callable parameters and collects batch dimensions
        from all non-None tensors. For matrices, batch shape excludes the last two
        dimensions (rows/columns); for vectors, excludes the last dimension.

        :param values: List of parameter values (tensors or None) from contexts.
        :param param_name: Name of the parameter (for error messages).
        :param matrix: If True, treats values as matrices (shape [..., M, N]);
                       otherwise as vectors (shape [..., D]).
        :return: List of batch shapes (torch.Size) from non-None values.
                 Returns [torch.Size([])] if all values are None.

        Raises:
            TypeError: If any non-None value is callable (time-dependent parameter).
        """
        if any(callable(v) for v in values if v is not None):
            raise TypeError(f"Callable {param_name} not supported in concatenation")

        batch_shapes = []
        for v in values:
            if v is not None:
                if matrix:
                    batch_shapes.append(v.shape[:-2])
                else:
                    batch_shapes.append(v.shape[:-1])
        return batch_shapes or [torch.Size([])]

    def _process_vectors(attr_name: str):
        """Construct block-concatenated vector by stacking context vectors along system dimension.

        Handles parameters like init_populations, decay_rates, and dephasing.
        Missing values (None) are filled with zeros in their respective blocks.
        Batch dimensions are broadcasted across all non-None inputs.

        :param attr_name: Name of the vector attribute to extract from each context
                          (e.g., 'init_populations', 'decay_rates').
        :return: Concatenated tensor of shape [..., total_dim] where total_dim = sum(dims),
                 or None if all contexts have None for this attribute.
        """
        values = [getattr(ctx, attr_name) for ctx in contexts]
        batch_shapes = _get_batch_shapes(values, attr_name)

        if all(v is None for v in values):
            return None

        common_batch_shape = torch.Size(torch.broadcast_shapes(*batch_shapes))
        composite = torch.zeros(common_batch_shape + (total_dim,), dtype=dtype, device=device)

        start = 0
        for i, val in enumerate(values):
            dim = dims[i]
            end = start + dim

            if val is not None:
                expanded = val.expand(common_batch_shape + val.shape[-1:])
                composite[..., start:end] = expanded

            start = end
        return composite

    def _process_matrices(attr_name, use_identity_for_none=False):
        """Construct block-diagonal matrix by placing context matrices on the diagonal.

        Handles square matrix parameters (thermal_rates, driven_rates, basis).
        Missing values are filled with zeros unless use_identity_for_none=True
        (used for basis where identity is physically meaningful).

        :param attr_name: Name of the matrix attribute to extract (e.g., 'basis', 'thermal_rates').
        :param use_identity_for_none: If True, replaces None values with identity matrices
                                      (required for basis transformations); otherwise uses zeros.
        :return: Block-diagonal tensor of shape [..., total_dim, total_dim],
                 or None if all contexts have None for this attribute.

        Block structure:
            [ M₁  0  0 ... ]
            [ 0  M₂  0 ... ]
            [ 0   0 M₃ ... ]
            [...         ]
        where Mᵢ is the matrix from context i (or identity/zero if missing).
        """
        values = [getattr(ctx, attr_name) for ctx in contexts]
        batch_shapes = _get_batch_shapes(values, attr_name, matrix=True)

        if all(v is None for v in values):
            return None

        common_batch_shape = torch.Size(torch.broadcast_shapes(*batch_shapes))
        composite = torch.zeros(common_batch_shape + (total_dim, total_dim), dtype=dtype, device=device)

        if attr_name == "basis" and use_identity_for_none:
            basis_dtype = next((ctx.basis.dtype for ctx in contexts if ctx.basis is not None), dtype)
            composite = composite.to(basis_dtype)

        start = 0
        for i, val in enumerate(values):
            dim = dims[i]
            end = start + dim

            if val is None and use_identity_for_none:
                block = torch.eye(dim, dtype=composite.dtype, device=device)
                block = block.expand(common_batch_shape + (dim, dim))
            elif val is not None:
                expanded_shape = common_batch_shape + val.shape[-2:]
                block = val.expand(expanded_shape)
            else:
                block = torch.zeros(common_batch_shape + (dim, dim), dtype=dtype, device=device)

            slices = [slice(None)] * len(common_batch_shape) + [slice(start, end), slice(start, end)]
            composite[slices] = block

            start = end
        return composite

    def _process_superoperators(attr_name: str):
        """Construct composite relaxation superoperator using correct Liouville-space embedding.

        Unlike Hilbert-space matrices, superoperators require non-contiguous embedding due to
        vectorization of block-diagonal density matrices. Uses
        transform.reshape_superoperators_tensor_to_direct_sum_basis for correct placement.

        :return: Composite superoperator of shape [..., N², N²] where N = Σᵢ Nᵢ,
                 or None if all contexts have None for relaxation_superop.

        Note:
            Batch dimensions are preserved by expanding all superoperators to a common batch shape
            before embedding. The embedding operation is applied identically across all batch elements.
        """
        superops = [getattr(ctx, attr_name) for ctx in contexts]
        if all(sop is None for sop in superops):
            return None

        batch_shapes = [sop.shape[:-2] for sop in superops if sop is not None]
        common_batch_shape = torch.Size(torch.broadcast_shapes(*batch_shapes)) if batch_shapes else torch.Size()

        superop_list = []
        for i, sop in enumerate(superops):
            Ni = dims[i]
            target_shape = common_batch_shape + (Ni * Ni, Ni * Ni)

            if sop is None:
                block = torch.zeros(target_shape, dtype=dtype, device=device)
            else:
                block = sop.expand(target_shape)

            superop_list.append(block)

        return transform.reshape_superoperators_list_to_direct_sum_basis(superop_list)

    def _process_density():
        """Construct block-diagonal initial density matrix from context density matrices.

        Unlike real-valued probabilities, density matrices are complex-valued.
        Missing density matrices are treated as zero blocks (no coherence/population).

        :return: Block-diagonal density matrix of shape [..., total_dim, total_dim]
                 with complex dtype (complex64 for float32, complex128 for float64),
                 or None if all contexts have None for init_density.

        Note:
            The resulting density matrix represents a statistical mixture of independent
            subsystems with no initial coherence between different spin systems.
        """
        densities = [ctx.init_density for ctx in contexts]
        batch_shapes = []

        for dens in densities:
            if dens is not None:
                batch_shapes.append(dens.shape[:-2])

        if not batch_shapes:
            return None

        common_batch_shape = torch.Size(torch.broadcast_shapes(*batch_shapes))
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128

        composite = torch.zeros(
            common_batch_shape + (total_dim, total_dim),
            dtype=complex_dtype,
            device=device
        )

        start = 0
        for i, dens in enumerate(densities):
            dim = dims[i]
            end = start + dim

            if dens is not None:
                expanded_shape = common_batch_shape + dens.shape[-2:]
                block = dens.expand(expanded_shape)
                composite[..., start:end, start:end] = block

            start = end
        return composite

    use_identity_for_basis = any(ctx.basis is not None for ctx in contexts)

    coupling_manager = concat_coupling_managers(contexts)
    if coupling_manager is not None:
        relaxation_coupling_channels = [channel for channel in coupling_manager.relaxation_channels]
    else:
        relaxation_coupling_channels = None
    return Context(
        basis=_process_matrices("basis", use_identity_for_none=use_identity_for_basis),
        init_populations=_process_vectors("init_populations"),
        init_density=_process_density(),
        thermal_rates=_process_matrices("thermal_rates"),
        driven_rates=_process_matrices("driven_rates"),
        decay_rates=_process_vectors("decay_rates"),
        dephasing=_process_vectors("dephasing"),
        relaxation_superop=_process_superoperators("_default_driven_superop"),
        relaxation_coupling_channels=relaxation_coupling_channels,
        profile=None,
        time_dimension=time_dimension,
        dtype=dtype,
        device=device
    )