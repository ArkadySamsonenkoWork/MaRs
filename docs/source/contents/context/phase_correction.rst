.. _phase_correction:


Phase correction for eigenstates
=================================

The problem
-----------

An eigenvector has an arbitrary complex phase. If

.. math::

   H |v_n\rangle = E_n |v_n\rangle,

then :math:`e^{i\phi_n}|v_n\rangle` is the same eigenstate. Numerical
diagonalization may therefore return different phases for different
orientations or field positions.

This ambiguity does not affect populations, but it does affect the numerical
representation of coherences. Under a rephasing of the basis,

.. math::

   \rho_{mn} \rightarrow e^{-i\phi_m}\rho_{mn}e^{i\phi_n}.

Thus, when a non-diagonal density matrix is defined in a matrix basis obtained
from eigenvectors, the phase convention of these basis vectors must be
consistent.


Matrix-defined bases
--------------------

This issue is relevant when the density matrix is defined in a basis supplied
explicitly as a matrix. The columns of the matrix define the basis vectors.

If such a basis is obtained from numerical diagonalization, each column may
contain an arbitrary complex phase. Therefore, independently calculated basis
matrices for different orientations or field positions may use different phase
conventions even when they represent the same physical eigenstates.

For a diagonal density matrix this phase choice is irrelevant. For a
non-diagonal density matrix, however, the matrix elements are defined relative
to the phases of the supplied basis vectors. Consequently, a consistent phase
convention is required before the density matrix is transformed or propagated.


Diagonal and off-diagonal elements
----------------------------------

The diagonal elements of a density matrix,

.. math::

   \rho_{nn},

represent state populations and are invariant under a phase change of the
basis vectors.

The off-diagonal elements,

.. math::

   \rho_{mn}, \qquad m \neq n,

represent coherences between basis states. If the basis vectors are rephased as

.. math::

   |n\rangle \rightarrow e^{i\phi_n}|n\rangle,

the corresponding density-matrix elements transform as

.. math::

   \rho_{mn}
   \rightarrow
   e^{-i\phi_m}\rho_{mn}e^{i\phi_n}.

Therefore, phase correction is required only when the density matrix contains
non-zero off-diagonal elements. A diagonal density matrix is unaffected by the
phase convention of the basis.


Phase correction in MaRs
------------------------

MaRs fixes the phase independently for every field position ``K``.

At ``R = 0``, the phase of each eigenvector is chosen so that the component
with the largest absolute value is real and positive. For the remaining
orientations, eigenvectors are aligned successively along ``R`` so that the
overlap with the previous orientation is real and positive.

In short:

#. each ``K`` chain receives an independent reference at ``R = 0``;
#. the ``R = 0`` reference uses the largest-component convention;
#. phases are then propagated continuously along ``R``.

This removes arbitrary sign and complex-phase changes produced by numerical
diagonalization. It does not resolve state permutations or arbitrary rotations
inside exactly degenerate subspaces.


Example: ``S = 1/2``
--------------------

For a two-level spin system, an eigensolver may return the same physical basis
as

.. math::

   |1\rangle = (-1, 0)^T, \qquad |2\rangle = (0, i)^T.

The ``R = 0`` phase correction converts these vectors to the convention

.. math::

   |1\rangle = (1, 0)^T, \qquad |2\rangle = (0, 1)^T,

and the phases of neighboring orientations are aligned to this reference.

For example, the coherent state

.. math::

   \rho = \frac{1}{2}
   \begin{pmatrix}
   1 & 1 \\
   1 & 1
   \end{pmatrix}

contains non-zero off-diagonal elements and therefore depends on the phase
convention of the two basis vectors. Phase correction makes this convention
consistent.

In contrast, the diagonal populations

.. math::

   \rho_{11} = \rho_{22} = \frac{1}{2}

are independent of the basis-vector phases and do not require phase
correction.
