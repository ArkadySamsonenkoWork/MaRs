"""Focused regressions for degenerate bases and thermal CP assumptions.

Run from the repository root:
    python -m unittest discover -s tests/context_and_transformation_tests/general_consistancies_tests -p 'test_*.py'
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from mars import spin_model
from mars.population import choi_transforms as choi, check_transforms as checks
from mars.population.transform import Liouvilleator
from mars.population.thermal_corrections import ThermalBalanceCorrector


class PhysicalGeneratorTests(unittest.TestCase):
    def test_degenerate_multiplet_basis(self):
        for spins in ([0.5, 0.5], [0.5] * 3, [0.5] * 4, [1.0, 1.0]):
            system = spin_model.SpinSystem(
                electrons=spins,
                g_tensors=[spin_model.Interaction(2.0, dtype=torch.float64) for _ in spins],
                dtype=torch.float64,
            )
            basis = system.get_spin_multiplet_basis()
            identity = torch.eye(basis.shape[-1], dtype=basis.dtype)
            self.assertTrue(torch.allclose(basis.mH @ basis, identity, atol=1e-12))
            for operator in (system.get_electron_squared_operator(), system.get_electron_z_operator()):
                represented = basis.mH @ operator @ basis
                self.assertLess((represented - represented.diag().diag()).abs().max().item(), 1e-11)

    def test_corrected_nonsecular_cp_violation_is_expected(self):
        import numpy as np
        from sample_generators import make_sample_case
        from context_generators import generate_contexts
        from composer import compose_cases
        from utils import get_superoperators

        rng = np.random.default_rng(42)
        samples = make_sample_case(rng, spins=(0.5, 1.0), partition=1,
                                   dtype=torch.float64, mesh=(5, 5))
        contexts = generate_contexts(samples, rng, secular=False, decay=False,
                                     dtype=torch.float64)
        case = compose_cases(samples, contexts)[4]
        margins = []
        for corrected in (False, True):
            operators = get_superoperators(case.sample, case.context,
                                           magnetic_field_t=0.34, temperature_k=3.0,
                                           thermally_corrected=corrected)
            margins.append(choi.generator_complete_positivity_margin(operators["thermal"]).item())
        self.assertGreaterEqual(margins[0], -1e-7)
        self.assertLess(margins[1], -1e-3)

    def test_secular_limit_and_missing_reverse_jump(self):
        rates = torch.tensor([[0., 2., 0.], [0., 0., 3.], [0., 0., 0.]], dtype=torch.float64)
        energies = torch.tensor([0., 8e9, 13e9], dtype=torch.float64)
        dephasing = Liouvilleator.lindblad_dephasing_from_rates(torch.tensor([1., 2., 3.], dtype=torch.float64))
        loss = -0.5 * Liouvilleator.anticommutator_superop(torch.diag(torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)))
        raw = Liouvilleator.lindblad_dissipator_from_rates(rates) + dephasing + loss
        for mode in ('skip', 'symmetric', 'complement'):
            corrector = ThermalBalanceCorrector(energies, mode)
            for value in (0.5, 2.0):
                temperature = torch.tensor(value, dtype=torch.float64)
                new_rates = corrector.apply_matrix_transform(temperature, rates)
                expected = Liouvilleator.lindblad_dissipator_from_rates(new_rates) + dephasing + loss
                corrected = corrector.apply_superoperator_transform(temperature, raw)
                self.assertTrue(torch.allclose(corrected, expected, atol=1e-11))
                self.assertGreater(choi.generator_complete_positivity_margin(corrected).item(), -1e-10)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
