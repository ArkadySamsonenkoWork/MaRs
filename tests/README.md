# Tests

Run notebooks from their containing folders using an environment with MaRs dependencies. MATLAB scripts require EasySpin on the MATLAB path.

| Test | What it tests |
|---|---|
| [tests/context_and_transformation_tests/concatenation_transformation_test.ipynb](context_and_transformation_tests/concatenation_transformation_test.ipynb) | Direct-sum superoperator assembly |
| [tests/context_and_transformation_tests/contexts_transformation_test.ipynb](context_and_transformation_tests/contexts_transformation_test.ipynb) | Context composition and basis transformations |
| [tests/context_and_transformation_tests/dephasing_matrix_test.ipynb](context_and_transformation_tests/dephasing_matrix_test.ipynb) | Coherence decay consistency |
| [tests/context_and_transformation_tests/general_consistancies_tests/relaxation_regression.ipynb](context_and_transformation_tests/general_consistancies_tests/relaxation_regression.ipynb) | Relaxation positivity, conservation and equilibrium |
| [tests/context_and_transformation_tests/general_consistancies_tests/test_physical_generators.py](context_and_transformation_tests/general_consistancies_tests/test_physical_generators.py) | Degenerate bases and thermal CP assumptions |
| [tests/context_and_transformation_tests/kinetic_rates_transformation_test.ipynb](context_and_transformation_tests/kinetic_rates_transformation_test.ipynb) | Composite kinetic and Liouville equivalence |
| [tests/context_and_transformation_tests/population_transfer_matrix_test.ipynb](context_and_transformation_tests/population_transfer_matrix_test.ipynb) | Population transfer and superoperator consistency |
| [tests/context_and_transformation_tests/redfield_relaxation_manual_test.ipynb](context_and_transformation_tests/redfield_relaxation_manual_test.ipynb) | Thermal-rate and Redfield spectral equivalence |
| [tests/context_and_transformation_tests/redfield_relaxation_test.ipynb](context_and_transformation_tests/redfield_relaxation_test.ipynb) | Redfield and Lindblad relaxation equivalence |
| [tests/context_and_transformation_tests/saturation_effect_test.ipynb](context_and_transformation_tests/saturation_effect_test.ipynb) | Microwave saturation and recovery times |
| [tests/context_and_transformation_tests/thermal_balance_corrector_test.ipynb](context_and_transformation_tests/thermal_balance_corrector_test.ipynb) | Detailed balance and dephasing correction |
| [tests/context_and_transformation_tests/thermal_equilibrium_redfield_test.ipynb](context_and_transformation_tests/thermal_equilibrium_redfield_test.ipynb) | Redfield thermal spectral-density balance |
| [tests/context_and_transformation_tests/thermal_equilibrium_test.ipynb](context_and_transformation_tests/thermal_equilibrium_test.ipynb) | Thermal rates and Boltzmann equilibrium |
| [tests/context_and_transformation_tests/transformations_test.ipynb](context_and_transformation_tests/transformations_test.ipynb) | Hilbert and Liouville basis transformations |
| [tests/optimization_test.ipynb](optimization_test.ipynb) | Optimization methods and penalty behavior |
| [tests/polarization_tests/easy_check_polarization.m](polarization_tests/easy_check_polarization.m) | EasySpin polarized spectral references |
| [tests/polarization_tests/mars_check_polarization.ipynb](polarization_tests/mars_check_polarization.ipynb) | Spin-polarized triplets and radical pairs |
| [tests/quality_tests/degenrated_examples/degenerate_examples.ipynb](quality_tests/degenrated_examples/degenerate_examples.ipynb) | Degenerate spin spectra and convergence |
| [tests/quality_tests/degenrated_examples/easyspin_degenerate_examples.m](quality_tests/degenrated_examples/easyspin_degenerate_examples.m) | EasySpin degenerate spectral references |
| [tests/quality_tests/easy_spin_comparison/easy_spic_checker.m](quality_tests/easy_spin_comparison/easy_spic_checker.m) | EasySpin quality spectral references |
| [tests/quality_tests/easy_spin_comparison/quality_test.ipynb](quality_tests/easy_spin_comparison/quality_test.ipynb) | MaRs and EasySpin spectral agreement |
| [tests/quality_tests/intensity_reference.ipynb](quality_tests/intensity_reference.ipynb) | Absolute spectral intensity normalization |
| [tests/quality_tests/zfs_dipolar_connection_test.ipynb](quality_tests/zfs_dipolar_connection_test.ipynb) | ZFS and dipolar relaxation scaling |
| [tests/serialization_tests/operation_tests.ipynb](serialization_tests/operation_tests.ipynb) | Serialization and MaRs object operations |
| [tests/spectra_manager_test/base_direct_computer_test.ipynb](spectra_manager_test/base_direct_computer_test.ipynb) | Direct spectrum computation |
| [tests/time_tests/cpu_gpu_dtype_comparison/cw_and_timeresolved_benchmarks.ipynb](time_tests/cpu_gpu_dtype_comparison/cw_and_timeresolved_benchmarks.ipynb) | CW and time-resolved computation speed |
| [tests/time_tests/cpu_gpu_dtype_comparison/results_analysis.ipynb](time_tests/cpu_gpu_dtype_comparison/results_analysis.ipynb) | CPU, GPU and precision timing comparison |
| [tests/time_tests/easy_spin_speed_comparison/easy_spin_speed_comparison.ipynb](time_tests/easy_spin_speed_comparison/easy_spin_speed_comparison.ipynb) | MaRs and EasySpin timing comparison |
| [tests/time_tests/easy_spin_speed_comparison/easy_spin_time_test.m](time_tests/easy_spin_speed_comparison/easy_spin_time_test.m) | EasySpin spectra and computation timings |
| [tests/time_tests/general_tests/time_test.ipynb](time_tests/general_tests/time_test.ipynb) | Spectral runtime across library versions |
