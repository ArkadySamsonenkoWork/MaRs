Installation
============

From PyPI:
::


   pip install mars-epr

From source:
::


   git clone https://github.com/ArkadySamsonenkoWork/MaRs.git
   cd MaRs
   pip install -e .


We highly recommend installing the package within a Python virtual environment (``venv``).
Conceptually, a virtual environment acts as an isolated workspace that keeps your project dependencies separate from your system-wide Python installation,
preventing version conflicts.
This ensures a clean, reproducible setup for running MaRs and associated tools like Jupyter.
For a detailed, beginner-friendly guide on creating and activating virtual environments, please refer to the Python Virtual Environments Primer by Real Python [1]_.

.. [1] Real Python, "Python Virtual Environments: A Primer", https://realpython.com/python-virtual-environme


Optimization Dependencies
-------------------------

To use the optimization features in MaRs, additional libraries are required. You can install them directly via pip:

::

   pip install "optuna>=4.3" "nevergrad>=1.0.12" "optuna_dashboard" "emcee" "hdbscan" "numdifftools" "optuna-integration[botorch]"