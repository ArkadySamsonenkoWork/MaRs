# Configuration file for the Sphinx documentation builder.

project = 'MaRs'
copyright = '2026, Arkady Samsonenko, Ivan Kurgansky'
author = 'Arkady Samsonenko, Ivan Kurgansky'
release = '2026.08.09'
version = '0.0.5.post5'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',  # Move this before viewcode
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'nbsphinx'
]

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,  # Changed to True to catch all members
    'exclude-members': '__weakref__',
    'inherited-members': True,
    'show-inheritance': True,
}

# Autosummary settings
autosummary_generate = True
autosummary_generate_overwrite = True  # Add this
autosummary_imported_members = False  # Change to False to avoid duplicates

autodoc_preserve_defaults = True
autodoc_mock_imports = [
    # Core dependencies
    "torch",
    "scipy",
    "numpy",
    "sklearn",
    "matplotlib",
    
    # Optional dependencies
    "optuna",
    "nevergrad",
    "optuna_dashboard",
    "torchdiffeq",
    "safetensors",
    "pywt",
    "emcee",
    
    # Visualization
    "seaborn",
    "plotly",
    "pandas",
    
    # Any torch submodules
    "torch.nn",
    "torch.optim",
    "torch.linalg",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
todo_include_todos = True

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False

autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Path setup
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(__file__, '../../../'))
sys.path.insert(0, PROJECT_ROOT)

# Debug output (will appear in RTD logs)
print(f"DEBUG: PROJECT_ROOT = {PROJECT_ROOT}")
print(f"DEBUG: sys.path[0] = {sys.path[0]}")

on_rtd = os.environ.get('READTHEDOCS') == 'True'