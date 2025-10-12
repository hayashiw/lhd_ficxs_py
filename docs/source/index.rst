.. LHD-FICXS-py documentation master file, created by
   sphinx-quickstart on Mon Oct  6 22:06:32 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LHD-FICXS-py documentation
==========================

Python tools to analyze fast ion data from the Large Helical Device.
--------------------------------------------------------------------
This repository is primarily concerned with analysis of data from the fast-ion charge-exchange spectroscopy (FICXS) system on the Large Helical Device (LHD).

Installation (Linux)
++++++++++++++++++++
To use the package, simply clone the repository from github and append the directory path to the pythonpath. ::
   git clone https://github.com/hayashiw/lhd_ficxs_py.git
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/lhd_ficxs_py"
::

Configuring directory paths
+++++++++++++++++++++++++++
Included in the package is the ``config.ini`` file.
This files contains the relevant directory paths for your LHD data.
Make sure the ``data_dir`` and ``pqt_dir`` are correctly set to the directories that you will be using. ::
   [data_directories]
   data_dir = /path/to/your/LHD/data
   pqt_dir = /path/to/your/parquet/data
::


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/modules
