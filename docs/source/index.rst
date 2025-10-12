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
For information on the use of parquet files see :ref:`handling-ficxs-data-files`

Reading in data
+++++++++++++++
LHD data is typically stored in comma-separated ASCII files with the .dat extension.
Most .dat files can be read in using the ``read_data`` subroutine.
This will return the data as a ``pandas.DataFrame`` object. ::

   from lhd_ficxs_py import read_data

   dat_file = './nb1pwr_temporal.dat'
   nb1_df = read_data(dat_file, use_postgres_names=True, convert_to_ms=True)
   # use_postgres_names converts column names to lower-snake-case format
   # convert_to_ms converts the time column to milliseconds
   pinj = nb1_df['pport_through_nb1']
   pinj_units = nb1_df.attrs['units'] # MW
::

.. _handling-ficxs-data-files
Handling FICXS data files
+++++++++++++++++++++++++
Calibrated FICXS data is written to .dat files, the standard file type for LHD data.
Due to the size of these files, file IO becomes a bottleneck when analyzing multiple cases.
It is helpful to compress FICXS .dat files into parquet files.
This can be easily done using the ``lhd_ficxs_py.io.convert_dat_to_pqt`` subroutine.
They can then be quickly read in using ``pandas.read_parquet``.
An additional subroutine ``lhd_ficxs_py.io.read_ficxs_from_pqt`` is available to format FICXS dataframes into the expected format for ``lhd_ficxs_py``.

Calculating timing for active FICXS signal
++++++++++++++++++++++++++++++++++++++++++
The main purpose of this package is to generate the diagnostic-beam-on and -off time stamps for the FICXS system.
This can be done using ``lhd_ficxs_py.run`` from the command line: ::

   (lhd_ficxs_py_env): python -m lhd_ficxs_py.run {shot_num} {los}
::
The first argument ``{shot_num}`` should be replaced with the LHD discharge number.
This is used to set the paths for the input data files, namely the NBI .dat files, ECH .dat file, and FICXS parquet file. 
These files have to be retrieved by the user before lhd_ficxs_py can be used.
The second argument ``{los}`` is an integer (6 or 7) that refers to the FICXS sightline (line-of-sight or LOS).
There are two LOS for the LHD FICXS system: 6-O and 7-O.
Details about the two sightlines can be found in [1]_, [2]_.

References
++++++++++
.. [1] Fujiwara, Y., Kamio, S., Yamaguchi, H., Garcia, A.V., Stagner, L., Nuga, H., Seki, R., Ogawa, K., Isobe, M., Yokoyama, M., Fast-ion D alpha diagnostic with 3D-supporting FIDASIM in the Large Helical Device *Nuclear Fusion* **60** (2020) 112014 doi:`10.1088/1741-4326/abae84 <https://doi.org/10.1088/1741-4326/abae84>`_.
.. [2] Hayashi, W.H.J., Heidbrink, W.W., Muscatello, C.M., Lin, D.J., Osakabe, M., Ogawa, K., Kawamoto, Y., Yamaguchi, H., Seki, R., Nuga, H., Charge-exchange measurements of high-energy fast ions in LHD using negative-ion neutral beam injection *Journal of Instrumentation* **19** (2024) P12006 doi:`10.1088/1748-0221/19/12/P12006 <https://doi.org/10.1088/1748-0221/19/12/P12006>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/modules
