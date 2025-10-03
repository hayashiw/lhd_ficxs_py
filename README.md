# lhd_ficxs_py
## Python tools to analyze fast ion data from the Large Helical Device.

This repository is primarily concerned with analysis of data from the fast-ion charge-exchange spectroscopy (FICXS) system on the Large Helical Device (LHD).

### Reading in data
LHD data is typically stored in comma-separated ASCII files with the .dat extension.

Most .dat files can be read in using the `read_data` subroutine. This will return the data as a `pandas.DataFrame` object.
```python
from lhd_ficxs_py import read_data

dat_file = './nb1pwr_temporal.dat'
nb1_df = read_data(dat_file, use_postgres_names=True, convert_to_ms=True)
# use_postgres_names converts column names to lower-snake-case format
# convert_to_ms converts the time column to milliseconds
pinj = nb1_df['pport_through_nb1']
pinj_units = nb1_df.attrs['units'] # MW
```

### Handling FICXS data files
Calibrated FICXS data is often written to "ficxs_2_calib" files. Due to the size of these files, file IO becomes a bottleneck when analyzing multiple cases. It is helpful to compress FICXS .dat files into parquet files. This can be easily done using the `lhd_ficxs_py.io.convert_dat_to_pqt` subroutine. They can then be quickly read in using `pandas.read_parquet`. An additional subroutine `lhd_ficxs_py.io.read_ficxs_from_pqt` is available to handle formatting the wavelength column as a dataframe index and to handle formatting of labels.

### Calculating timing for active FICXS signal
The main purpose of this package is to generate the diagnostic-beam-on and -off time stamps for the FICXS system. This can be done using `lhd_ficxs_py.run` from the command line:
```
(lhd_ficxs_py_env): python -m lhd_ficxs_py.run {shot_num} {los}
```
The first argument `{shot}` should be replaced with the LHD discharge number. This is used to set the paths for the input data files, namely the NBI .dat files, ECH .dat file, and FICXS parquet file. These files have to be retrieved by the user before lhd_ficxs_py can be used. The second argument `{los}` is an integer (6 or 7) that refers to the FICXS sightline (line-of-sight or LOS). There are two LOS for the LHD FICXS system: 6-O and 7-O. Details about the two sightlines can be found in [[1]](#1) [[2]](#2).

## Requirements
This package was written using the following:
- python 3.13.5
    - matplotlib 3.10.6
    - numpy 2.3.3
    - pandas 2.3.3
    - pyarrow 21.0.0
    - scipy 1.16.2

## References
<a id="1">[1]</a> Fujiwara, Y., Kamio, S., Yamaguchi, H., Garcia, A.V., Stagner, L., Nuga, H., Seki, R., Ogawa, K., Isobe, M., Yokoyama, M., Fast-ion D alpha diagnostic with 3D-supporting FIDASIM in the Large Helical Device *Nuclear Fusion* **60** (2020) 112014 doi:[10.1088/1741-4326/abae84](https://doi.org/10.1088/1741-4326/abae84)

<a id="2">[2]</a> Hayashi, W.H.J., Heidbrink, W.W., Muscatello, C.M., Lin, D.J., Osakabe, M., Ogawa, K., Kawamoto, Y., Yamaguchi, H., Seki, R., Nuga, H., Charge-exchange measurements of high-energy fast ions in LHD using negative-ion neutral beam injection *Journal of Instrumentation* **19** (2024) P12006 doi:[10.1088/1748-0221/19/12/P12006](https://doi.org/10.1088/1748-0221/19/12/P12006)