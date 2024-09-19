# epytope: Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v3.4.0 - 2024-01-10

### `Added`

-  [#93](https://github.com/KohlbacherLab/epytope/pull/93) Added NetMHCIIpan version 4.2 interface [#92](https://github.com/KohlbacherLab/epytope/issues/92)
-  [#96](https://github.com/KohlbacherLab/epytope/pull/96) Added NetMHCIIpan version 4.3 interface [#95](https://github.com/KohlbacherLab/epytope/issues/95)

### `Changed`

### `Fixed`

- [#93](https://github.com/KohlbacherLab/epytope/pull/93) Fixed length restriction of NetMHCIIpan interfaces [#91](https://github.com/KohlbacherLab/epytope/issues/91)

## v3.3.1 - 2023-04-25

### `Added`

### `Changed`

- [#85](https://github.com/KohlbacherLab/epytope/pull/85) Pinned pandas version >= 1.3.5 as requested by @e-dorigatti [#84](https://github.com/KohlbacherLab/epytope/issues/84).

### `Fixed`

- [#83](https://github.com/KohlbacherLab/epytope/pull/83) Fixed erroneous mouse allele parsing for PSSM predictors [#82](https://github.com/KohlbacherLab/epytope/issues/82)

## v3.3.0 - 2022-12-01

### `Added`

### `Changed`

- Switch from [PyVCF](https://github.com/jamescasbon/PyVCF) to [PyVCF3](https://github.com/dridk/PyVCF3) as dependency

### `Fixed`

## v3.2.0 - 2022-11-02

### `Added`

- [#74](https://github.com/KohlbacherLab/epytope/pull/74) - Add function to `Peptide` class to determine if peptide originates from a variant

### `Changed`

- [#63](https://github.com/KohlbacherLab/epytope/pull/63) - Outsource supported alleles of prediction tools [#61](https://github.com/KohlbacherLab/epytope/issues/61)
- [#69](https://github.com/KohlbacherLab/epytope/pull/69) - Improve `Ensembl` BioMart adapter [#57](https://github.com/KohlbacherLab/epytope/issues/57)

### `Fixed`

- [#73](https://github.com/KohlbacherLab/epytope/pull/73) - Fix `NetMHCII 4.0` parser [#72](https://github.com/KohlbacherLab/epytope/issues/72)

## v3.1.0 - 2022-06-15

### `Added`

- [#58](https://github.com/KohlbacherLab/epytope/pull/58) - Add check for `BioMart` transcript sequence availability [#55](https://github.com/KohlbacherLab/epytope/issues/55)
- [#59](https://github.com/KohlbacherLab/epytope/pull/59) - Add interface for `NetMHCpan 4.1` [#56](https://github.com/KohlbacherLab/epytope/issues/56)
- [#66](https://github.com/KohlbacherLab/epytope/pull/66) - Add interface for `NetMHCIIpan 4.1` [#65](https://github.com/KohlbacherLab/epytope/issues/65)

### `Changed`

- [#62](https://github.com/KohlbacherLab/epytope/pull/62) - Update the supported alleles of `syfpeithi` [#60](https://github.com/KohlbacherLab/epytope/issues/60)

### `Fixed`

- [#53](https://github.com/KohlbacherLab/epytope/pull/53) - Fix `ANN` predictor results [#52](https://github.com/KohlbacherLab/epytope/issues/52)

## v3.0.0 - 2022-01-26

Initial release of `epytope`. `epytope` is the successor project of [`FRED2`](https://github.com/FRED-2/Fred2), which was renamed to a more versioning friendly base name.

### `Added`

- [#6](https://github.com/KohlbacherLab/epytope/pull/6) - Add CI for external tools
- [#24](https://github.com/KohlbacherLab/epytope/pull/24) - Add `keras` dependency
- [#26](https://github.com/KohlbacherLab/epytope/pull/26) - Add a license file
- [#31](https://github.com/KohlbacherLab/epytope/pull/31) - Add deployment to `PyPI`
- [#42](https://github.com/KohlbacherLab/epytope/pull/42) - Add new `Syfpeithi` matrices
- [#46](https://github.com/KohlbacherLab/epytope/pull/46) - Add support for `NetMHCII 2.3` and `NetMHCIIpan 4.0`

### `Changed`

- [#1](https://github.com/KohlbacherLab/epytope/pull/1) - Switch CI/CD from Travis to GitHub Actions
- [#9](https://github.com/KohlbacherLab/epytope/pull/9) - Initial `Python 2` to `Python 3` conversion based on 2to3conv
- [#20](https://github.com/KohlbacherLab/epytope/pull/20) - Use logging module rather than print calls across the library
- [#23](https://github.com/KohlbacherLab/epytope/pull/23) - Rename the package from `FRED-2` to `epytope`
- [#25](https://github.com/KohlbacherLab/epytope/pull/25) - Refactor CI to use more `pip` and less `conda`
- [#42](https://github.com/KohlbacherLab/epytope/pull/42) - Extend `EpitopePredictionResult` structure to store `rank`-based scores

### `Fixed`

- [#11](https://github.com/KohlbacherLab/epytope/pull/11) - Fix `Python` version matrix in CI, remove versions that fail
- [#16](https://github.com/KohlbacherLab/epytope/pull/16) - Fix epitope prediction 2to3 bugs and tests
- [#17](https://github.com/KohlbacherLab/epytope/pull/17) - Fix Invalid subprocess handling and `mhcflurry` polluting stdout
- [#18](https://github.com/KohlbacherLab/epytope/pull/18) - Fix external (`NetMHC` tool family) epitope prediction error and temp file handling
- [#46](https://github.com/KohlbacherLab/epytope/pull/46) - Fix issues with MHC class-II `CombinedAlleles` [#45](https://github.com/KohlbacherLab/epytope/issues/45)
- [#46](https://github.com/KohlbacherLab/epytope/pull/46) - Do not override `Allele` objects [#38](https://github.com/KohlbacherLab/epytope/issues/38)
