# epytope: Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
