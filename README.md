# GMDN-EMDN Mapping Project

This repository contains the code described in the paper "[*A Data-driven Algorithm Streamlining Medical Device
Nomenclature Interoperability Through Large Language Models*]()"
by [Riccardo Gibello](https://orcid.org/0009-0004-7092-0029), [Yijun Ren](https://orcid.org/0000-0001-5361-2447),
and [Enrico Gianluca Caiani](https://orcid.org/0000-0002-1770-6486).

The repository provides an easy-to-use tool to build all the outputs described in the paper, from the extraction
of the data to the training of the models. This project has been implemented to allow users to easily rebuild an
updated version of the `emdn_gmdn_fda.csv` dataset, and to reuse it to compare the paper results with more
sophisticated mapping algorithms. We thank
the [Italian Ministry of Health](https://www.salute.gov.it/portale/home.html),
the [U.S. Food and Drug Administration](https://www.fda.gov/), and
the [European Commission](https://commission.europa.eu/index_en) for providing the datasets used in this project with
public access.

<div style="text-align: center;">
  <img src="./readme_images/logos.png" alt="Project data providers" style="max-height: 100px;"/>
</div>
If you want to use this work in your research, please cite the following paper:

```bibtex
@article{gibelloDataDriven2024,
  title={A Data-driven Algorithm Streamlining Medical Device Nomenclature Interoperability Through Large Language Models},
  author={Gibello, Riccardo and Ren, Yijun and Caiani, Enrico Gianluca},
  journal={},
  year={2024}
}
```

Moreover, it is possible to access all the input, output, and model files used in the paper through the following
Zenodo repository:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13890612.svg)](https://doi.org/10.5281/zenodo.13890612)

Access to the data contained in this Zenodo repository can be requested by following the instructions in the
landing page of the repository.

Feel free to fork this GitHub repository and to contribute to the project.
For more information, please refer to the [CONTRIBUTING section](./CONTRIBUTING.md).
If you have any questions or issues, please contact us
at [riccardo.gibello@polimi.it](mailto:riccardo.gibello@polimi.it) or open a GitHub issue.

## Requirements

The project has been developed and tested by using Python 3.11. To install Python, refer to
the [Python download page](https://www.python.org/downloads/). Every other Python requirement will be installed by
running the `setup_environment.py` file (see [Installation](#installation) section).

To run the data extraction pipeline, it is necessary to have a PostgreSQL database installed. To install it, refer to
the [PostgreSQL download page](https://www.postgresql.org/).

To run the entire pipeline, there are no specific CPU or GPU requirements. However, to utilize the GPU for model
training, you need a compatible NVIDIA GPU and the CUDA toolkit. To install the CUDA toolkit, refer to
the [CUDA download page](https://developer.nvidia.com/cuda-downloads). Additionally, **at least 32-64 GB of RAM is
required to handle peak usage of up to 27 GB during cross-checking of the American and Italian datasets**. If you have
less RAM, you can request an updated version of the `emdn_gmdn_fda.csv` dataset from the authors.

## Installation

To properly set up the Python environment, run in a terminal the following command:

```bash
python .\setup_environment.py
```

This command will run the script `setup_environment.py` which will create a virtual environment in the current root
of the project and install the required packages. Note that this action could take a few minutes. The script handles
the installation so that the Python packages taking advantage of the GPU are installed only if a GPU is available.

To set up the PostgreSQL environment, it is needed to create a *root* account. Save the credentials for this account, as
they will be needed to run the first phase of the pipeline (e.g., *postgres* as username and *password* as password).

## Input directory setup

Before running the pipeline, the input directory must be set up. The input directory must contain the following files,
that can be retrieved from the [Zenodo repository](link) under the `SourceData` folder:

- `EMDN.csv`, containing the EMDN dataset. For an updated version, please refer to
  the [EMDN download page](https://webgate.ec.europa.eu/dyna2/emdn/).
- `foiclass.zip`, containing the FDA Product Code Classification Database. For an updated version, please refer
  to
  the [FDA Product Code Classification Database page](https://www.fda.gov/medical-devices/classify-your-medical-device/download-product-code-classification-files).
- `GUDID_full_release.zip`, containing all the FDA GUDID data of American medical devices. For an updated version,
  please refer to the
  [GUDID download page](https://accessgudid.nlm.nih.gov/download) and download the **LATEST FULL RELEASE**.
- `italian_medical_device_full_list.csv`, containing the Italian medical devices' dataset. For an updated version,
  please refer to
  the [Italian medical devices download page](https://www.dati.salute.gov.it/it/dataset/dispositivi-medici/).

## Pipeline execution

Every output file created during the pipeline execution is placed under the `OutputData` folder.
If this is not existing, it will be created in the root of the project at the start of the pipeline execution.

The pipeline consists of two main phases:

- The data extraction and preprocessing phase, in which the device and nomenclature data are extracted and cross-checked
  between the Italian and American datasets to find the correspondences of EMDN-GMDN-FDA codes. The result of the
  pipeline consists of the creation of the `emdn_gmdn_fda.csv` file in the `OutputData/SqlCsvTables` folder. Other CSV
  files will be present in the folder, but not necessary to be understood for the project purposes.
- The model training phase, in which a baseline and data-driven GMDN-EMDN mapping models are built and evaluated on the
  `emdn_gmdn_fda.csv` dataset. All the trained EMDN category predictor models and visual train-test results are stored
  under `OutputData/Models/ffnn/*`. All the data-driven and baseline model results are stored under
  `OutputData/ValidationResults/GmdnEmdn/ffnn/*`.

To execute the pipeline, it is possible to run the pipeline either on Windows, Linux, or macOS. On Windows, the pipeline
can be executed by running the following command in a terminal:

```bash
./run.bat <PostgreSQL username> <PostgreSQL password>
```

where `<PostgreSQL username>` and `<PostgreSQL password>` are the credentials of the PostgreSQL database. On Linux and
macOS, the pipeline can be executed by running the following command in a terminal:

```bash
./run.sh <PostgreSQL username> <PostgreSQL password>
```

where `<PostgreSQL username>` and `<PostgreSQL password>` are the credentials of the PostgreSQL database.

During the execution, the pipeline will track the amount of carbon emissions produced through the Python package
`codecarbon`. An output file will be stored under the `OutputData/Logging` folder as `emissions.csv`. Please, refer to
the [CodeCarbon documentation](https://mlco2.github.io/codecarbon/) for details on how to interpret the results.

Moreover, the total pipeline execution time is recorded and stored in separate files, one for each phase, under the
`OutputData/ComputationFiles/PipelineComputation` folder.

## License

This project’s source code is licensed under the GNU General Public License (GPL), version 3 or later. See the full
license text in the [GNU_LICENSE.txt](./GNU_LICENSE.txt) file. For more information, please refer to the
[LICENSE](./LICENSE.md) page.

## Code of Conduct

This project has adopted
the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct.html). By
participating in this project, you agree to abide by its terms. For more information, please refer to the
[CODE_OF_CONDUCT](./CODE_OF_CONDUCT.md) page.

