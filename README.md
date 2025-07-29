<p align="center">
  <img src="https://img.shields.io/badge/Status-Completed-blue?style=flat-square" alt="Status"/>
  <img src="https://img.shields.io/github/license/GabrielMSchmidt/YOUR-REPO-NAME?style=flat-square&color=blue" alt="License"/>
</p>

# Exoplanet Identification Using Deep Learning Techniques

This repository contains the source code and materials developed for my Final Year Project (Bachelor's Thesis) in Computer Science at the **Midwestern Parana State University (UNICENTRO)**.

The project's objective was to develop and evaluate Deep Learning models (CNN, InceptionTime, and Mamba) for the classification of exoplanets from time-series data measuring a star's light intensity, known as light curves.

The dataset can be publicly accessed at [Astronet Dataset](https://zenodo.org/records/7411579).


## 🛠️ Tech Stack

This project was developed using some of the main tools from the Python Data Science ecosystem.

[![My Skills](https://skillicons.dev/icons?i=python,pytorch,sklearn,tensorflow)](https://skillicons.dev)

## 📁 Basic Project Structure

The main project files are listed below:
```
├── .idea
├── datasets/               <- Different pre-processed versions of the raw Dataset
├── models/                 <- Deep Learning Models: CNN, InceptionTime and Mamba
├── utils/                  <- Auxiliary execution scripts

├── LICENSE
├── main_datasets.py        <- Main script for dataset generation and pre-processing
├── main_model.py           <- Main script for model execution
├── README.md
└── tces_with_labels_v3.csv <- Auxiliary table for Dataset generation

```

## 🚀 Prerequisites

Dependencies required to use the project:
- Python
- TensorFlow
- Keras
- scikit-learn
- sktime
- astropy
- lightkurve
- matplotlib
- seaborn

To run the Mamba network, the following specifications are required:
- Linux
- NVIDIA GPU
- PyTorch 1.12+
- CUDA 11.6+
  
For more details, access the [official Mamba repository](https://github.com/state-spaces/mamba).

## 🤝 How to Contribute

Contributions are welcome! If you have suggestions for improving the project, feel free to follow these steps:

1.  **Fork** the project.
2.  Create a new Branch for your feature: `git checkout -b feature/MyFeature`
3.  **Commit** your changes: `git commit -m 'feat: Add MyFeature'`
4.  **Push** to your Branch: `git push origin feature/MyFeature`
5.  Open a **Pull Request**.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
