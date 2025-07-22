# Supplementary code and data for "Technical note: Transit times of reactive solutes under time-variable hydrologic conditions"

This repository contains Python scripts, illustrative Jupyter notebooks and the original data that can be used to generate some basic results, figures, and analyses presented in the paper:

> **Technical note: Transit times of reactive solutes under time-variable hydrologic conditions**  
> Raphaël Miazza and Paolo Benettin  
> Institute of Earth Surface Dynamics, Université de Lausanne, Switzerland  
> _Submitted, 2025_  
> DOI

---

## Contents

`model_initialization.ipynb`: Jupyter notebook demonstrating how the model can be set up and run, with basic visualizations.  
`randomly_sampled_model.py`: Python script containing the randomly sampled hydrologic model.  
`Hydrologic_datacsv`: Comma separated values file containing the original hydrologic timeseries used to run the model.

---

## Environment Setup

We recommend using a Python virtual environment (e.g. conda) for reproducibility. The notebooks require the following Python libraries (as well as their dependencies):
- Numpy (https://numpy.org/)
- Pandas (https://pandas.pydata.org/)
- Matplotlib (https://matplotlib.org/)
