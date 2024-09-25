# ops_analysis

## Installation instructions
Download the repository, then go into the `ops_analysis/` directory.

Create a conda environment:
```
conda create --name sbs_2023 python=3.8
```

Activate the conda environment:
```
conda activate sbs_2023
```

From within the `ops_analysis` directory:
```
pip install -r requirements.txt
pip install -e .
```

Typical installation time: <5 minutes.

See `example_image_analysis` subdirectory for examples of using this code to analyze images.
