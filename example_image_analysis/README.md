# Run example image analysis

This example demonstrates how to analyze raw image data: starting from the SBS and phenotype tiff files, this example walks through the steps required to get to a final csv file with cells, their properties (condensates, etc.), and associated barcode(s).

<ins>Set up</ins>: 
1. Copy this directory to a location where you want to run the analysis. Then within the directory, download the required raw data (supplementary data XX from Kappel et al, XXXX). You should have two directories within the `example_data_analysis` directory: `raw_SBS_images` and `raw_phenotype_images`.
2. Set up SBS analysis code (slightly modified version of Blainey Lab code): <br>
   `git clone git@github.com:kkappel1/OpticalPooledScreens2023.git`
   Then follow the installation instructions, including creation of conda environment `sbs_2023`.

<ins>Steps</ins>: <br>
*These steps are designed to be run with 6 CPUs and 1 GPU, but can be run with as little as a single CPU (run times will be longer though).*
1. Activate the conda environment: <br>
```
conda activate sbs_2023
```
2. Prepare the SBS files for analysis: <br>
```
python prepare_input_files_sbs.py
```

   *Expected run time*: < 1 minute. <br>
   *Expected output*: `input/` <br>

3. Run the SBS image analysis:
```
snakemake --cores 6 -s OpticalPooledScreens_nophenotype.smk --configfile config_cellpose_nophenotype.yaml
```
4. 
