# Run example image analysis

This example demonstrates how to analyze raw image data: starting from the SBS and phenotype tiff files, this example walks through the steps required to get to a final csv file with cells, their properties (condensates, etc.), and associated barcode(s).

## <ins>**Set up**</ins>: 
1. Copy this directory to a location where you want to run the analysis. Then within the directory, download the required raw data (supplementary data XX from Kappel et al, XXXX). You should have two directories within the `example_data_analysis` directory: `raw_SBS_images` and `raw_phenotype_images`.

2. Set up SBS analysis code (slightly modified version of Blainey Lab code): <br>
   `git clone git@github.com:kkappel1/OpticalPooledScreens2023.git`
   Then follow the installation instructions, including creation of conda environment `sbs_2023`.

## <ins>**Steps**</ins>: <br>
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
*Expected run time*: ~5 minutes. <br>
*Expected output*: ` process_cellpose_nophenotype_all/` <br>

4. Segment the nuclei in the phenotype images:
```
python {PATH_TO_OPS_ANALYSIS}/ops_analysis/image_analysis/segment_nuclei_cellpose.py -plate_num 6 -well_num B5 -out_tag example -output_tif_40x_base_dir process_phenotype/ -use_gpu -list_dapi_files
```
**Replace `{PATH_TO_OPS_ANALYSIS}` with the path to your `ops_analysis/` directory.** <br>
*Expected run time*: ~5-10 minutes. <br>
*Expected output*: `nuclei_masks_plate_6_well_B5_example/` and `process_phenotype/` <br>

5. Match the phenotype images to the SBS images:
```
python match_sbs_phenotype_tiles.py 
```
*Expected run time*: ~10 minutes. <br>
*Expected output*: `match_brute_force_plate_6_well_B5/` and `match_all_plate_6_well_B5/` <br>

6. Analyze the phenotype images:
```
python phenotype_and_match_cells_no_segment.py
```
*Expected run time*: ~5 minutes. <br>
*Expected output*: `phenotype_data_plate_6_well_B5_example.csv` (all data for the phenotyped cells), `all_output_data_plate_6_well_B5_example.csv` (matched phenotyped images), and `process_phenotype/plate_6/` <br>

7. Check for and remove any duplicate cells (possibly created if images were collected with any overlap):
```
python {PATH_TO_OPS_ANALYSIS}/ops_analysis/image_analysis/postprocess_well.py -data_file all_output_data_plate_6_well_B5_example.csv -bad_barcode_file '' -num_proc 6 -no_overwrite
```
**Replace `{PATH_TO_OPS_ANALYSIS}` with the path to your `ops_analysis/` directory.** <br>
*Expected run time*: < 1 minute. <br>
*Expected output*: `all_output_data_plate_6_well_B5_example_no_duplicates.csv` <br>

8. Filter the matches between the phenotype and SBS cell images:
```
python {PATH_TO_OPS_ANALYSIS}/ops_analysis/image_analysis/filter_matches_well.py -data_file all_output_data_plate_6_well_B5_example_no_duplicates.csv
```
**Replace `{PATH_TO_OPS_ANALYSIS}` with the path to your `ops_analysis/` directory.** <br>
*Expected run time*: < 1 minute. <br>
*Expected output*: `all_output_data_plate_6_well_B5_example_no_duplicates_filter_matches.csv` —> this is the final output containing cells, barcodes, and all phenotype information (condensates, etc.) <br>
