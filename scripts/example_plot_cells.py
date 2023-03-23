import pandas as pd
from image_analysis.plot_cells import plot_example_cells

barcodes = ['TGTCTAGG', 'TGGCATTC']
df_data = pd.read_csv( 'all_output_data_plate_10_well_C2_test_20230320.csv' )
plot_example_cells( df_data, barcodes, 'example_cells_test.png'  )
