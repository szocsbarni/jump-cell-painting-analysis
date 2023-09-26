# JUMP Cell Painting dataset analysis
The collection of methods used to analyze the JUMP cell painting dataset(https://jump-cellpainting.broadinstitute.org/). The required dependencies are as follows:

`opencv scipy umap-learn matplotlib PIL numpy pickle pathlib`

Please update these packages to the latest versions as some used functionalities are not available in older ones.

## Usage
The main entry point script is the `analyze_imageset.py`, and running it with `--help` argument will give a detailed rundown of the available argument options. Below though one can find some usefull argument combinations.

### Analyze whole dataset
To analyze all the images in the dataset and re-generate the exported channel 2 images with overlaid segmentation the following command can be used: 

`analyze_imageset.py --data-folder /path/to/images --summary-folder /path/to/save/csv/summary --export-plots-to /path/to/save/the/plots -vv`

This will generate the segmented images, with highlighted boundaries for nucleus/cell and a summary table in CSV format with intensity properties for cell/nucleus/cytoplasm regions of each located cell. Example properties: area, mean intensity, perimeter, major axes, 90th percentile intenisty, etc..

### Visual guide for the image processing workflow
To get a detailed visual overview of the step by step image processing workflow, while not using all the images from the input set, but only a few, handpicked images, use the following command:

`analyze_imageset.py --data-folder /path/to/images --interactive --selected -vv`

### Export data to pickle
To export the cell segmentation result to a pickle file, use the following command:

`analyze_imageset.py --data-folder /path/to/images -p /path/to/cell_data.pickle -vv`

This can come handy, because the processing of all images can take up to a few minutes, and this intermediary pickle file can be used later for data analysis without waiting for the whole data process workflow to be repeated.

### Exploratory plots
Making use of the exported pickle file to create some exploratory plots, in an attempt to distinguish groups of wells:

`analyze_imageset.py -p /path/to/cell_data.pickle --show-exploratory -vv`

### Dimensionality reduction
Apply UMAP dimensionality reduction, using the saved pickle data to visualize the data in 2D/3D:

`analyze_imageset.py -p /path/to/cell_data.pickle --attempt-dim-reduction -vv`

