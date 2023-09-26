"""
Main script to trigger the JUMP Cell painting dataset segmentation and then process the data. Run with --help to see the list of arguments.
Author: Szocs Barna
"""

import logging
import os
import pickle
import sys
import argparse
import pathlib
import numpy as np
from collections import OrderedDict

from io_operations import TiffImageSet
from io_operations import ImageWriter

from cell_segmentation import CellSegmenter
from cell_segmentation import CellStatistics
from cell_segmentation import CellExtendedStatistics
from cell_segmentation import ImageIntensityAnalysis

from summary import Summary

from data_analysis import exploratory_plots
from data_analysis import generate_umap_cell_attributes_2D_global
from data_analysis import generate_umap_cell_attributes_3D_global
from data_analysis import build_merged_ds_from_all_channels_for_umap

# set of images handpicked for development and short interactive sessions
complex_case_list = [(4, 8, 3), (12, 9, 4), (12, 9, 2), (12, 9, 7), (4, 8, 7), (12, 9, 3)]


def saveImageTask1(row, column, field, image_ch2_norm_segmented_blend, path):
    """
    Save image to disk in a 8 bit per pixel format, using 4 layers, RGBA. Builds the file name back in the same format as the input fileset was.
    :param row: row number of well
    :param column: column number of well
    :param field: image field
    :param image_ch2_norm_segmented_blend:  pixel-matrix to export
    :param path: file path where to save the image
    :return:
    """
    file_name = "task1_r{:02d}c{:02d}f{:02d}p01-ch2sk1fk1fl1".format(row, column, field)
    logger.warning("Saving task1 image: {}".format(file_name))
    ImageWriter.saveRGBA32BImageTIFF((image_ch2_norm_segmented_blend >> 8).astype(np.uint8), path, file_name)
    # ImageWriter.saveRGBA64BImagePNG(image_ch2_norm_segmented_blend, path, file_name)


def analyze(folder_path: pathlib.Path, export_path: pathlib.Path, interactive: bool, selection_only: bool) -> OrderedDict:
    """
    Main logic  to iterate through the JUMP dataset, do the IO operations, load the images into memory and organize them. First channel 5 and channel 2 images
    are used as a pair for cell segmentation, then the obtained cell/nucleus masks are used to export the intensity based metrics. Finally a dictionary is built
    where every entry corresponds to an image in the input dataset.
    :param folder_path: input folder path where the cell images are located
    :param export_path: output folder path, used to store the highlighted cell/nucleus masks images
    :param interactive: boolean flag, if set to True will show step by step interactive plot for low level image processing stages of each image frame
    :param selection_only: use a hard-coded selection of image frames instead fo the full dataset
    :return: a dictionary, where each entry is a unique tupple of (row, column, field, channel) and the value is dictionary too, holding pairs of cell IDs and
    cell_segmentation.CellExtendedStatistics objects.
    """
    img_set = TiffImageSet(folder_path, 30)
    img_set.loadImages()

    # initialize cell segmentation
    segmenter = CellSegmenter(np.uint16)
    logger.debug(segmenter)

    # define input set
    if selection_only:
        key_list = complex_case_list
    else:
        key_list = img_set.getSortedImageSetKeys()

    # initialize return variable
    cell_data = OrderedDict()
    frame_counter = 0
    cell_counter = 0
    # iterate through every selected image
    for item in key_list:
        r, c, f = item

        img5 = img_set.getImageAt(r, c, f, 5)
        img2 = img_set.getImageAt(r, c, f, 2)

        if img5 is None or img2 is None:
            logger.warning("Image set (row: {} column: {} field: {}) is incomplete in dataset.")
            continue

        logger.warning("Analyzing image from row: {} column: {} field: {} ({:0.1f}%)".format(r, c, f, frame_counter / len(key_list) * 100))

        figure_title = "row: {} column: {} field: {}".format(r, c, f)

        # do cell segmentation on channel 5 and channel 2 image pair, export image if requested
        if export_path is not None:
            cells, cell_segments, filtered_nucleus_segments, image_ch2_norm_segmented_blend = \
                segmenter.segmentImagePair(img5, img2, plot_intermediate=interactive, figure_title=figure_title, export_segmented_images=True)
            saveImageTask1(r, c, f, image_ch2_norm_segmented_blend, export_path)
        else:
            cells, cell_segments, filtered_nucleus_segments = \
                segmenter.segmentImagePair(img5, img2, plot_intermediate=interactive, figure_title=figure_title, export_segmented_images=False)

        # extend detected cell info with pixel intensity stats from all channels
        channel_based_intensity_stats = {}
        missing = False
        for channel_id in img_set.getAvailableChannels():
            channel_img = img_set.getImageAt(r, c, f, channel_id)
            if channel_img is None:
                logger.error("Expected channel {} image in image set row: {} column: {} field: {}, but not found.".format(channel_id, r, c, f))
                missing = True
                break
            requested_percentiles = [10, 90]
            # export pixel intensity metrics from selected image
            cell_intensity_stats = ImageIntensityAnalysis.calculateIntensityForLabels(channel_img, cell_segments, cells.keys(), requested_percentiles)
            nucleus_intensity_stats = \
                ImageIntensityAnalysis.calculateIntensityForLabels(channel_img, filtered_nucleus_segments, cells.keys(), requested_percentiles)
            cytoplasm_intensity_stats = \
                ImageIntensityAnalysis.calculateIntensityForLabels(channel_img, cell_segments - filtered_nucleus_segments, cells.keys(), requested_percentiles)

            # extended data object of each cell with the new values
            intensity_stat_per_image = {}
            for cell_label in cells:
                stat = CellExtendedStatistics(cells[cell_label])
                stat.cell_mean_intensity, stat.cell_var_intensity = cell_intensity_stats[cell_label][:2]
                stat.cell_10th_percentile_intensity, stat.cell_90th_percentile_intensity = cell_intensity_stats[cell_label][2]

                stat.nucleus_mean_intensity, stat.nucleus_var_intensity = nucleus_intensity_stats[cell_label][:2]
                stat.nucleus_10th_percentile_intensity, stat.nucleus_90th_percentile_intensity = nucleus_intensity_stats[cell_label][2]

                stat.cytoplasm_mean_intensity, stat.cytoplasm_var_intensity = cytoplasm_intensity_stats[cell_label][:2]
                stat.cytoplasm_10th_percentile_intensity, stat.cytoplasm_90th_percentile_intensity = cytoplasm_intensity_stats[cell_label][2]

                intensity_stat_per_image[cell_label] = stat

            channel_based_intensity_stats[channel_id] = intensity_stat_per_image

        if missing:
            logger.error("Dataset is not complete, image set of (row: {} column: {} field: {}) is missing image of channel {}.".format(r, c, f, channel_id))
            return None

        cell_data[item] = (cells, channel_based_intensity_stats)

        frame_counter += 1
        cell_counter += len(cells)

    logger.warning("Completed image processing phase, processed {} image sets, localized {} cells.".format(frame_counter, cell_counter))

    return cell_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool to analyze the JUMP Test dataset", prog="cell_analysis")
    parser.add_argument('--data-folder', '-d', required=False, type=pathlib.Path,
                        help='path to the folder containing the images, not needed if pickle file is provided')
    parser.add_argument('--summary-folder', '-s', required=False, type=pathlib.Path, help='folder to store summary csv')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="increase verbosity on std out, increase further with multiplying -v. For ex: -vvv")
    parser.add_argument('--export-plots-to', '-o', required=False, type=pathlib.Path, help='path to folder to save the plots to')
    parser.add_argument('--interactive', '-i', required=False, action="store_true", default=False,
                        help='run in interactive mode which will plot the results as they are generated on the fly')
    parser.add_argument('--selected', required=False, action="store_true", default=False,
                        help='use only a hard-coded, pre-selected image set from the database if present, '
                             'otherwise iterate through all the images in the dataset')
    parser.add_argument('--delimiter', required=False, default=";", type=str, help="fields delimiter character to be used for summary csv files")
    parser.add_argument('--pickle-file', '-p', required=False, type=pathlib.Path, help="location of the pickle file which contains all exported cell details."
                                                                                       "If defined, then all other flags are ignored (except --verbose) and "
                                                                                       "cell/nucleus/cytoplasm data is imported from this file. Useful for data"
                                                                                       " analysis and exploratory plots.")
    parser.add_argument("--attempt-dim-reduction", required=False, action="store_true", default=False,
                        help='boolean flag signaling whether to attempt cell attribute dimensionality reduction or not')
    parser.add_argument("--show-exploratory", required=False, action="store_true", default=False, help='boolean flag to control plotting exploratory figures')

    args = parser.parse_args()

    # setup logging
    logger = logging.getLogger("cell_analysis")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(name)s] %(levelname)-8s %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    if args.verbose <= 4:
        log_level = logging.CRITICAL - args.verbose * 10
        stdout_handler.setLevel(log_level)
    else:
        log_level = logging.DEBUG
        stdout_handler.setLevel(log_level)

    logger.addHandler(stdout_handler)

    if args.pickle_file is None or (args.pickle_file is not None and not os.path.exists(args.pickle_file)):

        if args.data_folder is not None:
            if not os.path.exists(args.data_folder):
                logger.critical("Input data path does not exist, quitting...")
                sys.exit(1)

        logger.critical("No exported data, analyze images from {}. This will take a few minutes...".format(args.data_folder))

        if args.export_plots_to is not None:
            if not os.path.exists(args.export_plots_to):
                logger.critical("Export path does not exist, quitting...")
                sys.exit(1)

        if args.summary_folder is not None:
            if not os.path.exists(args.summary_folder):
                logger.critical("Summary folder path does not exist, quitting...")
                sys.exit(1)

        # analyze images
        cell_dataset = analyze(args.data_folder, args.export_plots_to, args.interactive, args.selected)
        if cell_dataset is None:
            logger.critical("Abort analysis process, dataset incomplete.")
            sys.exit(1)

        # cell data is ready, task 1 images are exported
        # export summary files for task 1 and 4 if requested
        if args.summary_folder is not None:
            with open(os.path.join(args.summary_folder, 'short_summary.csv'), "w") as stream:
                stream.write(Summary.generateShortSummaryCSV(cell_data=cell_dataset, delimiter=args.delimiter))
                stream.flush()
            logger.warning("Wrote short summary to: {}".format(os.path.join(args.summary_folder, 'short_summary.csv')))

        if args.summary_folder is not None:
            with open(os.path.join(args.summary_folder, 'detailed_cell_summary.csv'), "w") as stream:
                stream.write(Summary.generateDetailedCellSummary(cell_data=cell_dataset, delimiter=args.delimiter))
                stream.flush()
            logger.warning("Wrote detailed cell summary to: {}".format(os.path.join(args.summary_folder, 'detailed_cell_summary.csv')))

        # export data if required
        if args.pickle_file is not None:
            with open(args.pickle_file, "wb") as bin_stream:
                pickle.dump(cell_dataset, bin_stream, protocol=4)
    else:
        with open(args.pickle_file, "rb") as bin_stream:
            cell_dataset = pickle.load(bin_stream)

    # exploratory plots
    if args.show_exploratory:
        logger.warning("Plotting exploratory figures...")
        exploratory_plots(cell_dataset)

    # dimensionality reduction
    if args.attempt_dim_reduction:
        logger.info("Reorganizing data for dimensionality reduction...")
        cell_data_table, cell_data_labels = build_merged_ds_from_all_channels_for_umap(cell_dataset=cell_dataset)
        logger.warning("Starting dimensionality reduction process, target component number = 2, this may take several seconds...")
        generate_umap_cell_attributes_2D_global(cell_data_table, cell_data_labels)
        logger.warning("Starting dimensionality reduction process, target component number = 3, this may take several seconds...")
        generate_umap_cell_attributes_3D_global(cell_data_table, cell_data_labels)

    logger.info("Exiting...")

