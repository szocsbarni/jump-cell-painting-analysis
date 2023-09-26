"""
Functions for exploratory plots and UMAP dimensionality reduction
Author: Szocs Barna
"""

import matplotlib.pyplot as plt
import umap
import numpy as np
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from cell_segmentation import CellExtendedStatistics


def build_ds_group_by_well_per_channel(cell_dataset: OrderedDict, channel_id, attr1_name, attr2_name):
    """
    Organizes the two selected cell attribute data by well (a unique row & column combination identifies a well)
    :param cell_dataset: the dataset produced by cell segmentation
    :param channel_id: the channel to be used when extracting cell intensity attributes, ranges between 1 and 5
    :param attr1_name: one of the field names defined in cell_segmentation.CellExtendedStatistics
    :param attr2_name: one of the field names defined in cell_segmentation.CellExtendedStatistics
    :return: a dictionary where every entry is a well, and every value consists from 3 list:
                - first list contains the values specified by attr1_name
                - second list contains the values specified by attr2_name
                - third list is a list of identifiers for the cells used to generate the above two lists
    """
    wells = {}
    for data_frame_id in cell_dataset.keys():
        r, c, f = data_frame_id
        # create unique identifier for the well
        well_name = "r{:02d}c{:02d}".format(r, c)
        for cell_id in cell_dataset[data_frame_id][1][channel_id]:
            # locate cell in input dataset
            cell: CellExtendedStatistics = cell_dataset[data_frame_id][1][channel_id][cell_id]
            # export attribute data
            if well_name not in wells.keys():
                # init lists
                cell_id = "f{:02d}-ch{:02d}-{:03d}".format(f, channel_id, cell.identifier)
                wells[well_name] = [[getattr(cell, attr1_name)], [getattr(cell, attr2_name)], [cell_id]]
            else:
                # dict entry already initialized
                wells[well_name][0].append(getattr(cell, attr1_name))
                wells[well_name][1].append(getattr(cell, attr2_name))
                wells[well_name][2].append("f{:02d}-ch{:02d}-{:03d}".format(f, channel_id, cell.identifier))
    return wells


def plot_channel_data_by_well(cell_dataset: OrderedDict, attr1_name, attr2_name, channel_id=2, title="", note="", show=True):
    """
    Do a simple 2D scatter plot, using the specified attribute values of all cells. Pixel intensity data is taken from the specified channel.
    :param cell_dataset: the dataset produced by cell segmentation
    :param attr1_name: one of the field names defined in cell_segmentation.CellExtendedStatistics
    :param attr2_name: one of the field names defined in cell_segmentation.CellExtendedStatistics
    :param channel_id: the channel to be used when extracting cell intensity attributes, ranges between 1 and 5
    :param title: text to be used as figure title
    :param note: text to be used as figure footnote
    :param show: boolean flag whether to block the thread to display the figure or not
    :return: nothing, but blocks execution if show==True
    """
    # create easy to plot dataset
    wells = build_ds_group_by_well_per_channel(cell_dataset, channel_id, attr1_name, attr2_name)

    # init plotting
    fig = plt.figure(title)
    ax = fig.subplots()

    # define colors to be used
    colors = plt.colormaps['Set1'].colors

    # plot cell data from each well
    color_count = 0
    for well_name in wells:
        sc = plt.scatter(wells[well_name][0], wells[well_name][1], label=well_name, color=colors[color_count], linewidths=2, marker="o", s=2)
        color_count += 1

    # figure meta info
    plt.xlabel(attr1_name)
    plt.ylabel(attr2_name)
    plt.legend()
    plt.title(title)
    plt.figtext(0.99, 0.01, note, horizontalalignment='right')
    if show:
        plt.show()


def exploratory_plots(cell_dataset: OrderedDict):
    """
    Create some exploratory plots using the result of cell segmentation
    :param cell_dataset: the dataset produced by cell segmentation
    :return: nothing, blocks execution by plots
    """
    # plot cell area vs cell perimeter grouped by well
    plot_channel_data_by_well(cell_dataset=cell_dataset, attr1_name="cell_area", attr2_name="perimeter", title="Cell area vs cell perimeter",
                              note="lower cluster is probably incorrect detections on the image margins", show=False)

    # plot cell major axis vs cell minor axis
    plot_channel_data_by_well(cell_dataset=cell_dataset, attr1_name="major_axis", attr2_name="minor_axis", title="Cell major axis vs cell minor axis",
                              note="confirms the validity of major vs minor axis selection -> no minor axis is bigger then a major", show=False)

    # plot nucleus area vs nucleus var intensity
    plot_channel_data_by_well(cell_dataset=cell_dataset, attr1_name="nucleus_area", attr2_name="nucleus_mean_intensity", channel_id=4,
                              title="Nucleus area vs nucleus var intensity from channel {:02d}".format(4), note="Nothing conclusive", show=False)

    # nucleus 10th percentile intensity vs nucleus 90th percentile intensity
    plot_channel_data_by_well(cell_dataset=cell_dataset, attr1_name="nucleus_10th_percentile_intensity", attr2_name="nucleus_90th_percentile_intensity",
                              channel_id=5, title="Nucleus area vs nucleus 90th percentile intensity from channel {:02d}".format(5),
                              note="Nothing conclusive", show=False)
    plt.show()


def build_merged_ds_from_all_channels_for_umap(cell_dataset: OrderedDict) -> (list, list):
    """
    Build a table-like dataset, list of lists, where every row represents one cell, and every column holds attributes values belonging to the same cell. Cell
    identifier data (well, field, id) are returned as a separate list, while respecting the same data order.
    :param cell_dataset: the dataset produced by cell segmentation
    :return: two lists:
        - first list is a list of list with shape of [number_of_cells x number_of_cell_attributes]
        - second list is list of string based identifiers (origin descriptors) for the cells used to generate the above list
    """
    # initialize return values
    data = []
    labels = []

    # iterate over every detected cell
    for data_frame_id in cell_dataset.keys():
        r, c, f = data_frame_id
        well_name = "r{:02d}c{:02d}".format(r, c)
        for cell_id in cell_dataset[data_frame_id][0]:
            # locate cell
            cell_basic = cell_dataset[data_frame_id][0][cell_id]
            # copy basic attributes
            cell_data = [cell_basic.cell_area, cell_basic.nucleus_area, cell_basic.major_axis, cell_basic.minor_axis, cell_basic.perimeter]
            # copy intensity attributes based on every channel
            for channel_id in range(1, 6):
                cell: CellExtendedStatistics = cell_dataset[data_frame_id][1][channel_id][cell_id]
                cell_data.extend([cell.cell_var_intensity, cell.cell_mean_intensity, cell.cell_10th_percentile_intensity, cell.cell_90th_percentile_intensity,
                                  cell.nucleus_var_intensity, cell.nucleus_mean_intensity, cell.nucleus_10th_percentile_intensity,
                                  cell.nucleus_90th_percentile_intensity,
                                  cell.cytoplasm_var_intensity, cell.cytoplasm_mean_intensity, cell.cytoplasm_10th_percentile_intensity,
                                  cell.cytoplasm_90th_percentile_intensity])

            # save for return
            data.append(cell_data)
            labels.append(well_name)
    return data, labels


def generate_umap_cell_attributes_2D_global(data: list, labels: list):
    """
    Utilizes UMAP (https://umap-learn.readthedocs.io) data dimension reduction functionality to obtain a 2D feature set for the input data. The input data is
    expected to hold column-wise attributes and row-wise samples. The values are normalized using z-scores, while the input data is not modified.
    :param data: list of lists, where ery row is a sample and every column is an attribute
    :param labels: list of sample identifier strings
    :return: nothing, blocks execution for plotting
    """
    # verify input shape
    if len(data) != len(labels):
        return None

    # initialize umap process
    reducer = umap.UMAP()

    # normalize input data
    scaled_data = StandardScaler().fit_transform(data)

    # perform dimensionality reduction
    reduced_data = reducer.fit_transform(scaled_data)

    # plot the resulting feature set
    label_color_map = {}
    label_counter = 0
    unique_labels = list(set(labels))
    for label in unique_labels:
        label_color_map[label] = label_counter
        label_counter += 1
    label_index = [label_color_map[i] for i in labels]
    plt.figure("Dimension Reduction with UMAP using (basic+intensity) cell attributes from all channels")
    sc = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=label_index, cmap=plt.colormaps['Set1'])

    # figure meta data
    plt.title("Dimension Reduction with UMAP using (basic+intensity) cell attributes from all channels")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(handles=sc.legend_elements()[0], labels=unique_labels)

    plt.show()


def generate_umap_cell_attributes_3D_global(data: list, labels: list):
    """
    Utilizes UMAP (https://umap-learn.readthedocs.io) data dimension reduction functionality to obtain a 3D feature set for the input data. The input data is
    expected to hold column-wise attributes and row-wise samples. The values are normalized using z-scores, while the input data is not modified.
    :param data: list of lists, where ery row is a sample and every column is an attribute
    :param labels: list of sample identifier strings
    :return: nothing, blocks execution for plotting
    """
    # verify input shape
    if len(data) != len(labels):
        return None

    # initialize umap process
    reducer = umap.UMAP(n_components=3)

    # normalize input data
    scaled_data = StandardScaler().fit_transform(data)

    # perform dimensionality reduction
    reduced_data = reducer.fit_transform(scaled_data)

    # plot the resulting feature set
    label_color_map = {}
    label_counter = 0
    uniq_labels = list(set(labels))
    for label in uniq_labels:
        label_color_map[label] = label_counter
        label_counter += 1
    label_index = np.asarray([label_color_map[i] for i in labels])
    fig = plt.figure("3D UMAP data for each well and data from all channels")
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=label_index, cmap=plt.colormaps['Set1'])

    # figure meta data
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    plt.title("Dimension Reduction (component=3) with UMAP using cell (basic+intensity) attributes from every channel")
    plt.legend(handles=sc.legend_elements()[0], labels=uniq_labels)

    plt.show()
