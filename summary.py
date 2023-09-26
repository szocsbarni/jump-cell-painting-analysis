"""
Text based summary generation
Author: Szocs Barna
"""

from cell_segmentation import CellExtendedStatistics


class Summary:
    """
    Helper class to define logic for CSV based summary creation
    """

    @classmethod
    def generateShortSummaryCSV(cls, cell_data: dict, delimiter: str):

        header = "row{}column{}field{}number_of_cells\n".format(delimiter, delimiter, delimiter)
        content = ""

        for data_frame_id in cell_data.keys():
            r, c, f = data_frame_id
            nr_of_cells = len(cell_data[data_frame_id][0])
            content += "{:d}{}{:d}{}{:d}{}{:d}\n".format(r, delimiter, c, delimiter, f, delimiter, nr_of_cells)

        return header + content

    @classmethod
    def generateDetailedCellSummary(cls, cell_data: dict, delimiter: str):
        header = "row" + delimiter + "column" + delimiter + CellExtendedStatistics.exportHeaderToCSV(delimiter) + "\n"
        content = ""
        for data_frame_id in cell_data.keys():
            r, c, f = data_frame_id
            for channel_id in cell_data[data_frame_id][1]:
                for cell_id in cell_data[data_frame_id][1][channel_id]:
                    cell_origin = "{:02d}{}{:02d}{}".format(r, delimiter, c, delimiter)
                    id_prefix = "f{:02d}-ch{:02d}-".format(f, channel_id)
                    content += cell_origin + cell_data[data_frame_id][1][channel_id][cell_id].exportToCSV(delimiter=delimiter, prefix_identifier=id_prefix) + "\n"
        return header + content
