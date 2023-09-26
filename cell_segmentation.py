"""
Logic for low level image processing steps, as well as data classes used to store/organize the results.
Author: Szocs Barna
"""

import logging
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.signal import argrelextrema


class CellStatistics:
    """
    Class to define basic attributes of located cell.
    """

    def __init__(self, identifier, cell_area, nucleus_area, major_axis, minor_axis, perimeter):
        self.identifier = identifier
        self.cell_area = cell_area
        self.nucleus_area = nucleus_area
        self.major_axis = major_axis
        self.minor_axis = minor_axis
        self.perimeter = perimeter

    def __str__(self):
        return "[Cell ID:{:0.0f}] Nucleus Size: {:0.0f}\tCell Size: {:0.0f}\t Axes: [{:0.1f} {:0.1f}]\tPerimeter: {:0.1f}".format(
            self.identifier, self.nucleus_area, self.cell_area, self.major_axis, self.minor_axis, self.perimeter)

    def exportToCSV(self, delimiter=';'):
        return "{:0.0f}{}{:0.0f}{}{:0.0f}{}{:0.1f}{}{:0.1f}{}{:0.1f}".format(
            self.identifier, delimiter, self.nucleus_area, delimiter, self.cell_area, delimiter, self.major_axis, delimiter,
            self.minor_axis, delimiter, self.perimeter)


class CellExtendedStatistics(CellStatistics):
    """
    Class to extend the basic information about a cell with intensity based metrics.
    """

    def __init__(self, cell: CellStatistics):
        # copy over ancestor class fields
        self.identifier = cell.identifier
        self.cell_area = cell.cell_area
        self.nucleus_area = cell.nucleus_area
        self.major_axis = cell.major_axis
        self.minor_axis = cell.minor_axis
        self.perimeter = cell.perimeter

        # define additional fields
        self.cell_mean_intensity = -1
        self.nucleus_mean_intensity = -1
        self.cytoplasm_mean_intensity = -1

        self.cell_var_intensity = -1
        self.nucleus_var_intensity = -1
        self.cytoplasm_var_intensity = -1

        self.cell_10th_percentile_intensity = -1
        self.nucleus_10th_percentile_intensity = -1
        self.cytoplasm_10th_percentile_intensity = -1

        self.cell_90th_percentile_intensity = -1
        self.nucleus_90th_percentile_intensity = -1
        self.cytoplasm_90th_percentile_intensity = -1

    @classmethod
    def exportHeaderToCSV(cls, delimiter=';'):
        return "cell_id" + delimiter + "nucleus_area" + delimiter + "cell_area" + delimiter + "cell_major_axis" + delimiter + "cell_minor_axis" + delimiter + \
               "cell_perimeter" + delimiter + "cell_mean_intensity" + delimiter + "nucleus_mean_intensity" + delimiter + "cytoplasm_mean_intensity" + delimiter + \
               "cell_var_intensity" + delimiter + "nucleus_var_intensity" + delimiter + "cytoplasm_var_intensity" + delimiter + \
               "cell_10th_percentile_intensity" + delimiter + "nucleus_10th_percentile_intensity" + delimiter + "cytoplasm_10th_percentile_intensity" + \
               delimiter + "cell_90th_percentile_intensity" + delimiter + "nucleus_90th_percentile_intensity" + delimiter + \
               "cytoplasm_90th_percentile_intensity"

    def exportToCSV(self, delimiter=';', prefix_identifier=""):
        base_info = "{}{:d}{}{:0.0f}{}{:0.0f}{}{:0.1f}{}{:0.1f}{}{:0.1f}".format(
            prefix_identifier, self.identifier, delimiter, self.nucleus_area, delimiter, self.cell_area, delimiter, self.major_axis, delimiter,
            self.minor_axis, delimiter, self.perimeter)

        mean_intensity_info = "{:0.1f}{}{:0.1f}{}{:0.1f}".format(self.cell_mean_intensity, delimiter, self.nucleus_mean_intensity, delimiter,
                                                                 self.cytoplasm_mean_intensity)
        var_intensity_info = "{:0.1f}{}{:0.1f}{}{:0.1f}".format(self.cell_var_intensity, delimiter, self.nucleus_var_intensity, delimiter,
                                                                self.cytoplasm_var_intensity)
        percentile_10_info = "{:0.1f}{}{:0.1f}{}{:0.1f}".format(self.cell_10th_percentile_intensity, delimiter,
                                                                self.nucleus_10th_percentile_intensity, delimiter,
                                                                self.cytoplasm_10th_percentile_intensity)
        percentile_90_info = "{:0.1f}{}{:0.1f}{}{:0.1f}".format(self.cell_90th_percentile_intensity, delimiter,
                                                                self.nucleus_90th_percentile_intensity, delimiter,
                                                                self.cytoplasm_90th_percentile_intensity)
        return "{}{}{}{}{}{}{}{}{}".format(base_info, delimiter, mean_intensity_info, delimiter, var_intensity_info, delimiter, percentile_10_info, delimiter,
                                           percentile_90_info)


class CellSegmenter:
    """
    Class responsible for the main logic of low level image processing. Filtering is mainly done with opencv and numpy, tunable parameters are extracted
    and can be controlled during initialization.
    """
    _default_morph_kernel_size = 5
    _default_gamma_factor = 0.08
    _default_bg_fg_percentile = 0.98
    _default_bin_count = 256

    @classmethod
    def apply_close(cls, img, kernel_size=None):
        if kernel_size is None:
            kernel_size = cls._default_morph_kernel_size
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)))

    @classmethod
    def apply_tophat(cls, img, kernel_size=None):
        if kernel_size is None:
            kernel_size = cls._default_morph_kernel_size
        return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)))

    @classmethod
    def apply_sharpen(cls, img):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

    @classmethod
    def getAxesFromContour(cls, contour):
        # Note: opencv minAreaRect fits the rotated bounding box on the objects between the contour pixels
        rot_rectangle = cv2.minAreaRect(contour)
        w, h = rot_rectangle[1]
        # compensation for the lost margin
        w += 2
        h += 2
        return max(w, h), min(w, h)

    def __init__(self, input_image_pixel_type, morph_kernel_size=None, gamma_factor=None, bg_fg_percentile=None, bin_count=None):
        self._input_image_pixel_type = input_image_pixel_type
        self._min_pixel_intensity = np.iinfo(self._input_image_pixel_type).min
        self._max_pixel_intensity = np.iinfo(self._input_image_pixel_type).max
        self._uint8_min = np.iinfo(np.uint8).min
        self._uint8_max = np.iinfo(np.uint8).max
        # factor to downscale pixel to fit into one byte
        self._downscale_factor = int(round(np.log2(self._max_pixel_intensity) / 2))

        if morph_kernel_size is None:
            self._morph_kernel_size = self.__class__._default_morph_kernel_size

        if gamma_factor is None:
            self._gamma_factor = self.__class__._default_gamma_factor

        if bg_fg_percentile is None:
            self._bg_fg_percentile = self.__class__._default_bg_fg_percentile

        if bin_count is None:
            self._bin_count = self.__class__._default_bin_count

        self._logger = logging.getLogger("cell_analysis")

    def __str__(self):
        return "CellSegmenter instance with the following settings:\n\tmorphology kernel size: {}\n\tgamma factor: {}\n\tpercentile for background/foreground" \
               " separation: {}\n\tbin count for histograms: {}\n\tpixel type: {}\n".format(
                self._morph_kernel_size, self._gamma_factor, self._bg_fg_percentile, self._bin_count, self._input_image_pixel_type)

    def apply_gamma(self, img: np.ndarray, gamma_factor: float = None):
        """
        Apply gamma correction, input image is not modified, corrected copy is returned
        :param img: input image
        :param gamma_factor: gamma factor, range [0.1], if not defined, default is used
        :return: corrected copy of input image
        """
        if gamma_factor is None:
            gamma_factor = self._default_gamma_factor
        mean = np.mean(img)
        max_value = np.max(img)
        gamma = math.log(gamma_factor * max_value) / math.log(mean)

        # do gamma correction on value channel
        return np.power(img, gamma).clip(0, max_value).astype(self._input_image_pixel_type)

    def boundingBoxOfLabel(self, image: np.ndarray, label_id: int):
        """
        Construct a bounding box around the segment specified by the label_id inside the input image
        :param image: input image, not modified during process
        :param label_id: discrete label id
        :return: a tuple of 4: minimum and maximum of row index, respectively minimum and maximum of column index
        """
        binary_mask = np.zeros(image.shape)
        binary_mask[image == label_id] = 1
        rows, cols = np.where(binary_mask != 0)
        if len(rows) == 0 or len(cols) == 0:
            self._logger.debug("Label ID: {} is not present in input image".format(label_id))
            return None
        return min(rows), max(rows), min(cols), max(cols)

    def getContourAndAreaOfLabel(self, image: np.ndarray, label_id: int):
        """
        Create contour of the segment in input image located by label_id value, and calculates area of the segment selection.
        :param image: input image, not modified during process
        :param label_id: discrete label id
        :return: a tuple of 2: opencv contour object and area as integer
        """
        ret = self.boundingBoxOfLabel(image, label_id)
        if ret is None:
            self._logger.debug("Can not create bounding box for label ID: {}".format(label_id))
            return None
        # bounding box edge values
        min_row, max_row, min_col, max_col = ret
        # create binary mask plus padding to copy over shape mask from the image
        binary_mask = np.zeros((max_row - min_row + 3, max_col - min_col + 3), dtype=np.uint8)
        # place the shape in the center of the binary mask
        binary_mask[1:-2, 1:-2] = image[min_row:max_row, min_col:max_col].astype(np.uint8)
        # set only the requested label to max value, the object border is already on max value
        binary_mask[binary_mask == label_id] = self._uint8_max
        # clear every other object out
        binary_mask[binary_mask < self._uint8_max] = 0

        # sum up area
        area = np.sum(binary_mask != 0)

        # export contour
        contour = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contour) == 2:  # valid contour found
            return contour[0][0], area
        else:
            self._logger.debug("Finding contour for label ID: {} failed, while the area of the object is: {}".format(label_id, area))
            return None

    def segmentImagePair(self, image_ch_5: np.ndarray, image_ch_2: np.ndarray, plot_intermediate: bool = False, figure_title: str = "Not provided",
                         export_segmented_images: bool = False) -> (OrderedDict, np.ndarray, np.ndarray):
        """
        Main logic for the image pair cell segmentation process: after normalizing both images, channel 5 image is used first to locate nucleus contour
        (done with Otsu's method). The filtered nucleus masks are stored and the image from channel is processed to separate background and foreground. The
        obtained foreground information is merged together with the nucleus masks in order to be used as input for the watershed segmentation. The resulting
        cell segments are filtered and organized in a dictionary and returned.
        :param image_ch_5: input image from channel 5, not modified during process
        :param image_ch_2: input image from channel 2, not modified during process
        :param plot_intermediate: boolean flag, if True then intermediate plots are shown for each run, note that this will block the processing
        :param figure_title: figure title to be used for plotting
        :param export_segmented_images: boolean flag, if True then the resulting cell/nucleus segmentation is layered over the input image in a colored manner.
        :return: tuple of 3:
            - cell dictionary where each key is a cell ID and every value is a CellExtendedStatistics object
            - image mask for cell segments
            - image mask for nucleus segments
            if export_segmented_images is True, then one more object is appended:
            - channel 2 image copy with overlaid colored cell segments and highlighted nucleus contours
        """
        if plot_intermediate:
            # display input images, channel 5 and channel 2
            fig, axs = plt.subplots(2, 4)
            axs[0, 0].imshow(image_ch_5, cmap='gray')
            axs[0, 0].set_title("Channel 5, min: {:0.0f}, max: {:0.0f}".format(np.min(image_ch_5), np.max(image_ch_5)))
            axs[1, 0].imshow(image_ch_2, cmap='gray')
            axs[1, 0].set_title("Channel 2, min: {:0.0f}, max: {:0.0f}".format(np.min(image_ch_2), np.max(image_ch_2)))

        # normalize channel 5 image
        img5_norm = self.apply_gamma(cv2.normalize(image_ch_5, None, alpha=self._min_pixel_intensity, beta=self._max_pixel_intensity,
                                                   norm_type=cv2.NORM_MINMAX), gamma_factor=self._gamma_factor)
        if plot_intermediate:
            # display normalized channel 5 image
            axs[0, 1].imshow(img5_norm, cmap='gray')
            axs[0, 1].set_title("min: {:0.0f}, max: {:0.0f}".format(np.min(img5_norm), np.max(img5_norm)))

        # locate nucleus
        th5_val, img5_th = cv2.threshold(img5_norm, self._min_pixel_intensity, self._max_pixel_intensity, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        nucleus_number, nucleus_labels, _, _ = cv2.connectedComponentsWithStats((img5_th >> self._downscale_factor).astype(np.uint8), 8)

        # filter out too small / too big detections
        too_small_th = 70
        too_big_th = 10000
        nucleus_filter_out = np.zeros(image_ch_5.shape)
        for label in range(1, nucleus_number-1):
            nucleus_area = np.sum(nucleus_labels == label)
            if too_small_th > nucleus_area or too_big_th < nucleus_area:
                self._logger.info("Nucleus filtered out with label ID: {} and nucleus area: {}".format(label, nucleus_area))
                nucleus_filter_out[nucleus_labels == label] = 1

        if plot_intermediate:
            # display the contours of detected nucleus candidates
            input_for_contour = np.copy(nucleus_labels).astype(np.uint8)
            input_for_contour[input_for_contour > 0] = self._uint8_max
            contours, _ = cv2.findContours(input_for_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            nice_contour_base = np.copy(img5_norm >> self._downscale_factor).astype(np.uint8)
            nice_contour = cv2.cvtColor(nice_contour_base, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(nice_contour, contours, -1, (255, 0, 0), 1)

            axs[0, 2].imshow(nice_contour)
            axs[0, 2].set_title("Nr. of components: {}, th.: {:0.0f}".format(nucleus_number, th5_val))

            # show pixel intensity histogram of channel 5 image
            # axs[0, 3].hist(img5_norm.ravel(), bins=self._bin_count, range=(self._min_pixel_intensity, self._max_pixel_intensity))
            # axs[0, 3].set_title("Pixel intensity hist")

            # show resulting components
            axs[0, 3].imshow(nucleus_labels, cmap="jet")
            axs[0, 3].set_title("Segmentation Result")

        # normalize channel 2 image
        img2_norm = cv2.normalize(image_ch_2, None, alpha=self._min_pixel_intensity, beta=self._max_pixel_intensity, norm_type=cv2.NORM_MINMAX)
        img2_norm_gamma = self.apply_sharpen(self.apply_gamma(img2_norm, gamma_factor=self._gamma_factor))

        # identify threshold for background/foreground separation
        bin_values, _ = np.histogram(img2_norm_gamma.ravel(), bins=self._bin_count, range=(0, 65535))
        # reduce margins
        exclude = round(self._bin_count * ((1 - self._bg_fg_percentile) / 2))
        result = argrelextrema(bin_values[exclude:-exclude], np.less)
        th2_val = (result[0][0] + exclude) * self._bin_count

        if plot_intermediate:
            # display normalized channel 2 image
            axs[1, 1].imshow(img2_norm_gamma, cmap='gray')
            axs[1, 1].set_title("Normalized, backgr. th.: {}".format(th2_val))

        # binary image background/foreground separation
        _, binary = cv2.threshold(img2_norm_gamma, th2_val, self._uint8_max, cv2.THRESH_BINARY)
        # remove noise from binary image
        img2_no_back_binary = self.apply_close(binary, kernel_size=self._morph_kernel_size)

        # rearrange labels, watershed expects the background with label 1, the basin seeds with index > 1, and the area to be segmented with 0
        components = np.copy(nucleus_labels)
        # move all objects into the range of [2..n]
        components[components > 0] += 1
        # mark the confident background exported from channel 2, leave on 0 the area around the nucleus
        components[img2_no_back_binary == 0] = 1  # highly confident background is now 1, 0 is questionable
        components[nucleus_filter_out == 1] = 1           # mark filtered out regions as confident background, so they will not be part of any other segment

        if plot_intermediate:
            # visualization
            axs[1, 2].imshow(components.astype(np.uint8), cmap='jet')
            axs[1, 2].set_title("Watershed input labels")

        # watershed segmentation
        # remove background first
        img2_no_back = np.copy(img2_norm_gamma)
        img2_no_back[img2_no_back < th2_val] = 0
        # convert image to 1B/px (OpenCV constraint)
        img = (img2_no_back >> self._downscale_factor).astype(np.uint8)
        # watershed expects the search region to be white, set the foreground (segmented nucleus + unsegmented cytoplasm) to white
        img[components > 1] = self._uint8_max
        img[components == 0] = self._uint8_max
        # watershed OpenCV implementation expects a 3 layer grayscale image as input, create it
        fake_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # perform watershed segmentation
        cell_segments = cv2.watershed(fake_rgb, components)

        # prepare output structure
        cells = OrderedDict()
        # process results
        for label in range(2, nucleus_number + 1):
            # calculate nucleus area, data already there
            nucleus_area = np.sum(nucleus_labels == (label - 1))
            # create cell area mask and build it's contour
            contour_result = self.getContourAndAreaOfLabel(cell_segments, label)
            if contour_result is None:
                # filtering out
                nucleus_filter_out[nucleus_labels == (label - 1)] = 1
                cell_segments[cell_segments == label] = 1
                self._logger.info("Cell filtered out with label ID: {} and nucleus area: {}".format(label, nucleus_area))
                continue
            # calculate requested features
            contour, cell_area = contour_result
            major_axis_len, minor_axis_len = self.getAxesFromContour(contour)
            perimeter = cv2.arcLength(contour, closed=True)

            # filter out cells where no cytoplasm was found, probably noise
            if nucleus_area >= cell_area:
                nucleus_filter_out[nucleus_labels == (label - 1)] = 1
                cell_segments[cell_segments == label] = 1
                self._logger.debug("Cell filtered out with label ID: {}, nucleus area: {} and cell area: {}".format(label, nucleus_area, cell_area))
                continue

            # save valid cell
            cells[label] = CellStatistics(label, cell_area, nucleus_area, major_axis_len, minor_axis_len, perimeter)
            self._logger.info("{}".format(cells[label]))

        self._logger.warning("Located {} cells".format(len(cells)))

        if plot_intermediate:
            # visualize the resulting segments
            # final = cv2.cvtColor((img2_norm_gamma >> self._downscale_factor).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            # final[markers == -1] = [255, 0, 0]
            axs[1, 3].imshow(cell_segments)
            axs[1, 3].set_title("Filtered cell segments")

            fig.suptitle(figure_title)
            plt.show()

        # create nice nucleus mask with aligned cell IDS from segmentation
        filtered_nucleus_segments = np.copy(nucleus_labels)
        filtered_nucleus_segments[nucleus_filter_out == 1] = 0
        filtered_nucleus_segments[filtered_nucleus_segments > 0] += 1

        self._logger.debug("Cell IDs: {}".format(np.unique(cell_segments)))
        self._logger.debug("Nucleus IDs: {}".format(np.unique(filtered_nucleus_segments)))

        if export_segmented_images:
            # create 8-bit input for contour detection as opencv expects
            input_for_contour = np.copy(filtered_nucleus_segments).astype(np.uint8)
            input_for_contour[input_for_contour > 0] = self._uint8_max
            contours, hierarchy = cv2.findContours(input_for_contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # create 4-channel version of channel 2 image
            # image_ch2_base = np.copy(self.apply_gamma(img=img2_norm, gamma_factor=0.01))
            image_ch2_base = np.copy(image_ch_2)
            image_ch2_rgba = cv2.cvtColor(image_ch2_base, cv2.COLOR_GRAY2RGBA)

            # apply valid contours
            contour_color_rgba = (38505, 51765, 52785, 0)
            cv2.drawContours(image_ch2_rgba, contours, -1, contour_color_rgba, 1)

            # create overlay image for cell segments
            segmentation_overlay = np.zeros(image_ch2_rgba.shape, dtype=np.uint16)
            # list of colors used for cell segment painting
            colors = plt.colormaps['Paired'].colors
            color_counter = 0
            # draw only valid cell segments
            for cell_id in cells:

                color = (np.append(np.asarray(colors[color_counter]), 0) * self._max_pixel_intensity).astype(np.uint16)  # appended 0 is for alpha channel
                segmentation_overlay[cell_segments == cells[cell_id].identifier] = color
                color_counter = color_counter + 1 if color_counter < len(colors) - 1 else 0  # iterate over colors in cycles
            # blend segments with base image
            alpha = 0.9
            image_ch2_norm_segmented_blend = cv2.addWeighted(image_ch2_rgba, alpha, segmentation_overlay, 1 - alpha, 0)

            if plot_intermediate:
                # display the blend result
                blend_figure = plt.figure()
                plt.imshow(image_ch2_norm_segmented_blend / self._max_pixel_intensity)
                blend_figure.suptitle("AGP (Channel 2) + Overlay Cell Segment + Nucleus Contour")
                plt.show()

            return cells, cell_segments, filtered_nucleus_segments, image_ch2_norm_segmented_blend

        return cells, cell_segments, filtered_nucleus_segments


class ImageIntensityAnalysis:
    """
    Helper class to define logic for pixel intensity metric calculations using input image and input mask
    """

    @classmethod
    def intensityStats(cls, image, binary_mask, percentiles_req):

        roi = binary_mask > 0
        intensity_values = image[roi]
        mean_intensity = np.mean(intensity_values)
        variance_intensity = np.var(intensity_values)
        percentiles = np.percentile(intensity_values, percentiles_req)

        return mean_intensity, variance_intensity, percentiles

    @classmethod
    def calculateIntensityForLabels(cls, intensity_image, segmented_image, label_list, percentiles_req):
        intensity_stats = {}
        for label in label_list:
            binary_mask = segmented_image == label
            intensity_stats[label] = cls.intensityStats(intensity_image, binary_mask, percentiles_req)

        return intensity_stats


