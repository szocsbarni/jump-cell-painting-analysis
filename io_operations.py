"""
IO operations
Author: Szocs Barna
"""

import os
import logging
import numpy as np
from PIL import Image
import cv2


class TiffImageSet:
    """
    Class responsible for reading in the TIFF images as well as organizing them based on the naming convention
    """

    def __init__(self, parent_folder, image_name_len):
        self._parent_folder = parent_folder
        self._image_name_len = image_name_len
        self._read_process_finished = False
        self._images = {}
        self._logger = logging.getLogger("cell_analysis")
        self._rows = set()
        self._cols = set()
        self._fields = set()
        self._channels = set()

    def _interpret_filename(self, name):
        if len(name) != self._image_name_len:
            self._logger.debug("Image file name ({}) not matching expected length of {}.".format(name, self._image_name_len))
            return None
        else:
            if name[1:3].isdigit() and name[4:6].isdigit() and name[7:9].isdigit() and name[15].isdigit():
                row = int(name[1:3])
                column = int(name[4:6])
                field = int(name[7:9])
                channel = int(name[15])

                return row, column, field, channel
            else:
                self._logger.debug("Image file name ({}) contains invalid characters.".format(name))
                return None

    def loadImages(self):
        files = os.listdir(self._parent_folder)
        for file in files:
            f_abs_path = os.path.join(self._parent_folder, file)
            if os.path.isfile(f_abs_path) and f_abs_path.endswith(".tiff"):
                ret = self._interpret_filename(file)
                if ret is None:
                    self._logger.warning("Image file ({}) is not following naming convention, skipping.".format(file))
                    continue
                row, column, field, channel = ret
                self._rows.add(row)
                self._cols.add(column)
                self._fields.add(field)
                self._channels.add(channel)

                image = Image.open(f_abs_path)
                self._images[(row, column, field, channel)] = np.array(image, dtype=np.uint16)

        self._logger.warning("Loaded {} tiff files.".format(len(self._images.keys())))
        self._read_process_finished = True

    def getImageSetKeys(self) -> set:
        all_image_keys = self._images.keys()
        image_set_only_keys = set()
        for key in all_image_keys:
            image_set_only_keys.add(key[:-1])
        return image_set_only_keys

    def getSortedImageSetKeys(self):
        return sorted(list(self.getImageSetKeys()))

    def getAvailableRows(self):
        return self._rows

    def getAvailableColumns(self):
        return self._cols

    def getAvailableFields(self):
        return self._fields

    def getAvailableChannels(self):
        return self._channels

    def getImageAt(self, row, column, field, channel):
        image_id = (row, column, field, channel)
        if image_id in self._images.keys():
            return self._images[image_id]
        else:
            self._logger.debug("Hash miss for image at row: {} column: {} field: {} channel: {}.".format(row, column, field, channel))
            return None


class ImageWriter:
    """
    Helper class to chain opencv image saving functionality
    """

    @classmethod
    def saveRGBA64BImagePNG(cls, image, path, filename):
        path_output = os.path.join(path, filename + '.png')
        cv2.imwrite(path_output, image)

    @classmethod
    def saveRGBA32BImageTIFF(cls, image, path, filename):
        path_output = os.path.join(path, filename + '.tiff')
        cv2.imwrite(path_output, image)
