import numpy as np
import pickle
import os
import urllib
import zipfile 
import tarfile
from six.moves import urllib

import matplotlib.pyplot as plt

class DatasetManager(object):

    def __init__(self, data_url, dataset_name, dataset_dir, img_size, num_channels, num_classes):
        # URL for the data-set on the internet.
        self.data_url = data_url

        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir

        # Width and height of each image.
        self.img_size = img_size

        # Number of channels in each image, 3 channels: Red, Green, Blue.
        self.num_channels = num_channels

        # Length of an image when flattened to a 1-dim array.
        self.img_size_flat = img_size * img_size * num_channels

        # Number of classes.
        self.num_classes = num_channels

        ########################################################################
        # Various constants used to allocate arrays of the correct size.

        # Number of files for the training-set.
        self._num_files_train = 5

        # Number of images for each batch-file in the training-set.
        self._images_per_file = 10000

        # Total number of images in the training-set.
        # This is used to pre-allocate arrays for efficiency.
        self._num_images_train = self._num_files_train * self._images_per_file


    ########################################################################
    # Private functions for downloading, unpacking and loading data-files.

    def download(self, base_url, filename, download_dir):
        """
        Download the given file if it does not already exist in the download_dir.
        :param base_url: The internet URL without the filename.
        :param filename: The filename that will be added to the base_url.
        :param download_dir: Local directory for storing the file.
        :return: Nothing.
        """

        # Path for local file.
        save_path = os.path.join(download_dir, filename)

        # Check if the file already exists, otherwise we need to download it now.
        if not os.path.exists(save_path):
            # Check if the download directory exists, otherwise create it.
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)

            print("Downloading", filename, "...")

            # Download the file from the internet.
            url = base_url + filename
            file_path, _ = urllib.request.urlretrieve(url=url,
                                                      filename=save_path)

            print(" Done!")


    def maybe_download_and_extract(self):
        """
        Download and extract the data if it doesn't already exist.
        Assumes the url is a tar-ball file.
        :param url:
            Internet URL for the tar-file to download.
            Example: "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        :param download_dir:
            Directory where the downloaded file is saved.
            Example: "data/CIFAR-10/"
        :return:
            Nothing.
        """

        download_dir=f'/tmp/{self.dataset_name}/'
        url = self.data_url

        # Filename for saving the file downloaded from the internet.
        # Use the filename from the URL and add it to the download_dir.
        filename = url.split('/')[-1]
        file_path = os.path.join(download_dir, filename)

        # Check if the file already exists.
        # If it exists then we assume it has also been extracted,
        # otherwise we need to download and extract it now.
        if not os.path.exists(file_path):
            # Check if the download directory exists, otherwise create it.
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)

            # Download the file from the internet.
            file_path, _ = urllib.request.urlretrieve(url=url,
                                                      filename=file_path)

            print()
            print("Download finished. Extracting files.")

            if file_path.endswith(".zip"):
                # Unpack the zip-file.
                zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
            elif file_path.endswith((".tar.gz", ".tgz")):
                # Unpack the tar-ball.
                tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

            print("Done.")
        else:
            print("Data has apparently already been downloaded and unpacked.")


    def _get_file_path(self, filename=""):
        """
        Return the full path of a data-file for the data-set.

        If filename=="" then return the directory of the files.
        """

        return os.path.join(f'/tmp/{self.dataset_name}/{self.dataset_dir}/', filename)


    def _unpickle(self, filename):
        """
        Unpickle the given file and return the data.

        Note that the appropriate dir-name is prepended the filename.
        """

        # Create full path for the file.
        file_path = self._get_file_path(filename)

        print("Loading data: " + file_path)

        with open(file_path, mode='rb') as file:
            # In Python 3.X it is important to set the encoding,
            # otherwise an exception is raised here.
            data = pickle.load(file,encoding='bytes')

        return data


    def _convert_images(self, raw):
        """
        Convert images from the dataset format and
        return a 4-dim array with shape: [image_number, height, width, channel]
        where the pixels are floats between 0.0 and 1.0.
        """

        # Convert the raw images from the data-files to floating-points.
        # raw_float = np.array(raw, dtype=float) / 255.0

        # Reshape the array to 4-dimensions.
        images = raw.reshape([-1, self.num_channels, self.img_size, self.img_size])

        # Reorder the indices of the array.
        images = images.transpose([0, 2, 3, 1])

        return images


    def _load_data(self, filename):
        """
        Load a pickled data-file from the data-set
        and return the converted images (see above) and the class-number
        for each image.
        """

        # Load the pickled data-file.
        data = self._unpickle(filename)

        # Get the raw images.
        raw_images = data[b'data']

        # Get the class-numbers for each image. Convert to numpy-array.
        cls = np.array(data[b'labels'])

        # Convert the images.
        images = self._convert_images(raw_images)

        return images, cls


    def load_class_names(self):
        # Load the class-names from the pickled file.
        raw = self._unpickle(filename="batches.meta")[b'label_names']

        # Convert from binary strings.
        names = [x.decode('utf-8') for x in raw]

        return names


    def load_training_data(self):
        """
        Load all the training-data for the data-set.
        The data-set is split into 5 data-files which are merged here.
        Returns the images, class-numbers and one-hot encoded class-labels.
        """

        # Pre-allocate the arrays for the images and class-numbers for efficiency.
        images = np.zeros(shape=[self._num_images_train, self.img_size, self.img_size, self.num_channels], dtype=np.uint8)
        cls = np.zeros(shape=[self._num_images_train], dtype=int)

        # Begin-index for the current batch.
        begin = 0

        # For each data-file.
        for i in range(self._num_files_train):
            # Load the images and class-numbers from the data-file.
            images_batch, cls_batch = self._load_data(filename="data_batch_" + str(i + 1))

            # Number of images in this batch.
            num_images = len(images_batch)

            # End-index for the current batch.
            end = begin + num_images

            # Store the images into the array.
            images[begin:end, :] = images_batch

            # Store the class-numbers into the array.
            cls[begin:end] = cls_batch

            # The begin-index for the next batch is the current end-index.
            begin = end

        return images, cls


    def load_validation_data(self, num_images=0):
        images, cls = self._load_data(filename="test_batch")
        if num_images > 0:
          images = images[num_images:, :, :, :]
          cls = cls[num_images:]
        return images, cls

    def load_testing_data(self, num_images=0):
        images, cls = self._load_data(filename="test_batch")
        if num_images > 0:
          images = images[:num_images, :, :, :]
          cls = cls[:num_images]
        return images, cls
