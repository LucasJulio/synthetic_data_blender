import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import os

# TODO: make generic
MAIN_PATH = '/home/ribeiro-desktop/POLI/TCC/blender_experiments/datasets/'
INPUTS_SUBPATH = 'inputs'
ANNOTATIONS_SUBPATH = 'annotations'
MAPS_SUBPATH = 'maps'
_DESCRIPTION = "PCB synthetic images dataset"
_NUM_SHARDS = 1


class PCB(tfds.core.GeneratorBasedBuilder):
    """
    PCB synthetic images dataset
    """

    def __init__(self, dataset_name):
        super().__init__()
        self.full_path = MAIN_PATH + dataset_name
    VERSION = tfds.core.Version("0.1.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "file_name": tfds.features.Text(),
                "image": tfds.features.Image(),
                "segmentation_mask": tfds.features.Image(shape=(None, None, 1)),
            }),
            supervised_keys=("image", "segmentation_mask"),
        )

    def _split_generators(self, dl_manager):
        images_path_dir = os.path.join(self.full_path, INPUTS_SUBPATH)
        annotations_path_dir = os.path.join(self.full_path, ANNOTATIONS_SUBPATH)

        # Setup train and test splits
        train_split = tfds.core.SplitGenerator(
            name="train",
            gen_kwargs={
                "images_dir_path": images_path_dir,
                "annotations_dir_path": annotations_path_dir,
                "images_list_file": os.path.join(annotations_path_dir,
                                                 "trainval.txt"),
            },
        )
        test_split = tfds.core.SplitGenerator(
            name="test",
            gen_kwargs={
                "images_dir_path": images_path_dir,
                "annotations_dir_path": annotations_path_dir,
                "images_list_file": os.path.join(annotations_path_dir,
                                                 "test.txt")
            },
        )
        return [train_split, test_split]

    def _generate_examples(self, images_dir_path, annotations_dir_path,
                           images_list_file):

        with tf.io.gfile.GFile(images_list_file, "r") as images_list:
            for line in images_list:

                image_name = line.strip()
                map_name = "m_" + image_name + ".png"
                input_name = "i_" + image_name + ".png"

                maps_dir_path = os.path.join(self.full_path, MAPS_SUBPATH)
                record = {
                    "file_name": input_name,
                    "image": os.path.join(images_dir_path, input_name),
                    "segmentation_mask": os.path.join(maps_dir_path, map_name)
                }
                yield image_name, record
