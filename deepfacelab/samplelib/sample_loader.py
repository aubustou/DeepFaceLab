from __future__ import annotations

import logging
import operator
import traceback
from pathlib import Path
from deepfacelab import config

from deepfacelab.samplelib import packed_faceset
from deepfacelab.core import pathex
from deepfacelab.core.interact import interact as io
from deepfacelab.core.joblib import Subprocessor
from deepfacelab.core.mplib import MPSharedList
from deepfacelab.facelib import FaceType
from deepfacelab.dflimg import dflimg

from .sample import Sample, SampleType

logger = logging.getLogger(__name__)


samples_cache = dict()

def get_person_id_max_count(samples_path):
    samples = None
    try:
        samples = packed_faceset.load(samples_path)
    except:
        io.log_err(
            f"Error occured while loading samplelib.PackedFaceset.load {str(samples_path)}, {traceback.format_exc()}"
        )

    if samples is None:
        raise ValueError("packed faceset not found.")
    persons_name_idxs = {}
    for sample in samples:
        persons_name_idxs[sample.person_name] = 0
    return len(list(persons_name_idxs.keys()))

def load(sample_type, samples_path, subdirs=False):
    """
    Return MPSharedList of samples
    """

    if str(samples_path) not in samples_cache.keys():
        samples_cache[str(samples_path)] = [None] * SampleType.QTY

    samples = samples_cache[str(samples_path)]

    if sample_type == SampleType.IMAGE:
        if samples[sample_type] is None:
            samples[sample_type] = [
                Sample(filename=filename)
                for filename in io.progress_bar_generator(
                    pathex.get_image_paths(samples_path, subdirs=subdirs), "Loading"
                )
            ]

    elif sample_type == SampleType.FACE:
        if samples[sample_type] is None:
            try:
                result = packed_faceset.load(samples_path)
            except:
                io.log_err(
                    f"Error occured while loading samplelib.PackedFaceset.load {str(samples_dat_path)}, {traceback.format_exc()}"
                )
            else:
                if result is not None:
                    io.log_info(
                        f"Loaded {len(result)} packed faces from {samples_path}"
                    )

                if result is None:
                    result = load_face_samples(
                        pathex.get_image_paths(samples_path, subdirs=subdirs)
                    )

                samples[sample_type] = MPSharedList(result)
    elif sample_type == SampleType.FACE_TEMPORAL_SORTED:
        result = load(SampleType.FACE, samples_path)
        result = upgrade_to_face_temporal_sorted_samples(result)
        samples[sample_type] = MPSharedList(result)

    return samples[sample_type]

def load_face_samples(image_paths):
    result = FaceSamplesLoaderSubprocessor(image_paths).run()
    sample_list = []

    for filename, data in result:
        if data is None:
            continue
        (
            face_type,
            shape,
            landmarks,
            seg_ie_polys,
            xseg_mask_compressed,
            eyebrows_expand_mod,
            source_filename,
        ) = data

        sample_list.append(
            Sample(
                filename=filename,
                sample_type=SampleType.FACE,
                face_type=FaceType.fromString(face_type),
                shape=shape,
                landmarks=landmarks,
                seg_ie_polys=seg_ie_polys,
                xseg_mask_compressed=xseg_mask_compressed,
                eyebrows_expand_mod=eyebrows_expand_mod,
                source_filename=source_filename,
            )
        )
    return sample_list

def upgrade_to_face_temporal_sorted_samples(samples):
    new_s = [(s, s.source_filename) for s in samples]
    new_s = sorted(new_s, key=operator.itemgetter(1))

    return [s[0] for s in new_s]


class FaceSamplesLoaderSubprocessor(Subprocessor):
    # override
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.image_paths_len = len(image_paths)
        self.idxs = [*range(self.image_paths_len)]
        self.result = [None] * self.image_paths_len
        super().__init__("FaceSamplesLoader", FaceSamplesLoaderSubprocessor.Cli, 60)

    # override
    def on_clients_initialized(self):
        io.progress_bar("Loading samples", len(self.image_paths))

    # override
    def on_clients_finalized(self):
        io.progress_bar_close()

    # override
    def process_info_generator(self):
        for i in range(config.CONFIG.cpu_number):
            yield "CPU%d" % (i), {}, {}

    # override
    def get_data(self, host_dict):
        if len(self.idxs) > 0:
            idx = self.idxs.pop(0)
            return idx, self.image_paths[idx]

        return None

    # override
    def on_data_return(self, host_dict, data):
        self.idxs.insert(0, data[0])

    # override
    def on_result(self, host_dict, data, result):
        idx, dflimg = result
        self.result[idx] = (self.image_paths[idx], dflimg)
        io.progress_bar_inc(1)

    # override
    def get_result(self):
        return self.result

    class Cli(Subprocessor.Cli):
        # override
        def process_data(self, data):
            idx, filename = data
            dflimg_ = dflimg.load(Path(filename))

            if dflimg_ is None or not dflimg_.has_data():
                self.log_err(f"FaceSamplesLoader: {filename} is not a dfl image file.")
                data = None
            else:
                data = (
                    dflimg_.get_face_type(),
                    dflimg_.get_shape(),
                    dflimg_.get_landmarks(),
                    dflimg_.get_seg_ie_polys(),
                    dflimg_.get_xseg_mask_compressed(),
                    dflimg_.get_eyebrows_expand_mod(),
                    dflimg_.get_source_filename(),
                )

            return idx, data

        # override
        def get_data_name(self, data):
            # return string identificator of your data
            return data[1]
