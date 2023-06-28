# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""String Operations Dataset for fast model development"""


import csv
import json
import os
import re

import datasets
import gzip

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {String Operations Dataset: A small set of string manipulation tasks for fast model development},
author={Michael Granitzer},
year={2023}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
Minimal dataset for intended for LM development and testing using python string operations.
 The dataset is created by running different one line python string operations on random strings
 The idea is, that transformer implementation can learn the string operations and that this task is a good
 proxy tasks for other transformer operations on real languages and real tasks. Consequently, the
 data set is small and can be used in the development process without large scale infrastructures.
 
There are different configurations for the data set.

- `small`: contains below 50k instances of various string length and only contains slicing operations, i.e. all python operations expressable with `s[i:j:s]` (which also includes string reversal).
  - you can further choose different subsets according to either length or the kind of operation
- `small10`: like small, but only strings to length 10
- `small15`: like small, but only strings to length 15
- `small20`: like small, but only strings to length 20

The fields have the following meaning:

 - `input`: input string, i.e. the string and the string operation
 - `output`: output of the string operation
 - `code`: code for running the string operation in python,
 - `res_var`: name of the result variable
 - `operation`: kind of operation: 
   - `step_x` for `s[::x]`
   - `char_at_x` for `s[x]`
   - `slice_x:y` for `s[x:y]`
   - `slice_step_x:y:z` for `s[x:y:z]`
   - `slice_reverse_i:j:k` for `s[i:i+j][::k]`

 Siblings of `data` contain additional metadata information about the dataset.

 - `prompt` describes possible prompts based on that data splitted into input prompts / output prompts


"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://huggingface.co/PaDaS-Lab/SynStOp"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "Apache 2.0 License"

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "small": {
            "train": ["./small/stop_10_train.json.gz", "./small/stop_20_train.json.gz", "./small/stop_15_train.json.gz",],
            "test": ["./small/stop_10_test.json.gz", "./small/stop_20_test.json.gz", "./small/stop_15_test.json.gz",]
              },
    "small15": {
            "train": [ "./small/stop_15_train.json.gz",],
            "test": [ "./small/stop_15_test.json.gz",]
              },
    "small10": {
            "train": ["./small/stop_10_train.json.gz"],
            "test": ["./small/stop_10_test.json.gz"]
              },
    "small20": {
            "train": [ "./small/stop_20_train.json.gz"],
            "test": [ "./small/stop_20_test.json.gz"]
              }
}


class SynStOpDatasetConfig(datasets.BuilderConfig):

    def __init__(self, subset="small", length=(10,15,20), **kwargs):
        """BuilderConfig for SynStOpDatasetConfig.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SynStOpDatasetConfig, self).__init__(**kwargs)
        self.subset = subset
        self.length = length
        self.files = {
            "train": ["./{subset}".format(subset=subset) + "/stop_{length}_train.json.gz".format(length=length) for length in length],
            "test": ["./{subset}".format(subset=subset) + "/stop_{length}_test.json.gz".format(length=length) for length in length],
        }



# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class SynStOpDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("0.0.1")

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'small')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [SynStOpDatasetConfig(name="small", length=(10,15,20),version=VERSION, description="Small set of string operations  with string slices only")] +\
                      [SynStOpDatasetConfig(name=f"small{l1}", length=(l1,), version=datasets.Version("0.0.1"), description="Small set of string operations  with string slices only") for l1 in [10,15, 20]]

    DEFAULT_CONFIG_NAME = "small"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        features = datasets.Features(
                {
                    "input": datasets.Value("string"),
                    "output": datasets.Value("string"),
                    "code": datasets.Value("string"),
                    "res_var": datasets.Value("string"),
                    "operation": datasets.Value("string"),
                    "id": datasets.Value("int32"),
                    # These are the features of your dataset like images, labels ...
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )
    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        urls = self.config.files
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir["train"],
                    "split": "train",
                },
            ),

            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath":  data_dir["test"],
                    "split": "test",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        count = 0
        for filename in filepath:
            with open(filename, encoding="utf-8") as f:
                dataset = json.load(f)
            for ix, data in enumerate(dataset):

                if self.config.name.startswith("small"):
                    # Yields examples as (key, example) tuples
                    id = data["id"] if "id" in data else count
                    count = count + 1
                    yield id,  {
                        "input": data["input"],
                        "output": data["output"],
                        "code": data["code"],
                        "res_var": data["res_var"],
                        "id": id,
                        "operation": data["operation"]
                    }
                else:
                    yield "", {
                        "sentence": data["sentence"],
                        "option2": data["option2"],
                        "second_domain_answer": "" if split == "test" else data["second_domain_answer"],
                    }
