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
Minimal dataset for intended for LM development and testing using python string operations. The dataset is created by running different one line python string operations on random strings The idea is, that transformer implementation can learn the string operations and that this task is a good proxy tasks for other transformer operations on real languages and real tasks. Consequently, the data set is small and can be used in the development process without large scale infrastructures.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "Apache 2.0 License"

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "small": ["https://github.com/mgrani/StOp-Dataset/raw/main/small/stop_10_train.json.gz",
              "https://github.com/mgrani/StOp-Dataset/raw/main/small/stop_10_test.json.gz",
              "https://github.com/mgrani/StOp-Dataset/raw/main/small/stop_20_train.json.gz",
              "https://github.com/mgrani/StOp-Dataset/raw/main/small/stop_20_test.json.gz",
              "https://github.com/mgrani/StOp-Dataset/raw/main/small/stop_15_train.json.gz",
              "https://github.com/mgrani/StOp-Dataset/raw/main/small/stop_15_test.json.gz",
              "https://github.com/mgrani/StOp-Dataset/raw/main/small/about.json",
              "https://github.com/mgrani/StOp-Dataset/raw/main/small/Readme.md"]
}


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class StopDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("0.0.1")

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'small')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="small", version=VERSION, description="Small string operations dataset with string slices only"),
        datasets.BuilderConfig(name="small[filter]", version=VERSION, description="Small string operations dataset with string slices only. [] allows to specify a comma separated list of filters on the length (i.e. l=X) and operations (i.e. o=y)"),

    ]

    DEFAULT_CONFIG_NAME = "small"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if self.config.name.startswith("small"):  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "input": datasets.Value("string"),
                    "output": datasets.Value("string"),
                    "code": datasets.Value("string"),
                    "res_var": datasets.Value("string"),
                    "operation": datasets.Value("string")
                    # These are the features of your dataset like images, labels ...
                }
            )
            self._init_filters(self.config.name[len("small"):].strip("[]").split(","))
            self.config.name= self.config.name[:len("small")]
        else:
            raise NotImplementedError()
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

    def _init_filters(self, filters):
        self.filter_operations = []
        self.filter_len = []
        for filter in filters:
            if filter =="": continue
            k, v = filter.split("=")
            if k=="l":
                self.filter_len.append(int(v))
            elif k=="o":
                self.filter_operations.append(re.compile(v))

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        urls = _URLS[self.config.name]
        if len(self.filter_len)>0:
            urls = [url for url in urls if any([f"stop_{str(len)}_t" in url for len in self.filter_len])]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": [data_dir[i] for i, n in enumerate(urls) if "_train.json" in n],
                    "split": "train",
                },
            ),

            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath":  [data_dir[i] for i, n in enumerate(urls) if "_test.json" in n],
                    "split": "test",
                },
            ),
        ]

    def _match_operations_filter(self, operation):
        if self.filter_operations is not None:
           matches = False
           for filter in self.filter_operations:
                if filter.matches(operation):
                    matches = True
                    break
           return matches
        else: return True

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

                    if self._match_operations_filter(data["operation"]):
                        continue

                    # Yields examples as (key, example) tuples
                    id = data["id"] if "id" in data else count
                    count = count + 1
                    yield id,  {
                        "input": data["input"],
                        "output": data["output"],
                        "code": data["code"],
                        "res_var": data["res_var"],
                        "operation": data["operation"]
                    }
                else:
                    yield "", {
                        "sentence": data["sentence"],
                        "option2": data["option2"],
                        "second_domain_answer": "" if split == "test" else data["second_domain_answer"],
                    }
