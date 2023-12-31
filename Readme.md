---
annotations_creators:
- synthetic
language_creators:
- other
language:
- python
license:
- Apache 2.0 Licences
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- extended|other
task_categories:
- token-classification
- text-generation
task_ids:
- natural-language-inference
pretty_name: String Operations
tags:
- development
- NLU
- small scale
dataset_info:
- config_name: small
  features:
  - name: input
    dtype: string
  - name: output
    dtype: string
  - name: code
    dtype: string
  - name: res_var
    dtype: string
  - name: operation
    dtype: string
  - name: id
    dtype: int32
  splits:
  - name: train
    num_bytes: 3222948
    num_examples: 33939
  - name: test
    num_bytes: 1392252
    num_examples: 14661
  download_size: 1178254
  dataset_size: 4615200
- config_name: small10
  features:
  - name: input
    dtype: string
  - name: output
    dtype: string
  - name: code
    dtype: string
  - name: res_var
    dtype: string
  - name: operation
    dtype: string
  - name: id
    dtype: int32
  splits:
  - name: train
    num_bytes: 956996
    num_examples: 11313
  - name: test
    num_bytes: 413404
    num_examples: 4887
  download_size: 312419
  dataset_size: 1370400
- config_name: small15
  features:
  - name: input
    dtype: string
  - name: output
    dtype: string
  - name: code
    dtype: string
  - name: res_var
    dtype: string
  - name: operation
    dtype: string
  - name: id
    dtype: int32
  splits:
  - name: train
    num_bytes: 1074316
    num_examples: 11313
  - name: test
    num_bytes: 464084
    num_examples: 4887
  download_size: 393420
  dataset_size: 1538400
- config_name: small20
  features:
  - name: input
    dtype: string
  - name: output
    dtype: string
  - name: code
    dtype: string
  - name: res_var
    dtype: string
  - name: operation
    dtype: string
  - name: id
    dtype: int32
  splits:
  - name: train
    num_bytes: 1191636
    num_examples: 11313
  - name: test
    num_bytes: 514764
    num_examples: 4887
  download_size: 472415
  dataset_size: 1706400
---

# Dataset Card for Small String Operations Dataset

## Dataset Description

 - **Homepage:** [PaDaS Lab](https://huggingface.co/PaDaS-Lab)
 - **Repository:** 
 - **Paper:**
 - **Leaderboard:**
 - **Point of Contact:** Michael Granitzer, michael.granitzer@uni-passau.de

### Other Metadata



### Dataset Summary

 Minimal dataset for intended for LM development and testing using python string operations.
 The dataset is created by running different one line python string operations on random strings
 The idea is, that transformer implementation can learn the string operations and that this task is a good
 proxy tasks for other transformer operations on real languages and real tasks. Consequently, the
 data set is small and can be used in the development process without large scale infrastructures.

## Dataset Structure

### Data Instances

 There are different configurations for the data set.

- `small`: contains below 50k instances of various string length and only contains slicing operations, i.e. all python operations expressable with `s[i:j:s]` (which also includes string reversal).
  - you can further choose different subsets according to either length or the kind of operation

 ### Data Fields

 all data instances can be found under the field "data".

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


 ### Data Splits

 The dataset is split into a train and test split for different string lengths

 ## Dataset Creation

 The dataset is synthetically created

 ### Licensing Information

 MIT License

 ### Citation Information

 [More Information Needed]

 ### Contributions

 [Chair of Data Science, University of Passau](https://huggingface.co/PaDaS-Lab)

 - Michael Granitzer, University of Passau

 Thanks to [@mgrani](https://github.com/mgrani) for adding this dataset.
        """
 

 ### Acknowledgements

This work is part of the OpenWebSearch.eu project and the SMAEGBot Project. The OpenWebSearch.eu Project is funded by the EU under the GA 101070014 and we thank the EU for their support. SMAEGBot is supported by funds of the Federal Ministry of Food and Agriculture (BMEL) based on a decision of
the Parliament of the Federal Republic of Germany. The Federal Office for Agriculture and Food (BLE) provides
coordinating support for artificial intelligence (AI) in agriculture as funding organisation, grant number 05119082.
