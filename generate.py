"""
This module generates datasets doing string manipulations
"""
import enum
import random
from collections import deque
import os
import string
import json


class StringOperations(str, enum.Enum):

    SLICE = "slicing"
    STARTS_ENDS_WITH = "starts_ends_with"
    LEN= "len"
    CONCAT= "concat"
    REPEAT= "repeat"
    UPPER_LOWER_SWAP_CASE= "upper_lower_swap_case"
    IS= "is"


def generate_random_string(length, charset = None):
    """
    Generates a random string of length length
    :param length: the length of the string
    :param alphabet: the alphabet to use
    :return: the string
    """
    if charset is None:
        charset = string.ascii_letters + string.digits
    return ''.join(charset[b % len(charset)] for b in os.urandom(length))

def generate_reverse_string_prompt(samples,  length=20, rev_op="'{sample}'[::-1]",result_var="res", **kwargs):
    """
    provides samples examples of reversing a random string.
    :param s: the string
    :return: the reversed string
    """
    for i in range(samples):
        s = generate_random_string(length)
        o = rev_op.format(sample=s)
        yield o, s[::-1], f"{result_var}='{o}'[::-1]", result_var, "reverse"

def generate_slicing_examples(samples, length=20, #char_at_op="'{sample}'[{pos}]",
                                     pos_range=(1,2),
                                     slice_range = (1,1),
                                     step_size = (1,1),
                                     result_var="res", **kwargs):
    """
    provides samples examples of reversing a random string.
    :param s: the string
    :return: yields inp, outp, code, res_var, todo_str
    """
    for _ in range(samples):
        s = generate_random_string(length)
        for i in range(*pos_range):
            for j in range(*slice_range):
                if j == 1: # only one character means no slice
                    o = f"'{s}'[{i}]"
                    yield o, s[i], f"{result_var}={o}", result_var, f"char_at_{i}"
                else:
                    for k in range(*step_size):
                        if k==0:
                            continue
                        elif j==0 and i==0: # only step size
                            o = f"'{s}'[::{k}]"
                            yield o, s[::k], f"{result_var}={o}", result_var, f"step_::{k}"
                        elif k==1:
                            o = f"'{s}'[{i}:{i+j}]"
                            yield o, s[i:i+j], f"{result_var}={o}", result_var, f"slice_{i}:{i+j}"
                        elif abs(k)<j: # step size needs to be smaller
                            if k>0:
                                o = f"'{s}'[{i}:{i+j}:{k}]"
                                yield o, s[i:i+j:k], f"{result_var}={o}", result_var, f"slice_step_{i}:{i+j}:{k}"
                            else:
                                o = f"'{s}'[{i}:{i+j}][::{k}]"
                                yield o, s[i:i+j][::k], f"{result_var}={o}", result_var, f"slice_reverse_{i}:{i+j}:{k}"



class StringOperationGenerator:
    """


    """
    data=None


    def set_samples(self, equations):
        self.equations = equations
        return self

    @staticmethod
    def get_prompt(template_name:str = "simple", with_code:bool = False):
        if template_name == "simple":
            returns = StringOperationGenerator._get_simple_string_op_prompt()
        else: returns = StringOperationGenerator._get_plain_prompt()
        if with_code:
            returns[-1].extend( [{"templates": ["###ACTION: exec-python\n{code}\n###/ACTION"],
                         "keys": ["code"],
                         "component": "action",},
                        ])
        return returns

    @staticmethod
    def _get_plain_prompt():
        """
        :return: template for generating value filled equation, e.g. 1+2=3 and mapping needed for the dataset
        """
        inp, out = [], []
        inp.extend([{"templates": ["{input}="],
                      "keys": ["input"],
                      "component": "input",
                     },
                    ])
        out.extend([{"templates": ["{output}\n"],
                     "keys": ["output"],
                     "component": "output",
                     "tags": ["exact"]}])

        return [inp, out]

    @staticmethod
    def _get_simple_string_op_prompt():
        inp, out = [], []
        # todo: this is a problem here, since the prompt is context sensitive, i.e. it depends on the data.
        inp.extend([{"templates": ["Conduct the string operation {operation} as follows: {input}.\n"],
                      "keys": ["operation", "input"],
                      "component": "input",
                     },

                    ])
        out.extend([{"templates": ["{res_var}={output}\n"],
                     "keys": ["res_var", "output"],
                     "component": "output",
                     "tags": ["exact"]}])

        return [inp, out]


    def create_data (self, samples=100, operations=(StringOperations.SLICE),
                              valid_data_only=True, **kwargs):
        self.data = deque()

        for op in operations:
            if op==StringOperations.SLICE.value:
                g = generate_slicing_examples(samples, **kwargs)
            else:
                raise NotImplementedError(f"Operation {op} not implemented")
            for inp, outp, code, res_var, todo_str in g:
                if valid_data_only and (outp=="" or outp is None): continue
                self.data.append({"input": inp, "output": outp, "code": code,"res_var": res_var, "operation": todo_str })

        self.data = list(self.data)
        return self

    def save(self, filename):
        # load prompts and equations from file
        import json
        with open(filename, "w") as f:
            json.dump(self.data, f)
        return self

    def load(self, filename):
        # load data and equations from file
        import json
        with open(filename, "r") as f:
            self.data = json.load(f)
        return self


def write_data(dump_dir, file, out, compress, indent=2 ):
    import json, gzip
    filename = os.path.join(dump_dir, file)
    if compress:
        with gzip.open(f'{filename}.gz', 'wt', encoding='utf-8') as f:
            json.dump(out, f, indent=indent)
    else:
        json.dump(out, open(filename, "w"), indent=indent)
    return filename


def generate_data_for_config(dump_dir, about, s_length = (10,25, 5), pos_range = (0,5), slice_range = (0,4),
                    step_size = (-1,2), samples_per_config = 10, valid_data_only = True):

    samples = samples_per_config * (step_size[1]-step_size[0]) \
              * (slice_range[1]-slice_range[0])\
              *  (pos_range[1]-pos_range[0])

    about["data_files"] = {"train": [], "test": []}

    markdown = ["", "|Length|Set|Group|Amount|File|", "|---|---|---|---|---|" ]
    train_total, test_total, id = 0, 0, 1
    for length in tqdm.tqdm(range(*s_length), desc="Generating data"):
        generator = StringOperationGenerator()
        data = generator.create_data(samples=samples,
                                            operations=("slicing",),
                                            length=length,
                                     pos_range=pos_range,
                                     slice_range = slice_range,
                                     step_size = step_size,
                                            valid_data_only=valid_data_only,
                                     result_var="res",).data

        for e in data:
            e["id"] = id
            id=id+1
        cnt = Counter([e["operation"] for e in data])
        test, train = {}, {}
        for d in cnt.keys(): test[d], train[d] = [], []

        for ix, e in enumerate(data): # not very smart, but it is late
            if len(test[e["operation"]])>cnt[e["operation"]] *(1-split_ratio):
                train[e["operation"]].append(e)
            else:
                test[e["operation"]].append(e)

        markdown.extend([f"|{length}|train|{k}|{len(v)}|stop_{length}_train.json|" for k, v in train.items()])
        markdown.extend([f"|{length}|test|{k}|{len(v)}|stop_{length}_train.json|" for k, v in test.items()])

        about["length"]= length
        about["set"] = "train"
        data = [v for value in train.values() for v in value]
        write_data(dump_dir, f"stop_{length}_train.json", data, compress)
        about["data_files"]["train"].append({"length": length,
                                             "files": [f"stop_{length}_train.json"],
                                             "entries": len(data),
                                            "groups": [{"name": k, "amount": len(v)} for k, v in train.items()]})
        train_total+=len(data)

        data = [v for value in test.values() for v in value]
        write_data(dump_dir,  f"stop_{length}_test.json", data, compress)
        about["data_files"]["test"].append({"length": length,
                                             "files": [f"stop_{length}_test.json"],
                                             "entries": len(data),
                                            "groups": [{"name": k, "amount": len(v)} for k, v in test.items()]})
        test_total+=len(data)

    about["items"] = {"train": train_total, "test": test_total}
    # now add all about key value pairs except data_files to makrdown varialbe as separate table
    pre_md = ["# Metadata", "|Key|Value|", "|---|---|"]
    pre_md.extend([f"|{k}|{v}|" for k, v in about.items() if k!="data_files"])
    markdown = pre_md + markdown
    with open(os.path.join(dump_dir, "about.json"), "w") as f:
        json.dump(about, f, indent=2)

    with open(os.path.join(dump_dir, "Readme.md"), "w") as f:
        f.write("\n".join(markdown))

    return about, markdown



if __name__=="__main__":
    import datetime, os, tqdm
    from collections import Counter

    split_ratio = 0.7
    compress = True
    about = {
        "dataset_name" : "StOp-small",
        "hfuser":"mgrani",
        "version": "0.0.1",
        # add date today as created field with the date of now
        "created" : datetime.datetime.now().strftime("%Y-%m-%d"),
        "creator" : "Michael Granitzer, michael.granitzer@uni-passau.de",
        "split_ratio" : split_ratio,
        "prompt": {"plain": StringOperationGenerator.get_prompt(template_name="plain"),
                   "simple": StringOperationGenerator.get_prompt(template_name="simple"),
                   "simple_with_code": StringOperationGenerator.get_prompt(template_name="simple_with_code")}
    }

    # get the date for today, but nicely formatted as string

    dump_dir = os.path.expanduser("./small")
    if not os.path.exists(dump_dir): os.mkdir(dump_dir)
    about, markdown = generate_data_for_config(dump_dir, about, s_length=(10,25, 5), pos_range=(0,5),
                                               slice_range=(0,4), step_size=(-1,2), samples_per_config=10,
                                               valid_data_only=True)


   # with open(os.path.join(dump_dir, "README.md"), "w") as readme:

   #     card = StringOperationGenerator.dataset_card(train_total, test_total,
   #                                       group_stats_table="\n".join(group_stats),
   #                                       metadata= "\n".join([f"{k}={v}" for k,v in about.items()]))

   #     readme.write(card)
   #     readme.close()


