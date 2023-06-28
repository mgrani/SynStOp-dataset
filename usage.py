import datasets

if __name__=="__main__":
    # load locally from this repo
    ds = datasets.load_dataset("./stop.py", "small")

    ds.push_to_hub("PaDaS-Lab/stop-small")

    from datasets import load_dataset

    dataset = load_dataset("PaDaS-Lab/stop-small")
    print(dataset)

    # load locally from this repo
    # load locally from this repo

