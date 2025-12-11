from datasets import Dataset, DatasetDict, concatenate_datasets

# Assume you have two datasets with 'train' and 'validation' splits
ds1 = DatasetDict({
    "train": Dataset.from_dict({"text": ["This is the training data", "More training data"]}),
    "validation": Dataset.from_dict({"text": ["This is validation data", "More validation data"]})
})

ds2 = DatasetDict({
    "train": Dataset.from_dict({"text": ["This is the training data 2", "More training data 2"]}),
    "validation": Dataset.from_dict({"text": ["This is validation data 2", "More validation data 2"]})
})

ds = concatenate_datasets([ds1["train"][0], ds2["train"][1]])
print(ds[0], ds[1], len(ds))
