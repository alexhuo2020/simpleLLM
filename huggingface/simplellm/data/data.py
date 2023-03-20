sentences = [
    "I like tea .",
    "I like coffee .",
    "You like tea .",
    "You do not like coffee ."
]
data = {"text":sentences}

from datasets import Dataset

dataset = Dataset.from_dict(data)

dataset.push_to_hub("alex2020/SimpleDataset")

