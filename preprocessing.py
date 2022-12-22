# build train and validation dataloaders from Amazon Reviews (2018) dataset for binary classification
# devide the dataset to different product categories and save the dataset for the following categories:
# books, DVDs, electronics and kitchen appliances
import os
import pandas as pd
from sklearn.model_selection import train_test_split

automotive_dataset = pd.read_json("data/Automotive_5.json.gz", lines=True)[["overall", "reviewText"]].dropna()
automotive_dataset = automotive_dataset[automotive_dataset["overall"] != 3]
automotive_dataset["label"] = (automotive_dataset["overall"] > 3).astype(int)
electronics_dataset = pd.read_json("data/Electronics_5.json.gz", lines=True)[["overall", "reviewText"]].dropna()
electronics_dataset = electronics_dataset[electronics_dataset["overall"] != 3]
electronics_dataset["label"] = (electronics_dataset["overall"] > 3).astype(int)
petSupplies_dataset = pd.read_json("data/Pet_Supplies_5.json.gz", lines=True)[["overall", "reviewText"]].dropna()
petSupplies_dataset = petSupplies_dataset[petSupplies_dataset["overall"] != 3]
petSupplies_dataset["label"] = (petSupplies_dataset["overall"] > 3).astype(int)

min_len = min(len(automotive_dataset), len(electronics_dataset), len(petSupplies_dataset))
automotive_dataset = automotive_dataset.sample(min_len)
electronics_dataset = electronics_dataset.sample(min_len)
petSupplies_dataset = petSupplies_dataset.sample(min_len)

automotive_dataset_train, automotive_dataset_test = train_test_split(automotive_dataset, test_size=0.2)
electronics_dataset_train, electronics_dataset_test = train_test_split(electronics_dataset, test_size=0.2)
petSupplies_dataset_train, petSupplies_dataset_test = train_test_split(petSupplies_dataset, test_size=0.2)

# save the balanced datasets
if not os.path.exists('data/automotive'):
    os.makedirs('data/automotive')
if not os.path.exists('data/electronics'):
    os.makedirs('data/electronics')
if not os.path.exists('data/petSupplies'):
    os.makedirs('data/petSupplies')

automotive_dataset_train.to_csv('data/automotive/train.csv')
electronics_dataset_train.to_csv('data/electronics/train.csv')
petSupplies_dataset_train.to_csv('data/petSupplies/train.csv')

automotive_dataset_test.to_csv('data/automotive/test.csv')
electronics_dataset_test.to_csv('data/electronics/test.csv')
petSupplies_dataset_test.to_csv('data/petSupplies/test.csv')
