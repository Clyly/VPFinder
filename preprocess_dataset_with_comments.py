import pandas as pd
from MemVul.util import replace_tokens_simple_1

def preprocess_dataset_csv():
    # preprocess the dataset
    df = pd.read_csv("data/dataset_tdc.csv", low_memory=False)

    print("begin process...")
    df["Comment_bodys"] = df["Comment_bodys"].map(replace_tokens_simple_1)
    print("finish process...")
    df.to_csv("data/dataset_description.csv")

if __name__ == '__main__':
    preprocess_dataset_csv()
