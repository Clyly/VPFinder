import pandas as pd
from MemVul.util import replace_tokens_simple_1

def preprocess_dataset_csv():
    dataset = pd.read_csv("data/dataset_description.csv")

    i = 0
    all_add_patch = []
    all_del_patch = []
    message = []

    for index, row in dataset.iterrows():
        print(i)
        i += 1
        url = row['Issue_Url']
        org, repo_name, issue_number = url.split("/")[-4], url.split("/")[-3], url.split("/")[-1]
        issue_number = issue_number.rstrip('\n')
        print(url)

        with open(f"patch/{org}/{repo_name}/{issue_number}/add.txt", 'r') as f:
            content = f.read()
            all_add_patch.append(str(content))
        with open(f"patch/{org}/{repo_name}/{issue_number}/del.txt", 'r') as f:
            content = f.read()
            all_del_patch.append(str(content))
        with open(f"message/{org}/{repo_name}/{issue_number}/message.txt", 'r') as f:
            content = f.read()
            message.append(str(content))

    dataset.insert(6, "Patch_add", all_add_patch)
    dataset.insert(6, "Patch_del", all_del_patch)
    dataset.insert(6, "Message", message)
    
    print("begin process...")
    dataset["Message"] = dataset["Message"].map(replace_tokens_simple_1)
    print("finish process...")
    dataset.to_csv("data/dataset.csv")

if __name__ == '__main__':
    preprocess_dataset_csv()
