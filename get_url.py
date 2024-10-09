import pandas as pd

def get_url():
    dataset = pd.read_csv('data/all_samples.csv')
    repo_url = dataset.loc[:, "Issue_Url"]
    repo_url = repo_url.tolist()
    with open('data/url.txt', 'w') as f:
        for i in range(len(repo_url)):
            f.write(repo_url[i])
            f.write('\n')

if __name__ == '__main__':
    get_url()