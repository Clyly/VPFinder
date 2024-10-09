import pandas as pd

dataset = pd.read_csv("data/all_samples_processed.csv", low_memory=False)

number = 0
temp = -1
cve = 0
ready_to_drop = []
for index, row in dataset.iterrows():
    temp += 1
    if row['code'] != 'code':
        number += 1
        try:
            if row['CVE_ID'].startswith('CVE'):
                print(row['Issue_Url'])
                cve += 1
        except:
            continue
    else:
        ready_to_drop.append(temp)

dataset.drop(dataset.index[ready_to_drop], inplace=True)
print(number)
print(cve)
print(len(dataset) - number)
print(len(dataset))

dataset = dataset.dropna(subset=["code"])

number = 0
temp = -1
cve = 0
ready_to_drop = []
for index, row in dataset.iterrows():
    temp += 1
    if row['code'] != 'code':
        number += 1
        try:
            if row['CVE_ID'].startswith('CVE'):
                print(row['Issue_Url'])
                cve += 1
        except:
            continue

print(number)
print(cve)
print(len(dataset) - number)
print(len(dataset))

dataset.to_csv("work/dataset_tc.csv", index=False)