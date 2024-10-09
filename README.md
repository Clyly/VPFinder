# VPFinder

## 1 Project Description

Datasets and codes for submitted papers.
## 2 Environments

1. OS: Ubuntu

   GPU: NVIDIA A100-SXM
   
2. Language: Python (v3.8)
3. CUDA: 11.3
4. Python packages:
   - [Allennlp 2.4.0](https://allennlp.org/)
   - [Pytorch 2.1.0](https://pytorch.org/)
   - [Transformers 4.1.0](https://huggingface.co/)
   
   Please refer the official docs for the use of these packages (especially **AllenNLP**).
5. Setup:

   We use the approach proposed by Pan *et, al.* [(Automated Unearthing of Dangerous Issue Reports, FSE 2022)](https://doi.org/10.1145/3540250.3549156), Zhou *et, al.* [(Finding a needle in a haystack: Automated mining of silent vulnerability fixes, ASE 2021)](https://ieeexplore.ieee.org/abstract/document/9678720) and Sun *et, al.* [(Silent Vulnerable Dependency Alert Prediction with Vulnerability Key Aspect Explanation, arXiv)](https://arxiv.org/abs/2302.07445) as our baselines.
   Pan *et, al.*'s work is archived at [link](https://github.com/panshengyi/MemVul).
   
   We use [bert-base-uncased](https://huggingface.co/bert-base-uncased) and [mrm8488/codebert-base-finetuned-detect-insecure-code](https://huggingface.co/mrm8488/codebert-base-finetuned-detect-insecure-code#details-of-the-downstream-task-code-classification---dataset-%F0%9F%93%9A) from HuggingFace Transformer Libarary.
   
## 3 Contents of the Folder

Let's explain some of the files that need to be in folders.

### 3.1 data

The `data` folder needs the following files:
   - 1000.csv
   - all_samples.csv
   - CVE_dict.json
   - cwe_1_8_classes.json
   - cwe_1_8_classes_old.json
   - cwe_2_4_classes.json
   - cwe_2_4_clasees_old.json
   - dataset.csv
   - dataset_project_relation_layer.csv
   - embedded_cwe.json
   - embedded_from_bottom_cwe.json
   - test_project.csv
   - test_project.json
   - train_project.csv
   - train_project.json

Files `all_samples.csv`, `CVE_dict.json`, `dataset.csv` and `dataset_project_relation_layer.csv` are shared anonymously at [here](https://zenodo.org/records/10387529?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjY2ZjAyNTY5LTczODYtNGJiOS1hYWJlLTcyZTE3ZjNiOTc2YiIsImRhdGEiOnt9LCJyYW5kb20iOiI5NjVkYjNlNWJmMzM0YTYxNzdiMzNhNzczMjg3NGNiZiJ9.SJtsIHOHPrUsf8DfTHCrCMWt55dOmLSvXbwFXHGhH47yo8IStQe6hva8YflFHpVzci8EIDmFNFrzVtZSLWB_kQ).

### 3.2 model_best

The `model_best` folder needs the following files:
   - VPFinder_multi_f1.txt
   - VPFinder_multi_model.pth
   - VPFinder-1+3n+4_multi_f1.txt
   - VPFinder-1+3n+4_multi_model.pth
   - VPFinder-h2+h3_multi_f1.txt
   - VPFinder-h2+h3_multi_model.pth

All files are shared anonymously at [here](https://zenodo.org/records/10393599?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjdkNDQzNDMxLWNiMjktNGZkMi05NTVlLWU1ZmIxNzEzMTk5MSIsImRhdGEiOnt9LCJyYW5kb20iOiJmMGRiNjdjZWU5ZGRhZTM1NGVmMzY3NjFiMTVkOTVmZCJ9.bKQueNbcMcwJI0hU_qXyW_p_oKNdR_J0MdAC7zDjN3DLn5pS1ph9GjqHJRryI9Lrb8i9c4n4mtw0ZhorxYRzOw).

### 3.3 mybert

Download the Huggingface model files `bert-base-uncased` and save them to `mybert` folder. 

Click [here](https://huggingface.co/bert-base-uncased/tree/main) to jump to the model file page.

Or you can directly modify the bert model name in the training and testing python file to download files online.

### 3.4 mycodebert

Download the Huggingface model files `mrm8488/codebert-base-finetuned-detect-insecure-code` and save them to `mycodebert` folder. 

Click [here](https://huggingface.co/mrm8488/codebert-base-finetuned-detect-insecure-code/tree/main) to jump to the model file page.

Or you can directly modify the bert model name in the training and testing python file to download files online.

### 3.5 build

The `build` folder needs the following file:
   - clang_ast

The file is shared anonymously at [here](https://zenodo.org/records/10388783?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImE1ZTE1YTEzLWY5MzMtNGQxOC1hZTgyLWY5YTY2NTMxYmIxZCIsImRhdGEiOnt9LCJyYW5kb20iOiI5OWQ4M2IzY2UxY2M4Y2NlM2Q5MzgyNzUwYjRlYzk0YiJ9.yIWIV4kx2Zm7Dl9Nj74rXLjVqFBhO4VhhHJuKh23JQksacPmU1KdpNx9TQAhupqo9GUdBR_si6w2OuzYqkS4bg).


## 4 Dataset

VPFinder uses the dataset `data/dataset_project_relation_layer.csv`, which is also used by Zhou *et, al.*'work and Sun *et, al.*'s work.
Pan *et, al.*'s work uses the dataset `data/train_project.json`, `data/validation.json` and `data/test_project.json`.

If you want to create a new dataset from scratch, the initial dataset is `dataset.csv`.
First, execute `python utils.py` to obtain the dataset applicable for MemVul.
Then, execute `python make_dataset.py` to obtain datasets for the remaining models.

## 5 Train & Test
Run the files starting with `for_train`, `train`, or `test`. For example:
```
python train_VPFinder_binary.py
```
For running baseline [MemVul](https://doi.org/10.1145/3540250.3549156), limited by the size of uploaded files, we are unable to provide relevant files, see more details [here](https://github.com/panshengyi/MemVul).

Because three models have slow convergence rates, we use alternative models for training.
The parameters obtained from the training are then used for testing original models.
The model parameters have been saved in the `model_best` folder, and the code files for alternative models begin with `for_train`.
```
python for_train_VPFinder_multi.py
```
```
python test_VPFinder_multi.py
```


## 6 Production of the Dataset
We use the [Github Rest API](https://docs.github.com/en/rest) to get data.
Please make sure you have a token and then change the fields `Your Token Here` in the python files.

Get the url of each issue from the original dataset `data/all_samples.csv`.
```
python get_url.py
```
Look for the commits by given urls.
```
python get_SHA.py
```
Download patched, and find parent commits and download files.
```
python get_and_download_parent_code.py
```
Extract snippets of java and python code and prepare for the extraction of C language snippets.(Modify the path in `tackle_C.py` as needed)
```
python slice.py
```
Move the `slice.sh` file to the `build` folder and execute for extract snippets of c language:
```
./slice.sh
```
Store the extracted code snippets in uniformly named files.
```
python merge_code.py
```
Add the code snippet to the original dataset `data/all_samples.csv`. The results are saved in `data/all_samples_processed.csv`.
```
python preprocess_dataset.py
```
Filter out samples with no code.
```
python filter_dataset.py
```
Get discussions of issue reports.
```
python get_comments.py
```
Add discussion to dataset.
```
python preprocess_dataset_with_comments.py
```
Get commit mesaages and patches.
```
python get_dataset_commits.py
```
Classify the patches by deletion and addition.
```
python commit_to_patch.py
```
Add commit messages and patches to dataset.
```
python preprocess_dataset_patch_and_message.py
```
We have complete dataset `data/dataset.csv`.

## 7 CWE Embedding
We have our cwe information ready, here is our work flow.

Concatenate CWE information and built CWE tree based on the relationships between nodes.
```
python CWE_relationship.py
```
Embed CWE information and then bottom-up aggregate the vectors.
```
python emb_tree.py
```
Find the nodes according to the requirements, and the order corresponds to the labels in the dataset `data/dataset_project_relation_layer.csv`, which serves as another part of the model input.
```
python multi_class_embedding.py
```
