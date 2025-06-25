import os
import os.path as osp
import pandas as pd
import datasets
import argparse
from dotenv import load_dotenv
from huggingface_hub import login
from SQUIDp.util import auto_expand

def main():
    parser = argparse.ArgumentParser(description="Download subset of HEST 1k")
    parser.add_argument('--hgf_token_path', type=str, required=True, help="Path to your huggingface token file")
    parser.add_argument('--hest_data_dir', type=str, default='~/hest_data', help="Directory to save the downloaded dataset")
    parser.add_argument('--member', type=str, required=True, help="Member name for the dataset")
    args = parser.parse_args()

    load_dotenv(dotenv_path=auto_expand(args.hgf_token_path))
    api_token = os.getenv("API_TOKEN")
    login(token=api_token)

    local_dir=auto_expand(args.hest_data_dir)
    if not osp.exists(local_dir):
        os.makedirs(local_dir)
    meta_df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_1_0.csv")
    meta_df = meta_df[meta_df['species'] == 'Homo sapiens']
    meta_df = meta_df[meta_df['st_technology'] == 'Xenium']

    # get tissue subsets
    member = args.member
    if member == "akshaya":
        tissues = ["Pancreas", "Colon", "Liver", "Kidney", "Bowel"]
    elif member == "harry":
        tissues = ["Heart", "Brain", "Lung"]
    elif member == "tanvi":
        tissues = ["Breast", "Skin", "Bone marrow", "Tonsil", "Prostate", "Lymph node", "Ovary", "Femur bone"]
    else:
        raise Exception("Team member not recognized")

    meta_df = meta_df[meta_df['tissue'].isin(tissues)]

    ids_to_query = meta_df['id'].values
    list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
    dataset = datasets.load_dataset(
        'MahmoodLab/hest', 
        cache_dir=local_dir,
        patterns=list_patterns
    )



if __name__ == "__main__":
    main()