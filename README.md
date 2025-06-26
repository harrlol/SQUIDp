# SQUID-p
BroadHacks Project, Zero-shot evaluation of Pathology FM at expression prediction task

# Install
Set up with the following commands:
```
cd SQUIDp
conda env create -f environment.yml
pip install --upgrade pip
pip install -e .
```

# Data Download
We use the [HEST-1k](https://github.com/mahmoodlab/HEST) ST library for the eval. Follow the below steps to download.
1. Obtain access to the hugging face repo [here](https://huggingface.co/datasets/MahmoodLab/hest).
2. Create and copy your token [here](https://huggingface.co/settings/tokens).
3. Create a `.token.env` file *parallel* to this repo, with a singular line that reads `API_TOKEN=YOURTOKEN`.
4. Run the below command, member={akshaya, harry, tanvi}
```
cd SQUIDp
python data/hest1k_download.py \
    --hgf_token_path PATH_TO_HGF_TOKEN_FILE \
    --hest_data_dir PATH_TO_HEST \
    --member harry
```