# SQUID:p (Spatial Quantification of Uniquely-Inferred Disease Pathways)
[BroadHacks](https://sites.google.com/broadinstitute.org/coderats/broadhacks/broadhacks-2025) Project, Zero-shot evaluation of Pathology FM at expression prediction task

# Install
Set up with the following commands:
``` 
cd SQUIDp
conda env create -f environment.yml
conda activate squidp
pip install --upgrade pip
pip install -e .
```

# Data Download
We use the [HEST-1k](https://github.com/mahmoodlab/HEST) ST library for the eval. Follow the below steps to download.
1. Obtain access to the hugging face repo [here](https://huggingface.co/datasets/MahmoodLab/hest).
2. Create and copy your token [here](https://huggingface.co/settings/tokens).
3. Create a `.token.env` file *parallel* to this repo, with a singular line that reads `API_TOKEN=YOURTOKEN`.
4. Run the below command, member={akshaya, harry, tanvi}, note the data should be between 100GB~200GB per person, should take at most 30min
```
cd SQUIDp
python data/hest1k_download.py \
    --hgf_token_path PATH_TO_HGF_TOKEN_FILE \
    --hest_data_dir PATH_TO_YOUR_DESIRED_DIRECTORY
```

# Data Processing
We extract the patches that has at least one cells, and pair each patch with the average expression vector of the cells found in that patch. Simply run the script below. Patch size is default 1024.
```
python data/patch_process.py \
    --hest_data_dir PATH_TO_YOUR_DESIRED_DIRECTORY \
    --output_dir PATH_TO_YOUR_DESIRED_DIRECTORY \
    --patch_size 1024
```
