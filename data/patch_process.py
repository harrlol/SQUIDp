## goal: modify such that it patches 224x224 pixel, and obtain average expression for that patch
import torch
import SQUIDp.util as sqd
import numpy as np
import os
import os.path as osp
import spatialdata as sd
from skimage.measure import regionprops
from PIL import Image
from torch.utils.data import Dataset
import random
from hest import iter_hest
import matplotlib.pyplot as plt
import datetime
from SQUIDp.util import auto_expand
import pickle
import warnings
import argparse
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning, module="zarr.creation")

# edited to match hest format 6/26
# given a patch range, get the cell ids
def get_cell_ids_in_patch(sdata, patch_size=224, log_file=None):
    '''
    Input:
    sdata: spatialdata object
    patch_range: a tuple of (x_start, x_end, y_start, y_end)

    Output:
    patch_id_to_cell_id: a dict that maps each patch id (y_patch_idx, x_patch_idx) to a list of cell ids
    '''
    
    patch_id_to_cell_id = dict()
    # initialize a dict to hold cell ids by patch
    # {(y_patch_idx, x_patch_idx): [cell_ids]}
    for cid, point in sdata['locations']['geometry'].items():

        # get id and coord of cell
        x_center, y_center = point.x, point.y

        # find y, x index of the bucket this cell should be in
        y_patch_idx = y_center // patch_size
        x_patch_idx = x_center // patch_size

        # if the patch is not in the dict, create it
        patch_key = (y_patch_idx, x_patch_idx)
        if patch_key not in patch_id_to_cell_id:
            patch_id_to_cell_id[patch_key] = []

        # append the cell
        patch_id_to_cell_id[patch_key].append(cid)
    
    cell_counts = [len(cell_ids) for cell_ids in patch_id_to_cell_id.values()]

    # collect stats
    if log_file is not None:
        with open(log_file, 'a') as f:
            # f.write(f"HE Nucleus Mask of dimension: {he_nuc_mask.shape}\n")
            f.write(f"Number of cells in this slide: {len(sdata['locations']['geometry'])}\n")
            f.write(f"Number of patches: {len(patch_id_to_cell_id)}\n")
            f.write(f"Patch size: {patch_size}\n")
            if cell_counts:
                f.write(f"Minimum number of cells in a patch: {min(cell_counts)}\n")
                f.write(f"Maximum number of cells in a patch: {max(cell_counts)}\n")
                f.write(f"Average number of cells in a patch: {np.mean(cell_counts):.3f}\n")
            else:
                f.write("No cells found in any patches.\n")

    return patch_id_to_cell_id

# matches each patch id (y_patch_idx, x_patch_idx) to the actual patch
def match_patch_id_to_PIL(sdata, wsi, patch_id_to_cell_id, patch_size=224, log_file=None):
    '''
    Input:
    patch_id_to_cell_id: a dict that maps each patch id (y_patch_idx, x_patch_idx) to a list of cell ids
    sdata: spatialdata object
    patch_size: size of the patch

    Output:
    patch_id_to_pil: a dict that maps each patch id (y_patch_idx, x_patch_idx) to the actual patch
    '''
    # initialize a dict to pil image by patch id
    # {(y_patch_idx, x_patch_idx): PIL image}
    patch_id_to_pil = dict()
    for patch_key in patch_id_to_cell_id.keys():

        y_patch_idx, x_patch_idx = patch_key

        # define the location of the patch for read_region
        x_loc = int(x_patch_idx * patch_size)
        y_loc = int(y_patch_idx * patch_size)
        
        # check if the patch is within the bounds of the image
        if x_loc < 0 or (x_loc + patch_size) > wsi.width or y_loc < 0 or (y_loc + patch_size) > wsi.height:
            if log_file is not None:
                with open(log_file, 'a') as f:
                    f.write(f"Patch ({x_loc}, {y_loc}) with dimension {patch_size} is out of bounds. Skipped. \n")
            continue
        # print(f"Processing patch {patch_key} at location ({x_loc}, {y_loc}) with size {patch_size}")
        
        # obtain the patch and store in dict
        patch_np = wsi.read_region(location=(x_loc, y_loc), level=0, size=(patch_size, patch_size))
        patch_pil = Image.fromarray(patch_np.astype(np.uint8))
        patch_id_to_pil[patch_key] = patch_pil

    # collect stats
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(f"Number of patches with valid images: {len(patch_id_to_pil)}\n")
    
    return patch_id_to_pil

# matches each patch id (y_patch_idx, x_patch_idx) to the average expression of cells in that patch
def match_patch_id_to_expr(sdata, patch_id_to_cell_id, patch_id_to_pil, log_file=None):
    '''
    Input:
    sdata: spatialdata object
    patch_id_to_cell_id: a dict that maps each patch id (y_patch_idx, x_patch_idx) to a list of cell ids

    Output:
    patch_id_to_expression: a dict that maps each patch id (y_patch_idx, x_patch_idx) to the average expression
    of cells in that patch, stored as (460,)
    '''

    # get the expression data
    expr_data = sdata['table']

    patch_id_to_expr = dict()
    patch_id_to_delete_from_cid = []
    patch_id_to_delete_from_pil = []
    for patch_key, cell_ids in patch_id_to_cell_id.items():

        # get a subset of expression data of shape (n_cells_in_this_patch, 460)
        subset = expr_data[expr_data.obs['instance_id'].isin(cell_ids)]

        # if subset is empty, this patch contains no cells that we can use for train
        # add this patch id to the list of patches to delete
        if subset.shape[0] == 0:
            patch_id_to_delete_from_cid.append(patch_key)
            patch_id_to_delete_from_pil.append(patch_key)
            continue

        # also, cases where boundary patches got omitted for the PIL step
        if patch_key not in patch_id_to_pil:
            patch_id_to_delete_from_cid.append(patch_key)
            continue

        # get average expression vector
        avg_expr = subset.X.toarray().mean(axis=0)

        # store in dict
        patch_id_to_expr[patch_key] = avg_expr

    # process previous dicts to remove empty patches
    for patch_key in patch_id_to_delete_from_cid:
        del patch_id_to_cell_id[patch_key]
    
    for patch_key in patch_id_to_delete_from_pil:
        del patch_id_to_pil[patch_key]
    
    # collect stats
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(f"Max average expression: {max([np.mean(expr) for expr in patch_id_to_expr.values()])}\n")
            f.write(f"Min average expression: {min([np.mean(expr) for expr in patch_id_to_expr.values()])}\n")
            f.write(f"Deleted {len(patch_id_to_delete_from_pil)} patches with no cells containing expression information\n")
            f.write(f"Deleted {len(patch_id_to_delete_from_cid)} patches that ran out of WSI boundaries\n")
            f.write(f"Number of remaining patches (which has valid expression data): {len(patch_id_to_expr)}\n")
            filter_only_1_patch_id = {key: value for key, value in patch_id_to_cell_id.items() if len(value) == 10}
            f.write(f"Number of patches with at least 10 cells: {len(filter_only_1_patch_id)}\n")
            filter_10_patch_id = {key: value for key, value in patch_id_to_cell_id.items() if len(value) >= 10}
            f.write(f"Number of patches with at least 10 cells: {len(filter_10_patch_id)}\n")
            filter_100_patch_id = {key: value for key, value in patch_id_to_cell_id.items() if len(value) >= 100}
            f.write(f"Number of patches with at least 100 cells: {len(filter_100_patch_id)}\n")

    return patch_id_to_cell_id, patch_id_to_pil, patch_id_to_expr

# makes plots and visualizations
def plots_n_visualizations(sdata, id, patch_id_to_cell_id, patch_id_to_pil, patch_id_to_expr, plot_dir):
    '''
    Input:
    sdata: spatialdata object
    patch_id_to_cell_id: a dict that maps each patch id (y_patch_idx, x_patch_idx) to a list of cell ids
    patch_id_to_pil: a dict that maps each patch id (y_patch_idx, x_patch_idx) to the actual patch
    patch_id_to_expr: a dict that maps each patch id (y_patch_idx, x_patch_idx) to the average expression

    Note these plots corresponds to the dicts that already filtered out the
    patches that contain cells with no expression information

    Output:
    None
    '''
    file_name = id

    # fig1 check distribution of cell counts
    cell_counts = [len(cell_ids) for cell_ids in patch_id_to_cell_id.values()]
    plt.hist(cell_counts, bins=50)
    plt.xlabel('Number of cells in patch')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of cell counts in patches for {file_name}')
    plt.savefig(f"{plot_dir}/cell_counts_per_patch_{file_name}.png")
    plt.close()

    # fig2 check distribution of average expression
    # plot 6 patches: max # cells, min # cells, (50,50), 3 random
    max_patch_idx = list(patch_id_to_cell_id.keys())[np.argmax([len(id_list) for id_list in patch_id_to_cell_id.values()])]    # rip readability
    max_patch_n_cells = len(patch_id_to_cell_id[max_patch_idx])
    min_patch_idx = list(patch_id_to_cell_id.keys())[np.argmin([len(id_list) for id_list in patch_id_to_cell_id.values()])]
    min_patch_n_cells = len(patch_id_to_cell_id[min_patch_idx])

    # approximate center patch
    patch_keys = list(patch_id_to_cell_id.keys())
    avg_y = int(np.round(np.mean([y for y, x in patch_keys])))
    avg_x = int(np.round(np.mean([x for y, x in patch_keys])))
    center_patch_idx = (avg_y, avg_x)
    # check if center patch is in the dict
    if center_patch_idx not in patch_id_to_cell_id:
        center_patch_idx = random.choice(patch_keys)

    center_patch_n_cells = len(patch_id_to_cell_id[center_patch_idx])
    random_patch_idx = random.sample(list(patch_id_to_cell_id.keys()), k=3)
    random_patch_n_cells = [len(patch_id_to_cell_id[idx]) for idx in random_patch_idx]

    _, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].imshow(patch_id_to_pil[max_patch_idx])
    axs[0, 0].set_title(f"Max cells: {max_patch_n_cells} at {max_patch_idx}")
    axs[0, 1].imshow(patch_id_to_pil[min_patch_idx])
    axs[0, 1].set_title(f"Min cells: {min_patch_n_cells} at {min_patch_idx}")
    axs[0, 2].imshow(patch_id_to_pil[center_patch_idx])
    axs[0, 2].set_title(f"Center patch: {center_patch_n_cells} at {center_patch_idx}")
    axs[1, 0].imshow(patch_id_to_pil[random_patch_idx[0]])
    axs[1, 0].set_title(f"Random patch 1: {random_patch_n_cells[0]} at {random_patch_idx[0]}")
    axs[1, 1].imshow(patch_id_to_pil[random_patch_idx[1]])
    axs[1, 1].set_title(f"Random patch 2: {random_patch_n_cells[1]} at {random_patch_idx[1]}")
    axs[1, 2].imshow(patch_id_to_pil[random_patch_idx[2]])
    axs[1, 2].set_title(f"Random patch 3: {random_patch_n_cells[2]} at {random_patch_idx[2]}")
    plt.suptitle(f"Sampled patches from {file_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{plot_dir}/sample_patches_viz_{file_name}.png")
    plt.close()

    # fig3 check distribution of expression and plots
    # for patch with at least 10 cells (arbitrary threshold)
    # edited to accomodate for hest data, 1 for now 6/26
    filter_10_patch_id = list({key: value for key, value in patch_id_to_cell_id.items() if len(value) >= 1}.keys())
    filtered_patch_id_to_cell_id = {k: patch_id_to_cell_id[k] for k in filter_10_patch_id}
    filtered_patch_id_to_pil = {k: patch_id_to_pil[k] for k in filter_10_patch_id}
    filtered_patch_id_to_expr = {k: patch_id_to_expr[k] for k in filter_10_patch_id}
    # patch with highest average expression across genes (patch with cells with high activity of the 460 gene pathway)
    # patch with highest spread of expression across genes (patch with high heterogeneity of the 460 gene pathway)
    # plot distribution of expression of the 460 genes for a random patch (expect right skew)
    # for all patch
    # plot the average of the expression vector across all patches (expect normal)
    file_name = id
    max_avg_patch_id = list(filtered_patch_id_to_cell_id.keys())[np.argmax([np.mean(expr) for expr in filtered_patch_id_to_expr.values()])]
    max_avg_patch = filtered_patch_id_to_expr[max_avg_patch_id]
    max_sd_patch_id = list(filtered_patch_id_to_cell_id.keys())[np.argmax([np.std(expr) for expr in filtered_patch_id_to_expr.values()])]
    max_sd_patch = filtered_patch_id_to_expr[max_sd_patch_id]
    random_patch_id = random.sample(list(filtered_patch_id_to_expr.keys()), k=1)
    random_patch = filtered_patch_id_to_expr[random_patch_id[0]]
    avg_expr_all_patches = np.mean(list(filtered_patch_id_to_expr.values()), axis=1)   # mean of (n_patch, 460) at axis=1, expect normal dist
    num_patches = len(filtered_patch_id_to_expr)
    # plots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0,0].imshow(filtered_patch_id_to_pil[max_avg_patch_id])
    axs[0,0].set_title(f"Max avg expression: {np.mean(max_avg_patch)} at {max_avg_patch_id} \n Number of cells: {len(filtered_patch_id_to_cell_id[max_avg_patch_id])}")
    axs[0,1].imshow(filtered_patch_id_to_pil[max_sd_patch_id])
    axs[0,1].set_title(f"Max sd expression: {np.std(max_sd_patch)} at {max_sd_patch_id} \n Number of cells: {len(filtered_patch_id_to_cell_id[max_sd_patch_id])}")
    axs[1,0].hist(random_patch, bins=50)
    axs[1,0].set_title(f"Random patch expression distributions: avg expression {np.mean(random_patch)} at {random_patch_id} \n Number of cells: {len(filtered_patch_id_to_cell_id[random_patch_id[0]])}")
    axs[1,0].set_xlabel('Expression value')
    axs[1,0].set_ylabel('Frequency')
    axs[1,1].hist(avg_expr_all_patches, bins=50)
    axs[1,1].set_title(f"Avg expression across all {num_patches} patches")
    axs[1,1].set_xlabel('Expression value')
    axs[1,1].set_ylabel('Frequency')
    plt.suptitle(
        f"Sample Expression Statistics and Visualization for {file_name}\n"
        "For patches with at least 10 cells",
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{plot_dir}/sample_expression_viz_{file_name}.png")
    plt.close()

    return

# save the patches and their expression data
def save_patches(sdata, id, patch_id_to_cell_id, patch_id_to_pil, patch_id_to_expr, output_dir):
    '''
    Input:
    sdata: spatialdata object
    patch_id_to_cell_id: a dict that maps each patch id (y_patch_idx, x_patch_idx) to a list of cell ids
    patch_id_to_pil: a dict that maps each patch id (y_patch_idx, x_patch_idx) to the actual patch
    patch_id_to_expr: a dict that maps each patch id (y_patch_idx, x_patch_idx) to the average expression
    output_dir: directory to save the patches

    Output:
    None
    '''
    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # save
    file_name = id
    
    data = []
    for patch_id in sorted(patch_id_to_cell_id.keys()):
        data.append({
            'patch_id': patch_id,
            'pil': patch_id_to_pil[patch_id],
            'expr': patch_id_to_expr[patch_id]
        })
    with open(osp.join(output_dir, 'patch_to_expr_' + file_name + '.pkl'), 'wb') as f:
        pickle.dump(data, f)
    
    return


# runs this script for all data at hand
def main():
    parser = argparse.ArgumentParser(description="Process downloaded HEST 1k")
    parser.add_argument('--hest_data_dir', type=str, required=True, help="Directory to downloaded dataset")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output patches and logs")
    parser.add_argument('--patch_size', type=int, default=224, help="Size of the patches to extract")

    # get args
    args = parser.parse_args()
    output_dir = auto_expand(args.output_dir)
    hest_data_dir = auto_expand(args.hest_data_dir)
    patch_size = args.patch_size

    # create output directories
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = osp.join(output_dir, "plots")
    log_file = osp.join(output_dir, f"log_{timestamp}.txt")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    with open(log_file, 'a') as f:
        f.write(f"Log file created at {timestamp}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Plot directory: {plot_dir}\n")
        f.write("\n")
    
    # establish the ids to process
    meta_df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_1_0.csv")
    tissue_list = ["Pancreas", "Colon", "Liver", "Kidney", "Bowel", 
                   "Heart", "Brain", "Breast", "Skin", "Bone marrow", "Tonsil", 
                   "Prostate", "Lymph node", "Ovary", "Femur bone"]
    id_list = sqd.get_ids(meta_df, tissue_list)

    # main loop
    for st in iter_hest(osp.expanduser(hest_data_dir), id_list=id_list, load_transcripts=True):
        id = st.meta['id']
        sdata = st.to_spatial_data()
        wsi = st.wsi
        
        patch_id_to_cell_id = get_cell_ids_in_patch(sdata, patch_size=patch_size, log_file=log_file)
        patch_id_to_pil = match_patch_id_to_PIL(sdata, wsi, patch_id_to_cell_id, patch_size=patch_size, log_file=log_file)
        patch_id_to_cell_id, patch_id_to_pil, patch_id_to_expr = match_patch_id_to_expr(sdata, patch_id_to_cell_id, patch_id_to_pil, log_file=log_file)
        plots_n_visualizations(sdata, id, patch_id_to_cell_id, patch_id_to_pil, patch_id_to_expr, plot_dir=plot_dir)
        save_patches(sdata, id, patch_id_to_cell_id, patch_id_to_pil, patch_id_to_expr, output_dir=output_dir)

        with open(log_file, 'a') as f:
            f.write(f"Finished processing {id}\n")
            f.write("\n")

if __name__ == "__main__":
    main()


########## OLD DO NOT USE ##########
class PatchCells(Dataset):
    """
    This class takes in an histology image,
    list of cell ids that is supposedly in that image
    a dictionary that maps cell ids to their centroid coordinates,
    and a patch size.
    """
    def __init__(self, image, cell_ids, cell_coords, patch_size=32, transform=None):
        self.image = image
        self.cell_ids = cell_ids
        self.cell_coords = cell_coords
        self.patch_size = patch_size
        self.transform = transform

    # do a filtering of the cell ids
        self.valid_cell_ids = [id for id in cell_ids if id in cell_coords]

    def __len__(self):
        return len(self.valid_cell_ids)

    # takes in a cell id, returns a {key: cell id, value: patch in tensor format}
    def __getitem__(self, idx):
        cell_id = self.valid_cell_ids[idx]
        x, y = self.cell_coords[cell_id]

        half_patch = self.patch_size // 2

        # taking care of edge cases as well
        x_start = max(0, x - half_patch)
        x_end = min(self.image.shape[0], x + half_patch)
        y_start = max(0, y - half_patch)
        y_end = min(self.image.shape[1], y + half_patch)

        # create a 0 patch first
        patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32)

        # populate with actual data, note the case where cell is at edge of image
        patch_from_in = self.image[y_start:y_end, x_start:x_end]
        patch[:patch_from_in.shape[0], :patch_from_in.shape[1], :] = patch_from_in

        patch = (patch * 255).astype(np.uint8)  
        patch = Image.fromarray(patch)   


        if self.transform:
            patch = self.transform(patch)

        return {
            'cell_id': torch.tensor(cell_id, dtype=torch.int64),
            'patch': patch.float()
        }

def get_patches(sdata, random_seed=209, transform=None, patch_size=32):
    # pull training cells ids into a list
    split_cell_id = sdata["cell_id-group"].obs.query("group == 'train'")["cell_id"].values

    # get mask, pull regions
    he_nuc_mask = sdata['HE_nuc_original'][0, :, :].to_numpy()
    regions = regionprops(he_nuc_mask)

    # pull centroid coordinate of each cell's regions
    # dict has key=cell id and value=centroid coordinate
    cell_coords = {}
    for props in regions:
        cid = props.label
        if cid in split_cell_id:
            y_center, x_center = int(props.centroid[0]), int(props.centroid[1])
            cell_coords[cid] = (x_center, y_center)

    # assemble the patch dataset
    he_image = np.transpose(sdata['HE_original'].to_numpy(), (1, 2, 0))

    np.random.seed(random_seed)
    shuffled = np.random.permutation(split_cell_id)
    total_len = len(split_cell_id)
    train_len = int(0.7 * total_len)
    val_len = int(0.2 * total_len)
    train_ids = shuffled[:train_len]
    val_ids = shuffled[train_len:train_len + val_len]
    test_ids = shuffled[train_len + val_len:]

    # create dataset objects
    dataset_patch_train = PatchCells(he_image, train_ids, cell_coords, patch_size=patch_size, transform=transform)
    dataset_patch_val = PatchCells(he_image, val_ids, cell_coords, patch_size=patch_size, transform=transform)
    dataset_patch_test = PatchCells(he_image, test_ids, cell_coords, patch_size=patch_size, transform=transform)

    return dataset_patch_train, dataset_patch_val, dataset_patch_test
