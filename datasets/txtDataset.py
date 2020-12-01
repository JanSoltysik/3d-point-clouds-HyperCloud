from os import listdir
from os.path import exists, join

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

synth_id_to_category = {
    '02958343': 'car',
}

category_to_synth_id = {v: k for k, v in synth_id_to_category.items()}
synth_id_to_number = {k: i for i, k in enumerate(synth_id_to_category.keys())}


def collate_fn(batch):
    """
    Remove elements in order to have batch made of clouds of the same size.
    """
    result = {key: [] for key in batch[0].keys()}
    max_len = [min([d[key] for d in batch], key=lambda x: len(x)).shape[0] for key in batch[0]]
    for d in batch:
        for i, key in enumerate(d):
            result[key].append(torch.from_numpy(d[key][:max_len[i]]))
    return {key: torch.stack(tensors) for key, tensors in result.items()}


class TxtDataset(Dataset):
    def __init__(self, root_dir, classes=[],
                 transform=None, split='train', config=None):
        """
        Args:
            root_dir (string): Directory with all the point clouds.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        if not config:
            raise ValueError("PhotogrammetryDataset JSON config is not set")

        self.config = config

        self._maybe_download_data()

        pc_df = self._get_names()
        if classes:
            if classes[0] not in synth_id_to_category.keys():
                classes = [category_to_synth_id[c] for c in classes]
            pc_df = pc_df[pc_df.category.isin(classes)].reset_index(drop=True)
        else:
            classes = synth_id_to_category.keys()

        self.point_clouds_names_train = pd.concat(
            [pc_df[pc_df['category'] == c][:int(0.85 * len(pc_df[pc_df['category'] == c]))].reset_index(drop=True) for c
             in classes])
        self.point_clouds_names_valid = pd.concat([pc_df[pc_df['category'] == c][
                                                   int(0.85 * len(pc_df[pc_df['category'] == c])):int(
                                                       0.9 * len(pc_df[pc_df['category'] == c]))].reset_index(drop=True)
                                                   for c in classes])
        self.point_clouds_names_test = pd.concat(
            [pc_df[pc_df['category'] == c][int(0.9 * len(pc_df[pc_df['category'] == c])):].reset_index(drop=True) for c
             in classes])

    def __len__(self):
        if self.split == 'train':
            pc_names = self.point_clouds_names_train
        elif self.split == 'valid':
            pc_names = self.point_clouds_names_valid
        elif self.split == 'test':
            pc_names = self.point_clouds_names_test
        else:
            raise ValueError('Invalid split. Should be train, valid or test.')
        return len(pc_names)

    def __getitem__(self, idx):
        if self.split == 'train':
            pc_names = self.point_clouds_names_train
        elif self.split == 'valid':
            pc_names = self.point_clouds_names_valid
        elif self.split == 'test':
            pc_names = self.point_clouds_names_test
        else:
            raise ValueError('Invalid split. Should be train, valid or test.')

        pc_category, pc_file_prefix = pc_names.iloc[idx].values

        pc_filedir = join(self.root_dir, pc_category)
        sample = self.load_object(directory=pc_filedir, prefix=pc_file_prefix)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_object(self, directory: str, prefix: str) -> dict:  # dict<str, np.ndarray>
        suffixes = ['_visible_mesh_data.txt', '_mesh_data.txt']
        parts = ['visible', 'non-visible']
        result = dict()

        for suffix, part in zip(suffixes, parts):
            filename = prefix + suffix
            path = join(directory, filename)
            df = pd.read_csv(path, sep=' ', header=None, engine='c', )

            if len(df.index) > self.config['n_points']:
                remove_n = len(df.index) - self.config['n_points']
                drop_indices = np.random.choice(df.index, remove_n, replace=False)
                df = df.drop(drop_indices)
            result[part] = df.to_numpy()

        # the result contains nodes containing visible and non-visible part and olso we are adding whole set of points
        result['cloud'] = np.concatenate((result[parts[0]], result[parts[-1]]), axis=0)
        return result

    def _get_names(self) -> pd.DataFrame:
        file_prefixes_by_category = []
        for category_id in synth_id_to_category.keys():
            file_prefixes = {f.split('_')[0] for f in listdir(join(self.root_dir, category_id))}

            for f_prefix in file_prefixes:
                if f_prefix not in ['.DS_Store']:
                    file_prefixes_by_category.append((category_id, f_prefix))

        return pd.DataFrame(file_prefixes_by_category, columns=['category', 'file_prefix'])

    def _maybe_download_data(self):
        if not exists(self.root_dir):
            raise FileNotFoundError("Dataset needs to be already downloaded.")


if __name__ == "__main__":
    import json

    from torch.utils.data import DataLoader
    with open("../settings/hyperparams.json") as f:
        config = json.load(f)

    dataset = TxtDataset(root_dir=config['data_dir'], classes=config['classes'], config=config)
    print(len(dataset))
    test = [dataset[i]['visible'] for i in range(10)]
    points_dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=1, drop_last=True,
                                   collate_fn=collate_fn)
    #for x in points_dataloader:
    #    print(x)
    X = next(iter(points_dataloader))
    print(X['visible'].shape)