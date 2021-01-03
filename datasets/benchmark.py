from os import listdir, makedirs, remove
from os.path import exists, join
from zipfile import ZipFile
import h5py
import shutil
import urllib
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.cluster import KMeans


synth_id_to_category = {
    '02691156': 'airplane',
    '02933112': 'cabinet',
    '02958343': 'car',
    '03001627': 'chair',
    '03636649': 'lamp',
    '04256520': 'sofa',
    '04379243': 'table',
    '04530566': 'vessel'
}

category_to_synth_id = {v: k for k, v in synth_id_to_category.items()}
synth_id_to_number = {k: i for i, k in enumerate(synth_id_to_category.keys())}


def cluster_data(original_shape_path: str, dst_data_path: str) -> None:
    with h5py.File(original_shape_path, 'r') as f:
        original_data = np.array(f['data'])

    kmeans = KMeans(n_clusters=2).fit(original_data)

    visible_data = original_data[kmeans.labels_ == 0]
    pocket_data = original_data[kmeans.labels_ == 1]

    file_prefix = original_shape_path.split('/')[-1].split('.')[0]

    np.savetxt(join(dst_data_path, f"{file_prefix}_full.txt"), original_data)
    np.savetxt(join(dst_data_path, f"{file_prefix}_pocket.txt"), pocket_data)
    np.savetxt(join(dst_data_path, f"{file_prefix}_visible.txt"), visible_data)


class Benchmark(Dataset):
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
        suffixes = ['_visible.txt', '_pocket.txt', '_full.txt']
        parts = ['visible', 'non-visible', 'cloud']
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

        return result

    def _get_names(self) -> pd.DataFrame:
        file_prefixes_by_category = []
        for category_id in synth_id_to_category.keys():
            file_prefixes = {f.split('_')[0] for f in listdir(join(self.root_dir, category_id))}

            for f_prefix in file_prefixes:
                if f_prefix not in ['.DS_Store']:
                    file_prefixes_by_category.append((category_id, f_prefix))

        return pd.DataFrame(file_prefixes_by_category, columns=['category', 'file_prefix'])

    def _create_required_clouds(self):
        # Merging all shapes from all of the dirs
        all_extracted_files = listdir(self.root_dir)
        created_dirs = ("train", "val")

        for created_dir in created_dirs:
            sub_dir = "gt"
            for shape in listdir(join(self.root_dir, created_dir, sub_dir)):
                src_shape_dir = join(self.root_dir, created_dir, sub_dir, shape)
                dst_shape_dir = join(self.root_dir, shape)

                makedirs(dst_shape_dir, exist_ok=True)

                # save all clouds in [visible/pocket] format in newly created directories
                for cloud in listdir(src_shape_dir):
                    cluster_data(join(src_shape_dir, cloud), dst_shape_dir)

        # remove all original files
        for extracted_file in all_extracted_files:
            shutil.rmtree(join(self.root_dir, extracted_file))

    def _maybe_download_data(self):
        if exists(self.root_dir):
            return

        print(f'Benchmark doesn\'t exist in root directory {self.root_dir}. '
              f'Downloading...')
        makedirs(self.root_dir)

        url = 'http://download.cs.stanford.edu/downloads/completion3d/dataset2019.zip'

        data = urllib.request.urlopen(url)
        filename = url.rpartition('/')[2]
        file_path = join(self.root_dir, filename)
        with open(file_path, mode='wb') as f:
            d = data.read()
            f.write(d)

        print('Extracting...')
        with ZipFile(file_path, mode='r') as zip_f:
            zip_f.extractall(self.root_dir)

        remove(file_path)
        extracted_dir = join(self.root_dir,
                             'shapenet')
        for d in listdir(extracted_dir):
            shutil.move(src=join(extracted_dir, d),
                        dst=self.root_dir)

        shutil.rmtree(extracted_dir)

        print('Transforming...')
        self._create_required_clouds()


if __name__ == "__main__":
    import json

    from torch.utils.data import DataLoader
    with open("settings/hyperparams.json") as f:
        config = json.load(f)

    dataset = Benchmark(root_dir=config['data_dir'], classes=config['classes'], config=config)
