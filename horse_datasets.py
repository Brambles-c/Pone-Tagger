from torch.utils.data import Dataset
from transformers import AutoImageProcessor
from PIL import Image
from pandas import DataFrame, Series
from utils import multi_hot
import numpy as np
from config import images_path
from torchvision.transforms import v2


class Horses(Dataset):
    def __init__(self, img_df: DataFrame, tag_names: tuple):
        self.df = img_df.copy()
        self.df['tags'] = self.df['tags'].str.split('|').map(lambda tags: multi_hot(tags, tag_names))
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.normalize = True

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        image = Image.open(images_path / f'{row['id']}.{row['image_format']}')

        return self.processor(image, do_normalize=self.normalize)['pixel_values'][0], row['tags']


class HorseEmbeddings(Dataset):
    def __init__(self, tag_col: Series, tag_names: tuple):
        self.tags = tag_col.map(lambda tags: multi_hot(tags, tag_names))
        self.embeddings = np.load('data/base_embeddings.npy')

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, index):
        return self.embeddings[index], self.tags[index]
