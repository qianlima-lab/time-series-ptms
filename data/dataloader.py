import torch
import torch.utils.data as data


# Dataset 仅用来加载5 fold中的一个fold
class UCRDataset(data.Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset
        # self.dataset = np.expand_dims(self.dataset, 1)
        if len(self.dataset.shape) == 2:
            self.dataset = torch.unsqueeze(self.dataset, 1)  # (num_size, 1, series_length)
        self.target = target

    def __getitem__(self, index):
        return self.dataset[index], self.target[index]

    def __len__(self):
        return len(self.target)


class UEADataset(data.Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset.permute(0, 2, 1)  # (num_size, num_dimensions, series_length)
        self.target = target

    def __getitem__(self, index):
        return self.dataset[index], self.target[index]

    def __len__(self):
        return len(self.target)


if __name__ == '__main__':
    pass
    '''
    train = pd.read_csv('/dev_data/zzj/hzy/datasets/UCR/Adiac/Adiac_TRAIN.tsv', sep='\t', header=None)

    train_target = train.iloc[:, 0]
    train_x = train.iloc[:, 1:]
    print(train_x.to_numpy())
    '''
