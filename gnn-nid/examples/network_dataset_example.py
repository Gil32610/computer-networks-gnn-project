from network.dataset import UNSWNB15Dataset

if __name__ == '__main__':
    root = '../data/train'
    dataset = UNSWNB15Dataset(root=root, num_neighbors=5)
    print(dataset)