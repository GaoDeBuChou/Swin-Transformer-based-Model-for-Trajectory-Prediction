from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from preprocessing import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MatrixDataset(Dataset):
    def __init__(self, matrix, output):
        self.matrix = matrix
        self.output = output

    def __len__(self):
        return len(self.matrix)

    def __getitem__(self, idx):
        return self.matrix[idx], self.output[idx]


class MLP(nn.Module):
    def __init__(self, i_size=256 * 256, h_size=100, h_layers=2, o_size=2):
        super(MLP, self).__init__()
        self.hiddens = nn.Sequential()
        for i in range(h_layers):
            if i == 0:
                hidden = nn.Sequential(nn.Linear(i_size, h_size), nn.Dropout(), nn.ReLU())
            else:
                hidden = nn.Sequential(nn.Linear(h_size, h_size), nn.Dropout(), nn.ReLU())
            self.hiddens.append(hidden)
        self.out = nn.Linear(h_size, o_size)

    def forward(self, x):
        x = x.flatten(1)
        for hidden in self.hiddens:
            x = hidden(x)
        return torch.sigmoid(self.out(x))


def train_model(model, train_loader, valid_loader, device=DEVICE):
    learning_rate = 0.001
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()

    train_losses, valid_losses = [], []
    for epoch in range(10):
        model.train()
        train_loss = []
        for mat, out in tqdm(train_loader):
            mat, out = mat.to(device), out.to(device)
            optimizer.zero_grad()
            pred = model(mat)
            loss = criterion(pred, out)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_losses.append(np.mean(train_loss))
        print("Train loss: {}".format(train_losses[-1]))
        model.eval()
        valid_loss = []
        for mat, out in valid_loader:
            mat, out = mat.to(device), out.to(device)
            pred = model(mat)
            loss = criterion(pred, out)
            valid_loss.append(loss.item())
        valid_losses.append(np.mean(valid_loss))
        print("Validation loss: {}".format(valid_losses[-1]))


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    # Filter missing data and useless columns
    train = train[train["MISSING_DATA"] == False]
    train = train[train["POLYLINE"].map(len) > 1]
    train = train[train["POLYLINE"] != "[]"]
    train = train[["POLYLINE"]]
    # Choose 10000 rows randomly from dataset to run
    train_1 = train.sample(10000, random_state=2023)
    transformed = transform(train_1, 256)

    matrix_tensor = transformed["MATRIX"].apply(matrix2tensor).values
    output_tensor = transformed["POLYLINE_DEST"].apply(output2tensor).values
    mat_train, mat_valid, out_train, out_valid = \
        train_test_split(matrix_tensor, output_tensor, test_size=0.2, random_state=2023)

    train_data = MatrixDataset(mat_train, out_train)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_data = MatrixDataset(mat_valid, out_valid)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)
    mlp = MLP()
    train_model(mlp, train_loader, valid_loader)
