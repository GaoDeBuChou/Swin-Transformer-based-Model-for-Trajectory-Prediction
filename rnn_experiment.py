from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from preprocessing import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceDataset(Dataset):
    def __init__(self, sequence, output):
        self.sequence = sequence
        self.output = output

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        return self.sequence[idx], self.output[idx]


class RNN(nn.Module):
    def __init__(self, i_size=3, h_size=6, n_layers=1, o_size=2, activation="relu"):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=i_size,
            hidden_size=h_size,
            num_layers=n_layers,
            nonlinearity=activation,
            batch_first=True
        )
        self.out = nn.Linear(h_size, o_size)

    def forward(self, x):
        out, h = self.rnn(x, None)  # None represents zero initial hidden state
        return torch.sigmoid(self.out(out[:, -1, :]))  # choose last time step of output


def train_model(model, train_loader, valid_loader, device=DEVICE):
    learning_rate = 0.001
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()

    train_losses, valid_losses = [], []
    for epoch in range(10):
        model.train()
        train_loss = []
        for seq, out in tqdm(train_loader):
            seq, out = seq.to(device), out.to(device)
            optimizer.zero_grad()
            pred = model(seq)
            loss = criterion(pred, out)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_losses.append(np.mean(train_loss))
        print("Train loss: {}".format(train_losses[-1]))
        model.eval()
        valid_loss = []
        for seq, out in valid_loader:
            seq, out = seq.to(device), out.to(device)
            pred = model(seq)
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

    sequence_tensor = transformed["POLYLINE_INIT"].apply(sequence2tensor).values
    output_tensor = transformed["POLYLINE_DEST"].apply(output2tensor).values
    seq_train, seq_valid, out_train, out_valid = \
        train_test_split(sequence_tensor, output_tensor, test_size=0.2, random_state=2023)

    train_data = SequenceDataset(seq_train, out_train)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_data = SequenceDataset(seq_valid, out_valid)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)
    rnn = RNN(3, 6, 1, 2)
    train_model(rnn, train_loader, valid_loader)
