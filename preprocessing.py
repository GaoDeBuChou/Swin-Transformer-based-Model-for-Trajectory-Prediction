import numpy as np
import pandas as pd
import torch


def _change_type(polyline):
    polyline = polyline[2:-2]
    cords_raw = polyline.split("],[")
    cords = []
    max_lon, min_lon, max_lat, min_lat = float("-inf"), float("inf"), float("-inf"), float("inf")
    for cord in cords_raw:
        cords.append([float(loc) for loc in cord.split(",")])
        max_lon = max(max_lon, cords[-1][0])
        min_lon = min(min_lon, cords[-1][0])
        max_lat = max(max_lat, cords[-1][1])
        min_lat = min(min_lat, cords[-1][1])
    return pd.Series({"POLYLINE": cords, "max_lon": max_lon, "min_lon": min_lon, "max_lat": max_lat, "min_lat": min_lat})


def filter_map(train, max_lat, min_lat, max_lon, min_lon):
    train.reset_index(drop=True, inplace=True)
    return train[(train["max_lat"] <= max_lat) & (train["min_lat"] >= min_lat) &
                 (train["max_lon"] <= max_lon) & (train["min_lon"] >= min_lon)]


def _normalize(polyline, max_lon, min_lon, max_lat, min_lat):
    final = [[(cord[0]-min_lon)/(max_lon-min_lon), (cord[1] - min_lat) / (max_lat - min_lat)] for cord in polyline]
    return pd.Series({"POLYLINE_INIT": final[:-1], "POLYLINE_DEST": final[-1]})


def _to_matrix(polyline, m):
    mat = np.zeros((m, m))
    n = len(polyline)
    for i in range(n):
        x = min(m - 1, int(polyline[i][0] * m))
        y = min(m - 1, int(polyline[i][1] * m))
        mat[y][x] = (i + 1) / n
    return mat


def transform(df_train, m):
    # Change type
    changed = df_train["POLYLINE"].apply(_change_type)
    # Filter map for max/min long/lat
    changed = filter_map(changed, 41.2, 41, -8.6, -8.7)
    # Get min-max
    max_longitude = changed["max_lon"].max()
    min_longitude = changed["min_lon"].min()
    max_latitude = changed["max_lat"].max()
    min_latitude = changed["min_lat"].min()
    # Normalize min-max and split
    cleaned = changed["POLYLINE"].apply(_normalize, args=(max_longitude, min_longitude, max_latitude, min_latitude))
    # Transform to matrices
    cleaned["MATRIX"] = cleaned["POLYLINE_INIT"].apply(_to_matrix, args=(m,))
    return cleaned


def sequence2tensor(sequence, seqlen=200):
    if len(sequence) > seqlen:
        sequence = sequence[-seqlen:]
    tensor = torch.zeros(seqlen, 3)
    for i, point in enumerate(sequence, start=seqlen-len(sequence)):
        tensor[i, :2] = torch.Tensor(point)
        tensor[i, 2] = 1  # 1 for in the sequence, 0 for out of the sequence
    return tensor


def matrix2tensor(matrix):
    return torch.Tensor(matrix)


def output2tensor(output):
    return torch.Tensor(output)


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
    # Save to CSV
    transformed.to_csv("data/transformed.csv", index=False)
