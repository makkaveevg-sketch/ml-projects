import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\Admin\Desktop\python\cs2_all_tiers_games.csv")

features = [
    "team1_player1_kills", "team1_player1_deaths", "team1_player1_adr","team1_player1_kast","team1_player1_kddiff",
    "team1_player2_kills", "team1_player2_deaths", "team1_player2_adr","team1_player2_kast","team1_player2_kddiff",
    "team1_player3_kills", "team1_player3_deaths", "team1_player3_adr","team1_player3_kast","team1_player3_kddiff",
    "team1_player4_kills", "team1_player4_deaths", "team1_player4_adr","team1_player4_kast","team1_player4_kddiff",
    "team1_player5_kills", "team1_player5_deaths", "team1_player5_adr","team1_player5_kast","team1_player5_kddiff",
    "team2_player1_kills", "team2_player1_deaths", "team2_player1_adr","team2_player1_kast","team2_player1_kddiff",
    "team2_player2_kills", "team2_player2_deaths", "team2_player2_adr","team2_player2_kast","team2_player2_kddiff",
    "team2_player3_kills", "team2_player3_deaths", "team2_player3_adr","team2_player3_kast","team2_player3_kddiff",
    "team2_player4_kills", "team2_player4_deaths", "team2_player4_adr","team2_player4_kast","team2_player4_kddiff",
    "team2_player5_kills", "team2_player5_deaths", "team2_player5_adr","team2_player5_kast","team2_player5_kddiff",
]

df_clean = df[features + ["team1_win"]].dropna()

X = df_clean[features].values
y = df_clean["team1_win"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

class CS2Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

model = CS2Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(290):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train).squeeze()
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_preds = (model(X_test).squeeze() > 0.5).float()
            acc = accuracy_score(y_test.numpy(), test_preds.numpy())
            print(f"Эпоха {epoch}: loss={loss.item():.4f} точность={acc*100:.1f}%")