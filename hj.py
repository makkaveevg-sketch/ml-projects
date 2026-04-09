import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
    "team2_player5_kills", "team2_player5_deaths", "team2_player5_adr","team2_player5_kast","team2_player5_kddiff","team1_id", "team2_id"
]

df_clean = df[features + ["map_name"]].dropna()

X = df_clean[features]
y = df_clean["map_name"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

модель_карта = RandomForestClassifier(n_estimators=100, random_state=42)
модель_карта.fit(X_train, y_train)

точность_карта = accuracy_score(y_test, модель_карта.predict(X_test))
print(f"Точность предсказания карты: {точность_карта*100:.1f}%")


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

X = df_clean[features]
y = df_clean["team1_win"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

модель_победитель = RandomForestClassifier(n_estimators=100, random_state=42)
модель_победитель.fit(X_train, y_train)

точность_победитель = accuracy_score(y_test, модель_победитель.predict(X_test))
print(f"Точность предсказания победителя: {точность_победитель*100:.1f}%")