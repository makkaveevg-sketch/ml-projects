import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Admin\Desktop\python\cs2_all_tiers_games.csv")

# 1. Сначала выбираем признаки
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

# 2. Убираем пропуски
df_clean = df[features + ["team1_win"]].dropna()

X = df_clean[features]
y = df_clean["team1_win"]

# 3. Делим данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Обучаем
модель = RandomForestClassifier(n_estimators=100, random_state=42)
модель.fit(X_train, y_train)

# 5. Проверяем
точность = accuracy_score(y_test, модель.predict(X_test))
print(f"Точность: {точность*100:.1f}%")
