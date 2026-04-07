import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\Admin\Desktop\python\vgsales.csv")
df_clean = df.dropna()

# Кодируем текстовые столбцы в числа
le_platform = LabelEncoder()
le_genre = LabelEncoder()

df_clean = df_clean.copy()
df_clean["Platform_enc"] = le_platform.fit_transform(df_clean["Platform"])
df_clean["Genre_enc"] = le_genre.fit_transform(df_clean["Genre"])

X = df_clean[["Platform_enc", "Genre_enc", "Year"]]
y = df_clean["Global_Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Предсказываем для новой игры
ризнаки = ["Платформа", "Жанр", "Год"]
важность = rf.feature_importances_

def predict_sales(platform, genre, year):
    platform_enc = le_platform.transform([platform])[0]
    genre_enc = le_genre.transform([genre])[0]
    prediction = rf.predict([[platform_enc, genre_enc, year]])[0]
    print(f"{genre} игра на {platform} в {year} году: {prediction:.2f}M копий")

predict_sales("PS4", "Action", 2015)
predict_sales("Wii", "Sports", 2008)
predict_sales("DS", "Role-Playing", 2010)
predict_sales("PC", "Shooter", 2012)

признаки = ["Платформа", "Жанр", "Год"]
важность = rf.feature_importances_

plt.bar(признаки, важность, color="steelblue")
plt.title("Что влияет на продажи игр?")
plt.ylabel("Важность")
plt.show()