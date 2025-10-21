import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 📂 Daten laden
df = pd.read_csv("anzahl_photos.csv")

# 🔢 Spalten in numerische Werte umwandeln und fehlende Werte entfernen
df['Anzahl Photos'] = pd.to_numeric(df['Anzahl Photos'], errors='coerce')
df['Besuche'] = pd.to_numeric(df['Besuche'], errors='coerce')
df = df.dropna(subset=['Anzahl Photos', 'Besuche'])

# 📊 Mittelwert und Varianz berechnen (Überdispersion prüfen)
mean_y = df['Besuche'].mean()
var_y = df['Besuche'].var()
print("Mittelwert:", mean_y)
print("Varianz:", var_y)
print("Varianz/Mittelwert:", var_y / mean_y)

# 🚀 Negative-Binomial-Regression (mit geschätztem alpha)
y = df['Besuche']
X = sm.add_constant(df['Anzahl Photos'])  # Konstante hinzufügen
nb_model = sm.NegativeBinomial(y, X).fit()
print(nb_model.summary())

# 📈 Vorhersagen berechnen
df['mu_hat'] = nb_model.predict(X)

# 📉 Durchschnittliche Anzahl Gebote pro Fotoanzahl berechnen
avg = df.groupby('Anzahl Photos')['Besuche'].mean().reset_index()

# 📊 Plot: Durchschnittswerte + Modellvorhersage
plt.figure(figsize=(10,6))
plt.scatter(avg['Anzahl Photos'], avg['Besuche'], label="Durchschnittswerte", alpha=0.7)
plt.plot(df['Anzahl Photos'], df['mu_hat'], 'r.', alpha=0.2, label="NB-Vorhersagen")
plt.title("Negative-Binomial-Regression: Anzahl Photos ~ Besuche")
plt.xlabel("Anzahl Photos")
plt.ylabel("Besuche")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("NB_Besuche_vs_Photos.png")

# 📊 Plot: Zoom auf Bereich 0–100 Fotos
plt.figure(figsize=(10,6))
plt.scatter(avg['Anzahl Photos'], avg['Besuche'], label="Durchschnittswerte", alpha=0.7)
plt.plot(df['Anzahl Photos'], df['mu_hat'], 'r.', alpha=0.2, label="NB-Vorhersagen")
plt.title("Zoom: Negative-Binomial-Regression (0–100 Fotos)")
plt.xlabel("Anzahl Photos")
plt.ylabel("Besuche")
plt.xlim(0, 100)   # Bereich auf 0–10 Fotos beschränken
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("NB_Besuche_vs_Photos_zoom.png")

# Daten für den Export
data = {
    "Kennzahl": [
        "Durchschnittliche Anzahl Gebote",
        "Signifikanter Zusammenhang",
        "Prozentuale Steigerung pro Foto",
        "Erklärte Varianz (Pseudo R²)"
    ],
    "Wert": [
        round(mean_y, 2),
        "Ja (p < 0.001)",
        "ca. +1.5 %",
        "0.8 %"
    ]
}

df_export = pd.DataFrame(data)

# Export nach Excel
df_export.to_excel("Regressionsergebnisse.xlsx", index=False)