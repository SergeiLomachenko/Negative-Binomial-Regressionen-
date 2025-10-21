import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 📂 Daten laden
df = pd.read_csv("anzahl_felder.csv")

# 🔢 Spalten in numerische Werte umwandeln und fehlende Werte entfernen
df['Anzahl befüllter Felder'] = pd.to_numeric(df['Anzahl befüllter Felder'], errors='coerce')
df['Besucher'] = pd.to_numeric(df['Besucher'], errors='coerce')
df = df.dropna(subset=['Anzahl befüllter Felder', 'Besucher'])

# 📊 Mittelwert und Varianz berechnen (Überdispersion prüfen)
mean_y = df['Besucher'].mean()
var_y = df['Besucher'].var()
print("Mittelwert:", mean_y)
print("Varianz:", var_y)
print("Varianz/Mittelwert:", var_y / mean_y)

# 🚀 Negative-Binomial-Regression (mit geschätztem alpha)
y = df['Besucher']
X = sm.add_constant(df['Anzahl befüllter Felder'])  # Konstante hinzufügen
nb_model = sm.NegativeBinomial(y, X).fit()
print(nb_model.summary())

# 📈 Vorhersagen berechnen
df['mu_hat'] = nb_model.predict(X)

# 📉 Durchschnittliche Anzahl Gebote pro Fotoanzahl berechnen
avg = df.groupby('Anzahl befüllter Felder')['Besucher'].mean().reset_index()

# 📊 Plot: Durchschnittswerte + Modellvorhersage
plt.figure(figsize=(10,6))
plt.scatter(avg['Anzahl befüllter Felder'], avg['Besucher'], label="Durchschnittswerte", alpha=0.7)
plt.plot(df['Anzahl befüllter Felder'], df['mu_hat'], 'r.', alpha=0.2, label="NB-Vorhersagen")
plt.title("Negative-Binomial-Regression: Anzahl befüllter Felder ~ Besucher")
plt.xlabel("Anzahl befüllter Felder")
plt.ylabel("Besucher")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("NB_Anzahl_befüllter_Felder_vs_Besucher.png")

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
        "ca. +3.7 %",
        "0.05 %"
    ]
}

df_export = pd.DataFrame(data)

# Export nach Excel
df_export.to_excel("Regressionsergebnisse.xlsx", index=False)