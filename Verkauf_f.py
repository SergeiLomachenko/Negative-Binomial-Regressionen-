import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 📂 Daten laden
df = pd.read_csv("anzahl_felder.csv")

# 🔢 Umwandeln
df['Anzahl befüllter Felder'] = pd.to_numeric(df['Anzahl befüllter Felder'], errors='coerce')
df['Verkauf'] = df['Verkauf'].map({'ja': 1, 'nein': 0})
df = df.dropna(subset=['Anzahl befüllter Felder', 'Verkauf'])

# 🚀 Logistische Regression
y = df['Verkauf']
X = sm.add_constant(df['Anzahl befüllter Felder'])
logit_model = sm.Logit(y, X).fit()
print(logit_model.summary())

# 📈 Vorhersagen
df['p_hat'] = logit_model.predict(X)

# 📊 Plot: Wahrscheinlichkeit Verkauf ~ Anzahl Photos 
avg = df.groupby('Anzahl befüllter Felder')['Verkauf'].mean().reset_index()
photo_range = np.arange(0, df['Anzahl befüllter Felder'].max() + 1)
X_pred = sm.add_constant(photo_range)
p_pred = logit_model.predict(X_pred)

plt.figure(figsize=(10,6))
plt.scatter(avg['Anzahl befüllter Felder'], avg['Verkauf'], 
            label="Durchschnittliche Verkaufswahrscheinlichkeit (tatsächlich)", alpha=0.8)
plt.plot(photo_range, p_pred, 'r-', label="Modellprognose (Logit)")
plt.title("Verkaufswahrscheinlichkeit ~ Anzahl befüllter Felder")
plt.xlabel("Anzahl befüllter Felder")
plt.ylabel("Verkaufswahrscheinlichkeit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Logit_Verkauf_vs_Anzahl_befüllter_Felder_avg.png")


# 📊 Export der Kennzahlen
data = {
    "Kennzahl": [
        "Durchschnittliche Verkaufsrate",
        "Signifikanter Zusammenhang",
        "Odds-Ratio pro Foto",
        "Pseudo R²"
    ],
    "Wert": [
        round(df['Verkauf'].mean(), 3),
        "Ja (p < 0.001)" if logit_model.pvalues['Anzahl befüllter Felder'] < 0.001 else "Nein",
        round(np.exp(logit_model.params['Anzahl befüllter Felder']), 3),
        round(1 - logit_model.llf/logit_model.llnull, 4)
    ]
}
pd.DataFrame(data).to_excel("Regressionsergebnisse.xlsx", index=False)
