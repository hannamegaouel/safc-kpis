import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Value Creation Analysis", page_icon="⚽", layout="wide")

# Titre
st.title("⚽ Value Creation Analysis")
st.subheader("50% of our players under contract are creating value")

# Charger les données
@st.cache_data
def load_data():
    df = pd.read_excel("KPI.xlsx")  # Mettez le fichier dans le même dossier
    return df

df = load_data()

# Créer le graphique
fig, ax = plt.subplots(figsize=(14, 9))

age_min = 18
age_max = 35

# =============================================
# FIXED BACKGROUND ZONES
# =============================================

# LAYER 1: RED
ax.axhspan(-10, 0, color='#FFCDD2', alpha=0.8, zorder=0, label='_nolegend_')
ax.axhspan(0, 100, color='#FFCDD2', alpha=0.5, zorder=0)

# LAYER 2: LIGHT ORANGE
ax.axhspan(0, 100, color='#FFE0B2', alpha=0.6, zorder=1)

# Create smooth diagonal line
ages_smooth = np.linspace(22, 28, 100)
thresholds_smooth = np.interp(ages_smooth, [22, 28], [10, 40])

# LAYER 3: DARK GREEN - COMBINED zone
ages_green = np.concatenate([[18], [22], ages_smooth])
thresholds_green = np.concatenate([[0], [0], thresholds_smooth])
ax.fill_between(ages_green, thresholds_green, 100, 
                color='#2E7D32', alpha=0.4, zorder=2)

# LAYER 4: LIGHT GREEN
ax.fill_between([28, 35], 40, 100, color='#C8E6C9', alpha=0.6, zorder=2)

# Threshold lines
ax.plot(ages_smooth, thresholds_smooth, 'k--', linewidth=2.5, alpha=0.7, zorder=3)
ax.plot([28, 35], [40, 40], 'k--', linewidth=2.5, alpha=0.7, zorder=3)

# =============================================
# PLOT PLAYERS
# =============================================

for idx, row in df.iterrows():
    x = row['age']
    y = row['playing_time_pct_PL']
    line_length = row['Time']
    
    # Orange line
    ax.plot([x - line_length, x], [y, y], 
            color='orange', linewidth=3, alpha=0.6, zorder=4)
    
    # Blue dot
    ax.scatter(x, y, s=150, c='blue', 
               edgecolors='black', linewidths=2, zorder=5)
    
    # Player name
    ax.text(x, y + 3, row['Name'], fontsize=9, ha='center', va='bottom', zorder=6)

# =============================================
# FORMATTING
# =============================================

ax.set_xticks(range(18, 35))
ax.set_yticks(range(-10, 101, 10))
ax.axhline(y=0, color='black', linewidth=1.5, alpha=0.5, zorder=3)

ax.set_xlabel('Age', fontsize=13, fontweight='bold')
ax.set_ylabel('Playing Time (%)', fontsize=13, fontweight='bold')
ax.set_title('Value Creation\n50% of our players under contract are creating value', 
             fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.2, zorder=0)
ax.set_xlim(age_min, age_max)
ax.set_ylim(-10, 105)

# Legend
from matplotlib.patches import Patch
import matplotlib.lines as mlines

legend_elements = [
    Patch(facecolor='#2E7D32', alpha=0.4, label='Dark Green (Value Creation)'),
    Patch(facecolor='#C8E6C9', alpha=0.6, label='Light Green (Age 28+ & >40%)'),
    Patch(facecolor='#FFE0B2', alpha=0.6, label='Light Orange (Performance)'),
    Patch(facecolor='#FFCDD2', alpha=0.8, label='Red (Not Selected)'),
    mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, 
                  label='Threshold Line')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.95)

plt.tight_layout()

# Afficher dans Streamlit
st.pyplot(fig)

# Ajouter des statistiques
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

total_players = len(df)
selected = len(df[df['selection'] == 1])
not_selected = len(df[df['selection'] == 0])

col1.metric("Total Players", total_players)
col2.metric("Selected", selected, f"{selected/total_players*100:.0f}%")
col3.metric("Not Selected", not_selected, f"{not_selected/total_players*100:.0f}%")
col4.metric("Avg Age", f"{df['age'].mean():.1f}")

# Afficher le tableau de données
if st.checkbox("Show raw data"):
    st.dataframe(df)
```

---

## **Étape 2: Structure des fichiers**
```
votre_dossier/
├── app.py
└── KPI.xlsx