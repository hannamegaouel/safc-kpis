import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =============================================
# CONFIGURATION DE LA PAGE
# =============================================
st.set_page_config(page_title="Value Creation Analysis", page_icon="⚽", layout="wide")

st.title("⚽ Value Creation Analysis")
st.subheader("50% of our players under contract are creating value")

# =============================================
# CHARGER LES DONNÉES
# =============================================
@st.cache_data
def load_data():
    df = pd.read_excel("KPI.xlsx")
    return df

df = load_data()

# =============================================
# FILTRES DANS LA SIDEBAR
# =============================================
st.sidebar.header("🔍 Filtres")

# Filtre 1: Statut de sélection
selection_filter = st.sidebar.multiselect(
    "Statut de sélection:",
    options=[0, 1],
    default=[0, 1],
    format_func=lambda x: "✅ Sélectionné" if x == 1 else "❌ Non sélectionné"
)

# Filtre 2: Tranche d'âge
age_range = st.sidebar.slider(
    "Tranche d'âge:",
    min_value=int(df['age'].min()),
    max_value=int(df['age'].max()),
    value=(int(df['age'].min()), int(df['age'].max()))
)

# Filtre 3: Temps de jeu
playtime_range = st.sidebar.slider(
    "Temps de jeu (%):",
    min_value=0,
    max_value=100,
    value=(0, 100)
)

# Filtre 4: Années au club
time_at_club_range = st.sidebar.slider(
    "Années au club:",
    min_value=int(df['Time'].min()),
    max_value=int(df['Time'].max()),
    value=(int(df['Time'].min()), int(df['Time'].max()))
)

# Filtre 5: Sélection de joueurs spécifiques
selected_players = st.sidebar.multiselect(
    "Joueurs spécifiques:",
    options=sorted(df['Name'].unique().tolist()),
    default=df['Name'].unique().tolist()
)

# =============================================
# APPLIQUER LES FILTRES
# =============================================
df_filtered = df[
    (df['selection'].isin(selection_filter)) &
    (df['age'] >= age_range[0]) &
    (df['age'] <= age_range[1]) &
    (df['playing_time_pct_PL'] >= playtime_range[0]) &
    (df['playing_time_pct_PL'] <= playtime_range[1]) &
    (df['Time'] >= time_at_club_range[0]) &
    (df['Time'] <= time_at_club_range[1]) &
    (df['Name'].isin(selected_players))
]

# Afficher le nombre de joueurs filtrés
st.sidebar.markdown("---")
st.sidebar.metric("Joueurs affichés", len(df_filtered), f"sur {len(df)}")

# =============================================
# CRÉER LE GRAPHIQUE
# =============================================
fig, ax = plt.subplots(figsize=(14, 9))

age_min = 18
age_max = 35

# BACKGROUND ZONES
# Layer 1: RED
ax.axhspan(-10, 0, color='#FFCDD2', alpha=0.8, zorder=0, label='_nolegend_')
ax.axhspan(0, 100, color='#FFCDD2', alpha=0.5, zorder=0)

# Layer 2: LIGHT ORANGE
ax.axhspan(0, 100, color='#FFE0B2', alpha=0.6, zorder=1)

# Create smooth diagonal line from (22, 10%) to (28, 40%)
ages_smooth = np.linspace(22, 28, 100)
thresholds_smooth = np.interp(ages_smooth, [22, 28], [10, 40])

# Layer 3: DARK GREEN - Combined zone
ages_green = np.concatenate([[18], [22], ages_smooth])
thresholds_green = np.concatenate([[0], [0], thresholds_smooth])
ax.fill_between(ages_green, thresholds_green, 100, 
                color='#2E7D32', alpha=0.4, zorder=2)

# Layer 4: LIGHT GREEN
ax.fill_between([28, 35], 40, 100, color='#C8E6C9', alpha=0.6, zorder=2)

# Threshold lines
ax.plot(ages_smooth, thresholds_smooth, 'k--', linewidth=2.5, alpha=0.7, zorder=3)
ax.plot([28, 35], [40, 40], 'k--', linewidth=2.5, alpha=0.7, zorder=3)

# =============================================
# PLOT PLAYERS (AVEC DONNÉES FILTRÉES)
# =============================================
for idx, row in df_filtered.iterrows():
    x = row['age']
    y = row['playing_time_pct_PL']
    line_length = row['Time']
    
    # Orange line extending LEFT
    ax.plot([x - line_length, x], [y, y], 
            color='orange', linewidth=3, alpha=0.6, zorder=4)
    
    # Blue dot
    ax.scatter(x, y, s=150, c='blue', 
               edgecolors='black', linewidths=2, zorder=5)
    
    # Player name ABOVE the dot
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

# =============================================
# AFFICHER LE GRAPHIQUE
# =============================================
st.pyplot(fig)

# =============================================
# STATISTIQUES
# =============================================
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

total_players = len(df_filtered)
selected = len(df_filtered[df_filtered['selection'] == 1])
not_selected = len(df_filtered[df_filtered['selection'] == 0])

col1.metric("Total Players", total_players)
col2.metric("Selected", selected, f"{selected/total_players*100:.0f}%" if total_players > 0 else "0%")
col3.metric("Avg Age", f"{df_filtered['age'].mean():.1f}" if total_players > 0 else "N/A")
col4.metric("Avg Playing Time", f"{df_filtered['playing_time_pct_PL'].mean():.1f}%" if total_players > 0 else "N/A")

# =============================================
# AFFICHER LES DONNÉES (optionnel)
# =============================================
if st.checkbox("Show filtered data"):
    st.dataframe(df_filtered[['Name', 'age', 'playing_time_pct_PL', 'Time', 'selection']])
