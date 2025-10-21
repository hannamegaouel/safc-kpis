import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =============================================
# CONFIGURATION DE LA PAGE
# =============================================
st.set_page_config(page_title="Value Creation Analysis", page_icon="⚽", layout="wide")

st.title("⚽ Value Creation Analysis")

# =============================================
# CHARGER LES DONNÉES
# =============================================
@st.cache_data
def load_data():
    df = pd.read_excel("KPI.xlsx")
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    return df

try:
    df = load_data()
    
 
    
    # Check if 'value' column exists (case-insensitive)
    value_col = None
    for col in df.columns:
        if col.lower() == 'value':
            value_col = col
            break
    
    if value_col is None:
        st.error("⚠️ Column 'value' not found in Excel file. Please check your data.")
        st.info("Available columns: " + ", ".join(df.columns.tolist()))
        st.stop()
    
    # Rename to standardized 'value' if needed
    if value_col != 'value':
        df = df.rename(columns={value_col: 'value'})
    
except FileNotFoundError:
    st.error("⚠️ File 'KPI.xlsx' not found. Please upload the file.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading data: {str(e)}")
    st.stop()
# =============================================
# FILTRES DANS LA SIDEBAR
# =============================================

st.sidebar.header("🔍 Filters")

# Filtre 1: Statut de sélection
selection_filter = st.sidebar.multiselect(
    "Statut de sélection:",
    options=[0, 1],
    default=[0, 1],
    format_func=lambda x: "✅ Played / bench" if x == 1 else "❌ Never played / bench"
)

# Filtre 2: Tranche d'âge
age_range = st.sidebar.slider(
    "Age range:",
    min_value=int(df['age'].min()),
    max_value=int(df['age'].max()),
    value=(int(df['age'].min()), int(df['age'].max()))
)

# Filtre 3: Temps de jeu
playtime_range = st.sidebar.slider(
    "Playing time (%):",
    min_value=0,
    max_value=100,
    value=(0, 100)
)

# Filtre 4: Années au club
time_at_club_range = st.sidebar.slider(
    "Years at Club:",
    min_value=int(df['Time'].min()),
    max_value=int(df['Time'].max()),
    value=(int(df['Time'].min()), int(df['Time'].max()))
)

# Filtre 5: Sélection de joueurs spécifiques
selected_players = st.sidebar.multiselect(
    "Players selection:",
    options=sorted(df['Name'].unique().tolist()),
    default=df['Name'].unique().tolist()
)

# Option: Afficher les valeurs dans les zones
show_zone_values = st.sidebar.checkbox("Show values by zone", value=True)

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

# =============================================
# CALCULER LES VALEURS PAR ZONE
# =============================================

def get_player_zone(row):
    """Détermine la zone d'un joueur"""
    age = row['age']
    playtime = row['playing_time_pct_PL']
    selection = row['selection']
    
    if selection == 0:
        return "Red"
    elif age <= 22:
        return "Dark Green"
    elif age > 22 and age < 28:
        # Calcul du threshold pour cette tranche d'âge
        threshold = 10 + (age - 22) * 5  # 10% à 22 ans, augmente de 5% par an
        if playtime >= threshold:
            return "Dark Green"
        else:
            return "Orange"
    elif age >= 28 and playtime >= 40:
        return "Light Green"
    else:
        return "Orange"

# Ajouter la colonne zone
df_filtered['zone'] = df_filtered.apply(get_player_zone, axis=1)

# Calculer les valeurs totales par zone
zone_values = df_filtered.groupby('zone')['value'].sum()
zone_counts = df_filtered['zone'].value_counts()

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
# AFFICHER LES VALEURS PAR ZONE
# =============================================
if show_zone_values:
    
    # Zone Dark Green (jeunes)
    dark_green_value = zone_values.get('Dark Green', 0)
    dark_green_count = zone_counts.get('Dark Green', 0)
    if dark_green_count > 0:
        ax.text(20, 70, f'€{dark_green_value:,.0f}M\n({dark_green_count} players)', 
                fontsize=12, weight='bold', color='darkgreen', ha='center',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                         edgecolor='darkgreen', linewidth=2))
    
    # Zone Light Green (28+)
    light_green_value = zone_values.get('Light Green', 0)
    light_green_count = zone_counts.get('Light Green', 0)
    if light_green_count > 0:
        ax.text(31, 75, f'€{light_green_value:,.0f}M\n({light_green_count} players)', 
                fontsize=11, weight='bold', color='green', ha='center',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                         edgecolor='green', linewidth=2))
    
    # Zone Orange
    orange_value = zone_values.get('Orange', 0)
    orange_count = zone_counts.get('Orange', 0)
    if orange_count > 0:
        ax.text(30, 20, f'€{orange_value:,.0f}M\n({orange_count} players)', 
                fontsize=11, weight='bold', color='darkorange', ha='center',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                         edgecolor='darkorange', linewidth=2))
    
    # Zone Red
    red_value = zone_values.get('Red', 0)
    red_count = zone_counts.get('Red', 0)
    if red_count > 0:
        ax.text(26, -6, f'€{red_value:,.0f}M\n({red_count} players)', 
                fontsize=10, weight='bold', color='darkred', ha='center',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                         edgecolor='darkred', linewidth=2))

# =============================================
# PLOT PLAYERS (AVEC DONNÉES FILTRÉES)
# =============================================
for idx, row in df_filtered.iterrows():
    x = row['age']
    y = row['playing_time_pct_PL']
    line_length = row['Time']
    
    #  line extending LEFT
    ax.plot([x - line_length, x], [y, y], 
            color='grey', linewidth=3, alpha=0.6, zorder=4)
    
    #  dot
    ax.scatter(x, y, s=150, c='blue', 
               edgecolors='black', linewidths=2, zorder=5)
    
    # Player name ABOVE the dot
    ax.text(x, y + 3, row['Name'], fontsize=9, ha='center', va='bottom', zorder=6)

# =============================================
# FORMATTING
# =============================================
ax.set_xticks(range(18, 35))
ax.set_yticks(range(0, 101, 10))
ax.axhline(y=0, color='black', linewidth=1.5, alpha=0.5, zorder=3)

ax.set_xlabel('Age', fontsize=13, fontweight='bold')
ax.set_ylabel('Playing Time (%)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.2, zorder=0)
ax.set_xlim(age_min, age_max)
ax.set_ylim(-10, 100)

# Legend
from matplotlib.patches import Patch
import matplotlib.lines as mlines

legend_elements = [
    Patch(facecolor='#2E7D32', alpha=0.4, label='Value Creation'),
    Patch(facecolor='#C8E6C9', alpha=0.6, label='Performance zone'),
    Patch(facecolor='#FFE0B2', alpha=0.6, label='To monitor'),
    Patch(facecolor='#FFCDD2', alpha=0.8, label='Never Selected'),
   
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

# Statistiques principales
col1, col2, col3, col4 = st.columns(4)

total_players = len(df_filtered)
selected = len(df_filtered[df_filtered['selection'] == 1])
total_value = df_filtered['value'].sum()

col1.metric("Total Players", total_players)
col2.metric("Selected", selected, f"{selected/total_players*100:.0f}%" if total_players > 0 else "0%")
col3.metric("Total Value", f"€{total_value:,.0f}M")
col4.metric("Avg Value", f"€{total_value/total_players:,.1f}M" if total_players > 0 else "€0M")

# =============================================
# TABLEAU DÉTAILLÉ PAR ZONE
# =============================================
st.markdown("---")
st.subheader("📊 Breakdown by zone")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**🟢 Dark Green**")
    dg_count = zone_counts.get('Dark Green', 0)
    dg_value = zone_values.get('Dark Green', 0)
    st.metric("Players", dg_count)
    st.metric("Total value", f"€{dg_value:,.0f}M")
    if dg_count > 0:
        st.metric("Average value", f"€{dg_value/dg_count:,.1f}M")

with col2:
    st.markdown("**🟩 Light Green**")
    lg_count = zone_counts.get('Light Green', 0)
    lg_value = zone_values.get('Light Green', 0)
    st.metric("Players", lg_count)
    st.metric("Total value", f"€{lg_value:,.0f}M")
    if lg_count > 0:
        st.metric("Average value", f"€{lg_value/lg_count:,.1f}M")

with col3:
    st.markdown("**🟧 Orange**")
    o_count = zone_counts.get('Orange', 0)
    o_value = zone_values.get('Orange', 0)
    st.metric("Players", o_count)
    st.metric("Total value", f"€{o_value:,.0f}M")
    if o_count > 0:
        st.metric("Average value", f"€{o_value/o_count:,.1f}M")

with col4:
    st.markdown("**🟥 Red**")
    r_count = zone_counts.get('Red', 0)
    r_value = zone_values.get('Red', 0)
    st.metric("Players", r_count)
    st.metric("Total value", f"€{r_value:,.0f}M")
    if r_count > 0:
        st.metric("Average value", f"€{r_value/r_count:,.1f}M")

# =============================================
# AFFICHER LES DONNÉES (optionnel)
# =============================================
st.markdown("---")
if st.checkbox("Show filtered data"):
    st.dataframe(df_filtered[['Name', 'age', 'playing_time_pct_PL', 'Time', 'selection', 'value', 'zone']])
















