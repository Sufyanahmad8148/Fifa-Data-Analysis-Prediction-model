"""
FIFA 24 - STREAMLINED STREAMLIT APPLICATION
Player search and rating predictions
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

st.set_page_config(
    page_title="FIFA 24 AI Rating Predictor",
    page_icon="⚽",
    layout="wide"
)

st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .growth-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
    }
    .growth-negative {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
    }
    .attribute-high {color: #27ae60; font-weight: bold;}
    .attribute-medium {color: #f39c12; font-weight: bold;}
    .attribute-low {color: #e74c3c; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)
@st.cache_data
def load_data():
    """Load and prepare player data"""
    try:
        if Path("players_with_predictions.csv").exists():
            df = pd.read_csv("players_with_predictions.csv")
            has_predictions = True
        elif Path("male_players.csv").exists():
            df = pd.read_csv("male_players.csv")
            has_predictions = False
        else:
            return None, False

        column_mapping = {
            "Name": "short_name", "Position": "positions", "Club": "club_name",
            "Nation": "nationality_name", "Overall": "overall", "Age": "age",
            "Pace": "pace", "Shooting": "shooting", "Passing": "passing",
            "Dribbling": "dribbling", "Defending": "defending", "Physicality": "physic",
            "Preferred foot": "preferred_foot", "Weak foot": "weak_foot", "Skill moves": "skill_moves"
        }
        df = df.rename(columns=column_mapping)

        if "potential" not in df.columns and "calculated_potential" not in df.columns:
            df["potential"] = df.apply(
                lambda r: min(r["overall"] + max(0, (28 - r["age"])) * 0.5, 99) if r["age"] < 28 else r["overall"],
                axis=1
            ).astype(int)

        for col, (low, high) in [("height_cm", (165, 195)), ("weight_kg", (60, 95))]:
            if col not in df.columns:
                df[col] = np.random.randint(low, high, len(df))

        return df, has_predictions
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, False

@st.cache_resource
def load_model():
    """Load trained model artifacts"""
    try:
        with open("fifa_rating_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("fifa_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("feature_columns.pkl", "rb") as f:
            features = pickle.load(f)
        return model, scaler, features, True
    except:
        return None, None, None, False

def get_rating_color(rating):
    if rating >= 85: return "🟢"
    elif rating >= 75: return "🟡"
    elif rating >= 65: return "🟠"
    return "🔴"

def get_attribute_class(value):
    if value >= 80: return "attribute-high"
    elif value >= 65: return "attribute-medium"
    return "attribute-low"

def calculate_derived_features(data_dict):
    """Calculate derived features for prediction"""
    derived = data_dict.copy()
    
    derived['pace_shooting_avg'] = (data_dict['pace'] + data_dict['shooting']) / 2
    derived['passing_dribbling_avg'] = (data_dict['passing'] + data_dict['dribbling']) / 2
    derived['technical_score'] = (data_dict['pace'] + data_dict['shooting'] + 
                                   data_dict['passing'] + data_dict['dribbling']) / 4
    derived['physical_score'] = (data_dict['defending'] + data_dict['physic']) / 2
    derived['age_squared'] = data_dict['age'] ** 2
    derived['skill_weak_product'] = data_dict['skill_moves'] * data_dict['weak_foot']
    
    return derived

df, has_predictions = load_data()
model, scaler, feature_columns, model_loaded = load_model()

st.sidebar.title("⚽ FIFA 24 AI Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate:", ["🔍 Player Search", "🤖 Rating Predictor", "📊 Insights"])

if df is not None:
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Players", f"{len(df):,}")
    st.sidebar.metric("Average Rating", f"{df['overall'].mean():.1f}")
    if has_predictions and 'ai_rating_change' in df.columns:
        avg_growth = df['ai_rating_change'].mean()
        st.sidebar.metric("Avg AI Predicted Growth", f"{avg_growth:+.2f}", delta=f"{avg_growth:.2f}")

if page == "🔍 Player Search":
    st.title("🔍 Player Search & Analysis")
    st.markdown("Search for any player to see their current stats and AI-predicted future potential")
    st.markdown("---")

    if df is None:
        st.error("Dataset not found. Please add 'male_players.csv' to the project folder.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("🔎 Enter player name", placeholder="e.g., Messi, Ronaldo, Haaland")
        with col2:
            search_type = st.selectbox("Search by", ["Name", "Club"])

        player = None

        if search_query:
            if search_type == "Name":
                matches = df[df["short_name"].str.contains(search_query, case=False, na=False)]
            else:
                matches = df[df["club_name"].str.contains(search_query, case=False, na=False)] if "club_name" in df.columns else pd.DataFrame()

            if len(matches) == 0:
                st.warning(f"No players found matching '{search_query}'.")
            else:
                options = [
                    f"{row['short_name']} - {row.get('club_name', 'N/A')} - {row.get('positions', 'N/A')} ({row['overall']})"
                    for _, row in matches.head(20).iterrows()
                ]
                selected = st.selectbox("Select a player", options)
                player = matches.iloc[options.index(selected)]

        if player is not None:
            st.markdown(f"## {player['short_name']}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"""<div class="metric-card"><h4>Current Rating</h4><h1>{player["overall"]:.0f} {get_rating_color(player["overall"])}</h1></div>""", unsafe_allow_html=True)
            
            with col2:
                potential_key = 'ai_predicted_potential' if has_predictions and 'ai_predicted_potential' in player else 'potential'
                potential_val = player.get(potential_key, player['overall'])
                st.markdown(f"""<div class="metric-card"><h4>{'AI ' if potential_key == 'ai_predicted_potential' else ''}Predicted</h4><h1>{potential_val:.1f}</h1></div>""", unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""<div class="metric-card"><h4>Age</h4><h1>{player["age"]:.0f}</h1></div>""", unsafe_allow_html=True)
            
            with col4:
                if has_predictions and "ai_rating_change" in player:
                    change = player["ai_rating_change"]
                    card_class = "growth-positive" if change > 0 else "growth-negative" if change < 0 else "metric-card"
                    st.markdown(f"""<div class="{card_class}"><h4>Growth</h4><h1>{change:+.1f}</h1></div>""", unsafe_allow_html=True)
            
            with col5:
                if 'value_eur' in player:
                    value_m = player['value_eur'] / 1_000_000
                    st.markdown(f"""<div class="metric-card"><h4>Value</h4><h1>€{value_m:.1f}M</h1></div>""", unsafe_allow_html=True)

            st.markdown("---")
            
            colL, colM, colR = st.columns(3)

            with colL:
                st.subheader("📋 Player Information")
                info = {
                    "Position": player.get("positions", "N/A"),
                    "Club": player.get("club_name", "N/A"),
                    "Nationality": player.get("nationality_name", "N/A"),
                    "Preferred Foot": player.get("preferred_foot", "N/A"),
                    "Height": f'{player.get("height_cm", 0):.0f} cm',
                    "Weight": f'{player.get("weight_kg", 0):.0f} kg',
                    "Weak Foot": f'{player.get("weak_foot", 0):.0f} ⭐',
                    "Skill Moves": f'{player.get("skill_moves", 0):.0f} ⭐'
                }
                for label, value in info.items():
                    st.text(f"{label}: {value}")

            with colM:
                st.subheader("⚡ Core Attributes")
                attrs = {
                    "Pace": player.get("pace"),
                    "Shooting": player.get("shooting"),
                    "Passing": player.get("passing"),
                    "Dribbling": player.get("dribbling"),
                    "Defending": player.get("defending"),
                    "Physical": player.get("physic")
                }
                attrs = {k: v for k, v in attrs.items() if pd.notna(v)}
                
                for attr_name, attr_val in attrs.items():
                    attr_class = get_attribute_class(attr_val)
                    st.markdown(f'<p class="{attr_class}">{attr_name}: {attr_val:.0f}/99</p>', unsafe_allow_html=True)

            with colR:
                st.subheader("📊 Attribute Radar")
                attrs_clean = {k: v for k, v in attrs.items() if pd.notna(v)}
                
                if attrs_clean:
                    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
                    
                    categories = list(attrs_clean.keys())
                    values = list(attrs_clean.values())
                    values += values[:1]  # Complete the circle
                    
                    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                    angles += angles[:1]
                    
                    ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
                    ax.fill(angles, values, alpha=0.25, color='#3498db')
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(categories)
                    ax.set_ylim(0, 100)
                    ax.grid(True)
                    
                    st.pyplot(fig)
                    plt.close()
        
            if has_predictions and 'ai_rating_change' in player:
                st.markdown("---")
                st.subheader("🔮 AI Prediction Insights")
                
                change = player['ai_rating_change']
                
                if change > 2:
                    st.success(f"🚀 **High Growth Potential!** This player is predicted to improve by {change:+.1f} points, making them a great investment for the future.")
                elif change > 0:
                    st.info(f"📈 **Steady Growth Expected.** This player should improve by {change:+.1f} points over time.")
                elif change < -1:
                    st.warning(f"📉 **Declining Phase.** This player may decline by {change:.1f} points due to age.")
                else:
                    st.info("➡️ **Stable Rating.** This player should maintain their current level.")
                
                st.markdown("**Key Factors in Prediction:**")
                if player['age'] < 23:
                    st.write("✓ Young age provides growth potential")
                if player.get('skill_moves', 0) >= 4:
                    st.write("✓ High skill moves indicate technical ability")
                if player.get('weak_foot', 0) >= 4:
                    st.write("✓ Strong weak foot shows versatility")
                
                tech_avg = (player.get('pace', 0) + player.get('shooting', 0) + 
                           player.get('passing', 0) + player.get('dribbling', 0)) / 4
                if tech_avg > player['overall'] + 2:
                    st.write("✓ Technical attributes exceed overall rating")

elif page == "🤖 Rating Predictor":
    st.title("🤖 Custom Player Rating Predictor")
    st.markdown("Create a custom player profile and see their predicted future potential")
    st.markdown("---")

    if not model_loaded:
        st.error("❌ Model not loaded. Run preprocessing and training scripts first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⚡ Technical Attributes")
            overall = st.slider("Current Overall", 40, 99, 75)
            age = st.slider("Age", 16, 45, 22)
            pace = st.slider("Pace", 0, 99, 70)
            shooting = st.slider("Shooting", 0, 99, 70)
            passing = st.slider("Passing", 0, 99, 70)
            dribbling = st.slider("Dribbling", 0, 99, 70)

        with col2:
            st.subheader("💪 Physical & Skills")
            defending = st.slider("Defending", 0, 99, 50)
            physic = st.slider("Physical", 0, 99, 70)
            height_cm = st.slider("Height (cm)", 150, 210, 180)
            weight_kg = st.slider("Weight (kg)", 50, 110, 75)
            weak_foot = st.slider("Weak Foot ⭐", 1, 5, 3)
            skill_moves = st.slider("Skill Moves ⭐", 1, 5, 3)

        col3, col4 = st.columns(2)
        with col3:
            intrep = st.slider("International Reputation", 1, 5, 2)
        with col4:
            value_eur = st.number_input("Market Value (€)", 0, 200_000_000, 5_000_000, 1_000_000)
        
        wage_eur = st.number_input("Weekly Wage (€)", 0, 1_000_000, 50_000, 5_000)

        if st.button("🔮 PREDICT FUTURE RATING", type="primary", use_container_width=True):
            input_dict = {
                "overall": overall, "age": age, "pace": pace, "shooting": shooting,
                "passing": passing, "dribbling": dribbling, "defending": defending,
                "physic": physic, "height_cm": height_cm, "weight_kg": weight_kg,
                "weak_foot": weak_foot, "skill_moves": skill_moves,
                "international_reputation": intrep, "value_eur": value_eur, "wage_eur": wage_eur
            }
            
            full_input = calculate_derived_features(input_dict)
 
            x = np.array([[full_input[col] for col in feature_columns]])
            x_scaled = scaler.transform(x)
            prediction = model.predict(x_scaled)[0]
            growth = prediction - overall

            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Rating", f"{overall:.1f}", delta=None)
            with col2:
                st.metric("Predicted Potential", f"{prediction:.1f}", delta=f"{growth:+.1f}")
            with col3:
                growth_pct = (growth / overall * 100) if overall > 0 else 0
                st.metric("Growth Percentage", f"{growth_pct:+.1f}%")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
           
            colors = ['#3498db', '#2ecc71'] if growth >= 0 else ['#3498db', '#e74c3c']
            ax1.bar(["Current", "Predicted"], [overall, prediction], color=colors, width=0.5)
            ax1.set_ylabel("Rating")
            ax1.set_ylim(0, 100)
            ax1.set_title("Rating Comparison")
            ax1.grid(True, alpha=0.3, axis="y")
            
            attrs = {
                'Pace': pace, 'Shooting': shooting, 'Passing': passing,
                'Dribbling': dribbling, 'Defending': defending, 'Physical': physic
            }
            ax2.barh(list(attrs.keys()), list(attrs.values()), color='steelblue')
            ax2.set_xlabel("Rating")
            ax2.set_xlim(0, 100)
            ax2.set_title("Attribute Breakdown")
            ax2.grid(True, alpha=0.3, axis="x")
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown("### 💡 Prediction Insights")
            if growth > 5:
                st.success(f"🚀 **Exceptional Growth!** This player profile shows potential to grow by {growth:+.1f} points!")
            elif growth > 2:
                st.success(f"📈 **Good Growth Potential!** Expected improvement of {growth:+.1f} points.")
            elif growth > 0:
                st.info(f"➡️ **Moderate Growth.** Predicted to improve by {growth:+.1f} points.")
            else:
                st.warning(f"📉 **Limited Growth.** Predicted change: {growth:+.1f} points.")
            
            st.markdown("**Key Factors Affecting Prediction:**")
            if age < 23:
                st.write("✅ Young age (under 23) - High growth potential")
            elif age > 30:
                st.write("⚠️ Age over 30 - Limited growth or potential decline")
            
            tech_score = (pace + shooting + passing + dribbling) / 4
            if tech_score > overall + 3:
                st.write("✅ Technical attributes exceed current rating")
            
            if skill_moves >= 4 and weak_foot >= 4:
                st.write("✅ High skill moves and weak foot rating")

elif page == "📊 Insights":
    st.title("📊 Dataset Insights & Statistics")
    st.markdown("---")
    
    if df is None:
        st.error("Dataset not found.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Players", f"{len(df):,}")
        with col2:
            st.metric("Avg Rating", f"{df['overall'].mean():.1f}")
        with col3:
            st.metric("Avg Age", f"{df['age'].mean():.1f}")
        with col4:
            if has_predictions and 'ai_rating_change' in df.columns:
                st.metric("Avg Growth", f"{df['ai_rating_change'].mean():+.2f}")
        
        st.markdown("---")
     
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top 10 Highest Rated")
            top_rated = df.nlargest(10, 'overall')[['short_name', 'overall', 'age', 'positions']]
            st.dataframe(top_rated, hide_index=True)
        
        with col2:
            if has_predictions and 'ai_rating_change' in df.columns:
                st.subheader("🚀 Top 10 Growth Potential")
                top_growth = df.nlargest(10, 'ai_rating_change')[['short_name', 'overall', 'ai_rating_change', 'age']]
                st.dataframe(top_growth, hide_index=True)
        
        st.markdown("---")
 
        st.subheader("📈 Distribution Analysis")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
 
        axes[0].hist(df['overall'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(df['overall'].mean(), color='red', linestyle='--', label=f'Mean: {df["overall"].mean():.1f}')
        axes[0].set_xlabel('Overall Rating')
        axes[0].set_ylabel('Number of Players')
        axes[0].set_title('Overall Rating Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(df['age'], bins=30, color='coral', edgecolor='black', alpha=0.7)
        axes[1].axvline(df['age'].mean(), color='red', linestyle='--', label=f'Mean: {df["age"].mean():.1f}')
        axes[1].set_xlabel('Age')
        axes[1].set_ylabel('Number of Players')
        axes[1].set_title('Age Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        if has_predictions and 'ai_rating_change' in df.columns:
            st.markdown("---")
            st.subheader("🔮 AI Prediction Analysis")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].hist(df['ai_rating_change'], bins=50, color='green', edgecolor='black', alpha=0.7)
            axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[0].set_xlabel('Predicted Rating Change')
            axes[0].set_ylabel('Number of Players')
            axes[0].set_title('Predicted Growth Distribution')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].scatter(df['age'], df['ai_rating_change'], alpha=0.3, s=10)
            axes[1].axhline(0, color='red', linestyle='--')
            axes[1].set_xlabel('Age')
            axes[1].set_ylabel('Predicted Rating Change')
            axes[1].set_title('Age vs Predicted Growth')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()