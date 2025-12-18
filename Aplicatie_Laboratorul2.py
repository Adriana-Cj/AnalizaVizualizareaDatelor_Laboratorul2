import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import io
from matplotlib.backends.backend_pdf import PdfPages

# Configurare pagină
st.set_page_config(
    page_title="Analiză vinuri",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizat
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 2rem;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)


# Funcție pentru generare PDF din figură matplotlib
def generate_pdf_matplotlib(fig):
    """Generează PDF din figură matplotlib"""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


# Funcție pentru generare PDF din figură plotly
def generate_pdf_plotly(fig):
    """Generează PDF din figură plotly"""
    buf = io.BytesIO()
    fig.write_image(buf, format='pdf')
    buf.seek(0)
    return buf.getvalue()


# Încărcare date
@st.cache_data
def load_data():
    """Încarcă și pregătește datele"""
    try:
        df = pd.read_csv("wine_clean_final.csv")
        # Calculare raport preț/calitate dacă nu există
        if 'price_quality_ratio' not in df.columns:
            df['price_quality_ratio'] = df['points'] / df['price'].replace(0, np.nan)
        return df
    except FileNotFoundError:
        st.error("Fișierul 'wine_clean_final.csv' nu a fost găsit!")
        st.stop()
    except Exception as e:
        st.error(f"Eroare la încărcarea datelor: {e}")
        st.stop()


# Încărcare date
df_original = load_data()

# ========== SIDEBAR - FILTRE ==========
st.sidebar.title("Filtre avansate")

# Filtru Țară
all_countries = sorted(df_original['country'].dropna().unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Țări",
    options=all_countries,
    # default=all_countries
    default=[]
)

# Filtru Preț
price_min = float(df_original['price'].min())
price_max = float(df_original['price'].max())
price_range = st.sidebar.slider(
    "Interval Preț ($)",
    min_value=price_min,
    max_value=price_max,
    value=(price_min, price_max)
)

# Filtru Punctaj
points_min = int(df_original['points'].min())
points_max = int(df_original['points'].max())
points_range = st.sidebar.slider(
    "Interval Punctaj",
    min_value=points_min,
    max_value=points_max,
    value=(points_min, points_max)
)

# Filtru Categorie - MODIFICAT pentru selecție multiplă
if 'category' in df_original.columns:
    all_categories = sorted(df_original['category'].dropna().unique().tolist())
    selected_categories = st.sidebar.multiselect(
        "Categorii",
        options=all_categories,
        # default=all_categories
        default=[]
    )
else:
    selected_categories = []

# Filtru Raport Preț/Calitate
if 'price_quality_ratio' in df_original.columns:
    ratio_min = float(df_original['price_quality_ratio'].min())
    ratio_max = float(df_original['price_quality_ratio'].max())
    ratio_range = st.sidebar.slider(
        "Raport Preț/Calitate",
        min_value=ratio_min,
        max_value=ratio_max,
        value=(ratio_min, ratio_max)
    )
else:
    ratio_range = None

# Aplicare filtre
df_filtered = df_original.copy()

if selected_countries:
    df_filtered = df_filtered[df_filtered['country'].isin(selected_countries)]

df_filtered = df_filtered[
    (df_filtered['price'] >= price_range[0]) &
    (df_filtered['price'] <= price_range[1])
    ]

df_filtered = df_filtered[
    (df_filtered['points'] >= points_range[0]) &
    (df_filtered['points'] <= points_range[1])
    ]

# filtrare pentru multiple categorii
# if selected_categories and 'category' in df_filtered.columns:
if selected_categories:
    df_filtered = df_filtered[df_filtered['category'].isin(selected_categories)]

if ratio_range and 'price_quality_ratio' in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered['price_quality_ratio'] >= ratio_range[0]) &
        (df_filtered['price_quality_ratio'] <= ratio_range[1])
        ]

# Afișare statistici filtrare
st.sidebar.markdown("---")
filtered_count = len(df_filtered)
total_count = len(df_original)
st.sidebar.markdown(
    f"""<div style='padding: 10px; border: 2px solid #ff4b4b; border-radius: 5px; background-color: #ffe5e5;'>
    <p style='margin: 0; color: #333; font-size: 14px;'>
    Înregistrări filtrate: <strong>{filtered_count:,}</strong> din <strong>{total_count:,}</strong>
    </p>
    </div>""",
    unsafe_allow_html=True
)

# ========== HEADER ==========
st.title("Aplicație de analiză a vinurilor")
st.markdown("Explorează și analizează datele despre vinuri folosind filtre și vizualizări interactive.")

# ========== CĂUTARE TEXTUALĂ ==========
st.markdown("---")
with st.expander("Căutare după descriere"):
    search_query = st.text_input(
        "Introdu cuvinte cheie pentru căutare în descrieri:",
        placeholder="ex: oak, fruity, elegant"
    )

    if search_query:
        # Căutare simplă în descrieri
        search_terms = search_query.lower().split()
        mask = df_filtered['description'].str.lower().str.contains(
            '|'.join(search_terms),
            na=False,
            regex=True
        )
        df_filtered = df_filtered[mask]
        st.markdown(
            f"""<div style='padding: 10px; border: 2px solid #ff4b4b; border-radius: 5px; background-color: #ffe5e5;'>
            <p style='margin: 0; color: #333; font-size: 14px;'>
            Găsite <strong>{len(df_filtered):,}</strong> vinuri care corespund criteriilor de căutare.
            </p>
            </div>""",
            unsafe_allow_html=True
        )

# Verificare dacă avem date după filtrare
if len(df_filtered) == 0:
    st.warning("Nu există date care să corespundă filtrelor selectate. Te rugăm să ajustezi criteriile.")
    st.stop()

# ========== TABS PENTRU VIZUALIZĂRI ==========
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Corelații generale",
    "Analiză regională",
    "Analiză varietăți",
    "Analiză textuală",
    "Distribuții și comparații"
])

# ========== TAB 1: CORELAȚII GENERALE ==========
with tab1:
    st.header("Corelații generale")

    # Viz 1: Corelație Preț vs Points
    st.subheader("Analiza corelației preț-calitate")

    if len(df_filtered) >= 2:
        col1, col2 = st.columns([3, 1])

        with col1:
            corr = df_filtered[['price', 'points']].corr()

            fig1, ax1 = plt.subplots(figsize=(6, 6))
            im = ax1.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            ax1.set_xticks([0, 1])
            ax1.set_xticklabels(['price', 'points'])
            ax1.set_yticks([0, 1])
            ax1.set_yticklabels(['price', 'points'])
            ax1.set_title("Corelație: Preț vs Points", fontsize=14, pad=15)
            plt.colorbar(im, ax=ax1)

            for i in range(2):
                for j in range(2):
                    ax1.text(j, i, f"{corr.values[i, j]:.2f}",
                             ha="center", va="center", fontsize=12)

            plt.tight_layout()
            st.pyplot(fig1)

        with col2:
            st.caption(
                "Această matrice arată corelația dintre preț și punctajul vinurilor. O valoare apropiată de 1 indică o corelație puternică pozitivă.")
            st.download_button(
                "Descarcă PDF",
                data=generate_pdf_matplotlib(fig1),
                file_name="correlatie_pret_punctaj.pdf",
                mime="application/pdf",
                key="dl_corr_price_points"
            )
    else:
        st.warning("Nu există suficiente date pentru a calcula corelația.")

    st.markdown("---")

    # Viz 2: Corelație Alcool vs Points
    st.subheader("Analiza corelației conținut alcoolic-calitat")

    if 'alcohol' in df_filtered.columns and len(df_filtered) >= 2:
        col1, col2 = st.columns([3, 1])

        with col1:
            df_alc = df_filtered[['alcohol', 'points']].dropna()

            if len(df_alc) >= 2:
                corr = df_alc.corr()

                fig2, ax2 = plt.subplots(figsize=(6, 6))
                im = ax2.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
                ax2.set_xticks([0, 1])
                ax2.set_xticklabels(['alcohol', 'points'])
                ax2.set_yticks([0, 1])
                ax2.set_yticklabels(['alcohol', 'points'])
                ax2.set_title("Corelație: Alcool vs Points", fontsize=14, pad=15)
                plt.colorbar(im, ax=ax2)

                for i in range(2):
                    for j in range(2):
                        ax2.text(j, i, f"{corr.values[i, j]:.2f}",
                                 ha="center", va="center", fontsize=12)

                plt.tight_layout()
                st.pyplot(fig2)
            else:
                st.warning("Nu există suficiente date cu alcool pentru a calcula corelația.")

        with col2:
            st.caption(
                "Corelația dintre conținutul de alcool și punctajul vinurilor. Permite identificarea legăturii dintregrad alcoolic și calitatea percepută.")
            if len(df_alc) >= 2:
                st.download_button(
                    "Descarcă PDF",
                    data=generate_pdf_matplotlib(fig2),
                    file_name="correlatie_alcool_punctaj.pdf",
                    mime="application/pdf",
                    key="dl_corr_alcohol_points"
                )
    else:
        st.info("Coloana 'alcohol' nu este disponibilă în dataset.")

    st.markdown("---")

    # Viz 7: Matrice de corelație - Variabile Numerice
    st.subheader("Analiza interdependențelor numerice")

    col1, col2 = st.columns([3, 1])

    with col1:
        df_numeric = df_filtered.select_dtypes(include=[np.number])

        if len(df_numeric.columns) >= 2 and len(df_numeric) >= 2:
            corr_matrix = df_numeric.corr()

            fig7, ax7 = plt.subplots(figsize=(10, 8))
            im = ax7.imshow(corr_matrix, vmin=-1, vmax=1, cmap='coolwarm')
            ax7.set_xticks(range(len(corr_matrix.columns)))
            ax7.set_xticklabels(corr_matrix.columns, rotation=90)
            ax7.set_yticks(range(len(corr_matrix.columns)))
            ax7.set_yticklabels(corr_matrix.columns)
            ax7.set_title("Matrice de corelație – variabile numerice", fontsize=14, pad=15)
            plt.colorbar(im, ax=ax7)

            for i in range(corr_matrix.shape[0]):
                for j in range(corr_matrix.shape[1]):
                    ax7.text(j, i, f"{corr_matrix.values[i, j]:.2f}",
                             ha="center", va="center", fontsize=9)

            plt.tight_layout()
            st.pyplot(fig7)
        else:
            st.warning("Nu există suficiente variabile numerice pentru a calcula matricea de corelație.")

    with col2:
        st.caption(
            "Matricea completă de corelație pentru toate variabilele numerice disponibile în dataset. Oferă o imagine de ansamblu asupra relațiilor dintre diferitele caracteristici.")
        if len(df_numeric.columns) >= 2 and len(df_numeric) >= 2:
            st.download_button(
                "Descarcă PDF",
                data=generate_pdf_matplotlib(fig7),
                file_name="matrice_correlatie_numerice.pdf",
                mime="application/pdf",
                key="dl_corr_matrix"
            )

# ========== TAB 2: ANALIZĂ REGIONALĂ ==========
with tab2:
    st.header("Analiză regională")

    # Viz 3: Corelație Regiuni vs Price/Points
    st.subheader("Analiza regiunilor predominante: corelații cu preț și calitate")

    if 'region_1' in df_filtered.columns:
        col1, col2 = st.columns([3, 1])

        with col1:
            top_regions = df_filtered['region_1'].value_counts().head(20).index

            if len(top_regions) > 0:
                ohe = pd.get_dummies(df_filtered['region_1'])[top_regions]
                corr = pd.concat([df_filtered[['price', 'points']], ohe], axis=1).corr().iloc[:2, 2:]

                fig3, ax3 = plt.subplots(figsize=(14, 4))
                im = ax3.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                ax3.set_xticks(range(corr.shape[1]))
                ax3.set_xticklabels(corr.columns, rotation=90)
                ax3.set_yticks([0, 1])
                ax3.set_yticklabels(['price', 'points'])
                ax3.set_title("Corelație: Regiuni vs Price/Points", fontsize=14, pad=15)
                plt.colorbar(im, ax=ax3)

                for i in range(2):
                    for j in range(corr.shape[1]):
                        ax3.text(j, i, f"{corr.values[i, j]:.2f}",
                                 ha="center", va="center", fontsize=8)

                plt.tight_layout()
                st.pyplot(fig3)
            else:
                st.warning("Nu există suficiente regiuni pentru această vizualizare.")

        with col2:
            st.caption(
                "Corelația dintre cele mai frecvente 20 regiuni și prețul/punctajul vinurilor. Identifică regiunile asociate cu vinuri mai scumpe sau mai bine evaluate.")
            if len(top_regions) > 0:
                st.download_button(
                    "Descarcă PDF",
                    data=generate_pdf_matplotlib(fig3),
                    file_name="correlatie_regiuni.pdf",
                    mime="application/pdf",
                    key="dl_corr_regions"
                )
    else:
        st.info("Coloana 'region_1' nu este disponibilă în dataset.")

    st.markdown("---")

    # Viz 11: Prețuri Medii pe Țară
    st.subheader("Analiza comparativă: prețuri medii pe piețe geografice")

    col1, col2 = st.columns([3, 1])

    with col1:
        mean_price_country = df_filtered.groupby('country')['price'].mean().sort_values(ascending=False)

        if len(mean_price_country) > 0:
            fig11, ax11 = plt.subplots(figsize=(16, 8))
            bars = ax11.bar(range(len(mean_price_country)), mean_price_country.values,
                            color=sns.color_palette("viridis", len(mean_price_country)))
            ax11.set_xticks(range(len(mean_price_country)))
            ax11.set_xticklabels(mean_price_country.index, rotation=90, ha='right')
            ax11.set_title("Prețurile medii ale vinurilor pe țară", fontsize=16, fontweight='bold')
            ax11.set_xlabel("Țara", fontsize=13)
            ax11.set_ylabel("Preț mediu ($)", fontsize=13)
            ax11.grid(axis='y', alpha=0.3)

            for i, value in enumerate(mean_price_country.values):
                ax11.text(i, value + (value * 0.015), f"{value:.1f}",
                          ha='center', va='bottom', fontsize=10, rotation=90)

            plt.tight_layout()
            st.pyplot(fig11)
        else:
            st.warning("Nu există date suficiente pentru această vizualizare.")

    with col2:
        st.caption(
            "Comparația prețurilor medii ale vinurilor între diferite țări. Permite identificarea piețelor cu vinuri premium sau accesibile.")
        if len(mean_price_country) > 0:
            st.download_button(
                "Descarcă PDF",
                data=generate_pdf_matplotlib(fig11),
                file_name="preturi_medii_tara.pdf",
                mime="application/pdf",
                key="dl_price_country"
            )

    st.markdown("---")

    # Viz 12: Distribuția după Categorii și Regiuni
    st.subheader("Profilul categorial al regiunilor predominante")

    if 'category' in df_filtered.columns and 'region_1' in df_filtered.columns:
        col1, col2 = st.columns([3, 1])

        with col1:
            df_cat_reg = df_filtered.dropna(subset=['category', 'region_1'])

            if len(df_cat_reg) > 0:
                counts = df_cat_reg.groupby(['region_1', 'category']).size().unstack(fill_value=0)
                top_regions = counts.sum(axis=1).sort_values(ascending=False).head(20).index
                counts_top = counts.loc[top_regions]

                colors = sns.color_palette("Set1", n_colors=len(counts_top.columns))

                fig12, ax12 = plt.subplots(figsize=(14, 8))
                counts_top.plot(kind='barh', stacked=True, ax=ax12, color=colors,
                                edgecolor=None, linewidth=0, alpha=0.9)
                ax12.set_title("Distribuția vinurilor după categorii și regiuni (top 20 regiuni)",
                               fontsize=16, weight='bold')
                ax12.set_xlabel("Număr de vinuri", fontsize=14)
                ax12.set_ylabel("Regiune", fontsize=14)
                ax12.legend(title="Categorie", fontsize=10, title_fontsize=11,
                            bbox_to_anchor=(1.05, 1), loc='upper left')
                ax12.grid(axis='x', alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig12)
            else:
                st.warning("Nu există date suficiente pentru această vizualizare.")

        with col2:
            st.caption(
                "Distribuția tipurilor de vinuri (categorii) în cele mai importante 20 regiuni. Arată specializarea regională în producția de vinuri.")
            if len(df_cat_reg) > 0:
                st.download_button(
                    "Descarcă PDF",
                    data=generate_pdf_matplotlib(fig12),
                    file_name="distributie_categorii_regiuni.pdf",
                    mime="application/pdf",
                    key="dl_cat_regions"
                )
    else:
        st.info("Coloanele 'category' sau 'region_1' nu sunt disponibile.")

# ========== TAB 3: ANALIZĂ VARIETĂȚI ==========
with tab3:
    st.header("Analiză varietăți de vinuri")

    # Viz 4: Corelație Varietăți vs Price/Points
    st.subheader("Analiza varietăților predominante: corelații cu preț și calitate")

    if 'variety' in df_filtered.columns:
        col1, col2 = st.columns([3, 1])

        with col1:
            top_var = df_filtered['variety'].value_counts().head(20).index

            if len(top_var) > 0:
                ohe = pd.get_dummies(df_filtered['variety'])[top_var]
                corr = pd.concat([df_filtered[['price', 'points']], ohe], axis=1).corr().iloc[:2, 2:]

                fig4, ax4 = plt.subplots(figsize=(16, 6))
                im = ax4.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                ax4.set_xticks(range(corr.shape[1]))
                ax4.set_xticklabels(corr.columns, rotation=90)
                ax4.set_yticks([0, 1])
                ax4.set_yticklabels(['price', 'points'])
                ax4.set_title("Corelație: Varietăți vs Price/Points", fontsize=14, pad=15)
                plt.colorbar(im, ax=ax4)

                for i in range(2):
                    for j in range(corr.shape[1]):
                        ax4.text(j, i, f"{corr.values[i, j]:.2f}",
                                 ha="center", va="center", fontsize=8)

                plt.tight_layout()
                st.pyplot(fig4)
            else:
                st.warning("Nu există suficiente varietăți pentru această vizualizare.")

        with col2:
            st.caption(
                "Corelația dintre cele mai comune 20 varietăți de vinuri și prețul/punctajul acestora. Identifică varietățile asociate cu vinuri premium.")
            if len(top_var) > 0:
                st.download_button(
                    "Descarcă PDF",
                    data=generate_pdf_matplotlib(fig4),
                    file_name="correlatie_varietati.pdf",
                    mime="application/pdf",
                    key="dl_corr_varieties"
                )
    else:
        st.info("Coloana 'variety' nu este disponibilă în dataset.")

# ========== TAB 4: ANALIZĂ TEXTUALĂ ==========
with tab4:
    st.header("Analiză textuală a descrierilor")

    # Viz 5: Corelație Top 20 Cuvinte
    st.subheader("Analiza frecvenței terminologice: corelații cu preț și rating")

    if 'description' in df_filtered.columns:
        col1, col2 = st.columns([3, 1])

        with col1:
            with st.spinner("Procesare text..."):
                all_words = " ".join(df_filtered["description"].dropna())
                words = re.findall(r'\b[a-z0-9]+\b', all_words.lower())
                word_counts = Counter(words)
                top20_words = [w for w, c in word_counts.most_common(20)]

                if len(top20_words) > 0:
                    df_temp = df_filtered.copy()
                    for word in top20_words:
                        df_temp[f"{word}"] = df_temp["description"].str.count(rf"\b{word}\b")

                    corr_cols = top20_words + ["price", "points"]
                    corr_matrix = df_temp[corr_cols].corr().loc[["price", "points"], top20_words]

                    fig5, ax5 = plt.subplots(figsize=(14, 3))
                    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                                linewidths=.5, ax=ax5)
                    ax5.set_title("Correlația top 20 cuvinte cu preț și rating", fontsize=14)
                    ax5.set_xlabel("Cuvinte")
                    plt.tight_layout()
                    st.pyplot(fig5)
                else:
                    st.warning("Nu s-au găsit cuvinte în descrieri.")

        with col2:
            st.caption(
                "Identifică cuvintele din descrieri care sunt cel mai puternic asociate cu prețuri mari sau punctaje ridicate. Util pentru înțelegerea vocabularului calității.")
            if len(top20_words) > 0:
                st.download_button(
                    "Descarcă PDF",
                    data=generate_pdf_matplotlib(fig5),
                    file_name="correlatie_cuvinte.pdf",
                    mime="application/pdf",
                    key="dl_corr_words"
                )
    else:
        st.info("Coloana 'description' nu este disponibilă în dataset.")

    st.markdown("---")

    # Viz 6: Heatmap Cuvinte vs Soiuri
    st.subheader("Caracterizare lingvistică a soiurilor predominante")

    if 'description' in df_filtered.columns and 'variety' in df_filtered.columns:
        col1, col2 = st.columns([3, 1])

        with col1:
            with st.spinner("Procesare text și soiuri..."):
                all_words = " ".join(df_filtered['description'].dropna().astype(str))
                words = re.findall(r'\b\w+\b', all_words.lower())
                top20_words = [w for w, c in Counter(words).most_common(20)]

                if len(top20_words) > 0:
                    df_temp = df_filtered.copy()
                    for word in top20_words:
                        df_temp[word] = df_temp['description'].str.count(r'\b{}\b'.format(word))

                    top_varieties = df_temp['variety'].value_counts().head(10).index
                    df_top = df_temp[df_temp['variety'].isin(top_varieties)]

                    if len(df_top) > 0:
                        df_grouped = df_top.groupby('variety')[top20_words].mean()

                        fig6, ax6 = plt.subplots(figsize=(12, 6))
                        sns.heatmap(df_grouped, annot=True, fmt=".2f", cmap="coolwarm",
                                    linewidths=0.5, ax=ax6)
                        ax6.set_title("Corelarea top 20 cuvinte cu top 10 soiuri", fontsize=14)
                        ax6.set_xlabel("Cuvinte")
                        ax6.set_ylabel("Soiuri")
                        plt.tight_layout()
                        st.pyplot(fig6)
                    else:
                        st.warning("Nu există suficiente date pentru această vizualizare.")
                else:
                    st.warning("Nu s-au găsit cuvinte în descrieri.")

        with col2:
            st.caption(
                "Arată cum diferite cuvinte din descrieri sunt asociate cu diferite soiuri de vinuri. Ajută la înțelegerea caracteristicilor distinctive ale fiecărui soi.")
            if len(top20_words) > 0 and len(df_top) > 0:
                st.download_button(
                    "Descarcă PDF",
                    data=generate_pdf_matplotlib(fig6),
                    file_name="cuvinte_soiuri.pdf",
                    mime="application/pdf",
                    key="dl_words_varieties"
                )
    else:
        st.info("Coloanele 'description' sau 'variety' nu sunt disponibile.")

# ========== TAB 5: DISTRIBUȚII ȘI COMPARAȚII ==========
with tab5:
    st.header("Distribuții și comparații")

    # Viz 8: Scatter Preț vs Punctaj
    st.subheader("Reprezentarea grafică a relației preț-punctaj")

    col1, col2 = st.columns([3, 1])

    with col1:
        fig8, ax8 = plt.subplots(figsize=(10, 5))
        ax8.scatter(df_filtered['price'], df_filtered['points'], alpha=0.5, c='blue', s=30)
        ax8.set_xlabel('Price ($)', fontsize=12)
        ax8.set_ylabel('Points', fontsize=12)
        ax8.set_title('Scatter plot: Price vs Points', fontsize=14)
        ax8.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig8)

    with col2:
        st.caption(
            "Reprezentarea grafică a relației dintre preț și punctaj pentru fiecare vin. Permite identificarea vinurilor cu raport excepțional preț/calitate.")
        st.download_button(
            "Descarcă PDF",
            data=generate_pdf_matplotlib(fig8),
            file_name="scatter_pret_punctaj.pdf",
            mime="application/pdf",
            key="dl_scatter_price"
        )

    st.markdown("---")

    # Viz 9: Scatter Alcool vs Punctaj
    st.subheader("Analiza relației conținut alcoolic-calitate")

    if 'alcohol' in df_filtered.columns:
        col1, col2 = st.columns([3, 1])

        with col1:
            df_alc = df_filtered.dropna(subset=['alcohol', 'points'])

            if len(df_alc) > 0:
                fig9, ax9 = plt.subplots(figsize=(10, 5))
                ax9.scatter(df_alc['alcohol'], df_alc['points'], alpha=0.5, c='orange', s=30)
                ax9.set_xlabel('Alcohol (%)', fontsize=12)
                ax9.set_ylabel('Points', fontsize=12)
                ax9.set_title('Scatter plot: Alcohol vs Points', fontsize=14)
                ax9.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig9)
            else:
                st.warning("Nu există date cu alcool pentru această vizualizare.")

        with col2:
            st.caption(
                "Relația dintre conținutul de alcool și punctajul vinurilor. Ajută la înțelegerea impactului gradului alcoolic asupra calității percepute.")
            if len(df_alc) > 0:
                st.download_button(
                    "Descarcă PDF",
                    data=generate_pdf_matplotlib(fig9),
                    file_name="scatter_alcool_punctaj.pdf",
                    mime="application/pdf",
                    key="dl_scatter_alcohol"
                )
    else:
        st.info("Coloana 'alcohol' nu este disponibilă în dataset.")

    st.markdown("---")

    # Viz 10: Distribuția Punctajelor
    st.subheader("Analiza distribuției scorurilor de calitate")

    col1, col2 = st.columns([3, 1])

    with col1:
        fig10, ax10 = plt.subplots(figsize=(12, 6))
        ax10.hist(df_filtered["points"].dropna(), bins=20, edgecolor="black",
                  alpha=0.8, color='steelblue')
        ax10.set_title("Distribuția punctajelor vinurilor", fontsize=16)
        ax10.set_xlabel("Puncte (rating)", fontsize=13)
        ax10.set_ylabel("Frecvență", fontsize=13)
        ax10.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig10)

    with col2:
        st.caption(
            "Histograma punctajelor arată cum sunt distribuite evaluările în dataset. Majoritatea vinurilor tind să aibă punctaje în jurul valorilor medii.")
        st.download_button(
            "Descarcă PDF",
            data=generate_pdf_matplotlib(fig10),
            file_name="distributie_punctaje.pdf",
            mime="application/pdf",
            key="dl_dist_points"
        )

    st.markdown("---")

    # Viz 13: Scatter Interactiv cu Plotly
    st.subheader("Analiză interactivă a relației preț-calitate pe categorii")

    if 'category' in df_filtered.columns and 'price_quality_ratio' in df_filtered.columns:
        col1, col2 = st.columns([3, 1])

        with col1:
            df_plot = df_filtered.dropna(subset=['price', 'points', 'category', 'price_quality_ratio'])
            df_plot = df_plot[df_plot['price'] > 0]

            if len(df_plot) > 0:
                categories = df_plot['category'].unique()
                palette = sns.color_palette("Set1", n_colors=len(categories))
                palette_hex = [f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}' for r, g, b in palette]

                fig13 = px.scatter(
                    df_plot,
                    x='price',
                    y='points',
                    color='category',
                    hover_data=['country', 'region_1', 'variety', 'winery', 'vintage', 'price_quality_ratio'],
                    color_discrete_sequence=palette_hex,
                    labels={'price': 'Preț ($)', 'points': 'Punctaj'},
                    title='Relația dintre preț și punctajul vinurilor',
                    template='plotly_white',
                    height=700
                )

                fig13.update_yaxes(range=[df_plot['points'].min() - 1, df_plot['points'].max() + 1])
                fig13.update_layout(
                    legend_title_text='Categorie vin',
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True),
                    title=dict(font=dict(size=18))
                )

                st.plotly_chart(fig13, use_container_width=True)
            else:
                st.warning("Nu există date suficiente pentru această vizualizare interactivă.")

        with col2:
            st.caption(
                "Vizualizare interactivă care permite explorarea relației dintre preț și punctaj, cu detalii suplimentare la hover. Filtrează după categorii și explorează caracteristicile vinurilor.")
            if len(df_plot) > 0:
                st.info("Pentru a exporta graficul interactiv, folosește funcția de export din meniul graficului.")
    else:
        st.info("Coloanele necesare pentru vizualizarea interactivă nu sunt complete.")

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Aplicație de analiză a vinurilor | Dezvoltat cu Streamlit</p>
    </div>
""", unsafe_allow_html=True)
