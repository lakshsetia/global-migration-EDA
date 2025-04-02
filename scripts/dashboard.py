import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.set_page_config(
    page_title="Global Migration Trends",
    page_icon="🌍",
    layout="wide"
)

@st.cache_data
def load_file(filepath, sheetname=0):
    return pd.read_excel(filepath, sheet_name=sheetname, dtype={"2024":str})

page = st.sidebar.radio("Go to", ["Home", "Geographic Visualization", "Detailed Analysis", "Data Sources"])

if page == "Home":
    html_title = """
        <style>
            .title-text {
                font-weight:bold;
                padding=5px;
                border-radius:6px; 
            }
        </style>
        <center><h1 class="title-test">Global Migration Trends Analysis and Visualization</h1></center>
    """
    st.markdown(html_title, unsafe_allow_html=True)
    html_intro = """
        <style>
            .intro-text {
                font-size:1.25rem; !important
                line-height:2rem; !important
                margin-bottom:2rem; !important
            }
        </style>
        <br>
        <div class="intro-text">Our platform consolidates and transforms complex migration data into actionable intelligence, bridging critical gaps with precision. By integrating real-time insights from governments, NGOs, and climate monitors, we empower predictive modeling to address migration trends before crises emerge. Explore interactive dashboards that reveal hidden patterns, enabling agencies to strategically position aid and make informed decisions. Experience the future of migration analysis—driven by data, powered by impact.</div>
        <br>
        <div class="intro-text">This goes beyond mere data visualization—it's a transformative approach to decision-making that eliminates guesswork, delivering precise insights to ensure resources are allocated effectively and policies achieve real results.</div>
        <br>
    """
    st.markdown(html_intro, unsafe_allow_html=True)
    image_file = os.path.join("data", "global-migration.jpg")
    image = Image.open(image_file)
    col26, col27, col28 = st.columns([1, 4, 1])
    with col27:
        st.image(image, width=700, use_container_width="auto")

elif page == "Geographic Visualization":
    html_title = """
        <style>
            .title-test {
                font-weight:bold;
                padding=5px;
                border-radius:6px; 
            }
        </style>
        <center><h1 class="title-test">Global Migration Trends Analysis and Visualization</h1></center>
    """
    ### Choropleth map ###
    # Internatinal migration based on country of origin
    st.subheader("Number of international migrations by country of origin")
    col1, col2 = st.columns([0.2, 0.8])
    year1 = 2024
    with col1:
        year1 = st.selectbox("Enter year: ", (2024, 2020, 2015, 2010, 2005, 2000, 1995, 1990), key="origin_year")
    with col2:
        filepath = os.path.join("data", "international-migrant-stock", "country_of_origin_migration.xlsx")
        try:
            df_origin = load_file(filepath)
        except Exception as e:
            print(f"Error while reading {filepath}: {e}")
        origin_map = px.choropleth(df_origin, 
            locations="Country", 
            locationmode="country names", 
            color=year1, 
            hover_name="Country", 
            hover_data=["Continent", st.session_state["origin_year"]], 
            color_continuous_scale="Turbo", 
            title="International Migration by origin",
            range_color=[0, 20000000],
        )
        origin_map.update_layout(
            margin={"r": 0, "t": 30, "l": 0, "b":0},
            coloraxis_colorbar={'title': "Migration Population"}
        )
        st.plotly_chart(origin_map, use_container_width=True)

    st.divider()

    # International migration based on country of destination
    st.subheader("Number of international migrations by country of destination")
    col3, col4 = st.columns([0.2, 0.8])
    year2 = 2024
    with col3:
        year2 = st.selectbox("Enter year: ", (2024, 2020, 2015, 2010, 2005, 2000, 1995, 1990), key="destination_year")
    with col4:
        filepath = os.path.join("data", "international-migrant-stock", "country_of_destination_migration.xlsx")
        try:
            df_destination = load_file(filepath)
        except Exception as e:
            print(f"Error while reading {filepath}: {e}")

        destination_map = px.choropleth(
            df_destination,
            locations="Country",
            locationmode="country names",
            color=year2,
            hover_name="Country",
            hover_data=["Continent", year2],
            color_continuous_scale="Turbo",
            title="International Migration by destination",
            range_color=[0, 50000000]
        )
        destination_map.update_layout(
            margin={"r":0, "t":30, "l": 0, "b": 0},
            coloraxis_colorbar={'title': "Migration Population"}
        )
        st.plotly_chart(destination_map, use_container_width=True)

    st.divider()


    ### Snakey Diagram ###
    st.subheader("Number of migrations from origin to destination continents")
    col20, col21, col22 = st.columns([0.2, 0.6, 0.2])
    with col21:
        filepath = os.path.join("data", "international-migrant-stock", "total.xlsx")
        try:
            df_total = load_file(filepath)
        except Exception as e:
            print(f"Error while reading {filepath}: {e}")
        df_total = df_total.melt(
            id_vars="region",
            var_name="source",
            value_name="migrants"
        )
        df_total.columns = ["target", "source", "migrants"]
        unique_regions = list(pd.unique(df_total[["source", "target"]].values.ravel("K")))
        map_source = {k: v for v,k in enumerate(unique_regions)}
        map_target = {k: v+6 for v,k in enumerate(unique_regions)}
        df_total["source"] = df_total["source"].map(map_source)
        df_total["target"] = df_total["target"].map(map_target)
        node_colors = [
            "#FFA500",  # Africa (origin)
            "#4682B4",  # Asia
            "#32CD32",  # Europe
            "#FFD700",  # Oceania
            "#FF6347",  # North America
            "#9370DB",  # Latin America and Caribbean
            "#FFA500",  # Africa (destination)
            "#4682B4",  # Asia
            "#32CD32",  # Europe
            "#FFD700",  # Oceania
            "#FF6347",  # North America
            "#9370DB"   # Latin America and Caribbean
        ]
        df_total['color'] = [node_colors[i] for i in df_total['source']]
        link = dict(source=df_total["source"], target=df_total["target"], value=df_total["migrants"], hoverinfo="all", color=df_total['color'])
        node = dict(label=unique_regions+unique_regions, pad=40, thickness=25, line=dict(color="black", width=0.4), color=node_colors)
        sankey = go.Sankey(link=link, node=node)
        diagram = go.Figure(sankey)
        diagram.update_layout(
            hovermode="x",
            title="Migration from origin to destination",
            title_font_color="black",
            title_font_size=20,
            font=dict(size=10, color="black"),
            paper_bgcolor="#FFFFFF"
        )
        st.plotly_chart(diagram, use_container_width=True)

elif page == "Detailed Analysis":
    ### Distribution ###
    # Age distribution in international migrants
    st.subheader("Distribution of international migrants by age groups")
    filepath = os.path.join("data", "international-migrant-stock", "age_distribution.xlsx")
    col9, col10 = st.columns([0.2, 0.8])
    with col9:
        region_sheet = {
            "World": 0,
            "Asia": 1,
            "Africa": 2,
            "Latin America and the Caribbean": 3,
            "North America": 4,
            "Oceania": 5,
            "Europe": 6
        }
        region = st.selectbox("Enter region: ", ("World", "North America", "Europe", "Asia", "Africa", "Oceania", "Latin America and the Caribbean"), key="age_distribution")
        year3 = st.selectbox("Enter year: ", (2020, 2015, 2010, 2005, 2000, 1995, 1990))
        try:
            df_ageDistribution = load_file(filepath, region_sheet[region])
        except Exception as e:
            print(f"Error while reading {filepath}: {e}")
        df_ageDistribution = df_ageDistribution.T
        df_ageDistribution.columns = df_ageDistribution.iloc[0]
        df_ageDistribution.drop(index="year", inplace=True)
        df_ageDistribution.reset_index(inplace=True)

    with col10:
        age_distribution = go.Figure()
        age_distribution.add_trace(go.Bar(x=df_ageDistribution['index'][:16], y=df_ageDistribution[year3][:16], name="International migrants distribution by age", hoverinfo="y"))
        age_distribution.add_trace(go.Scattergl(x=df_ageDistribution['index'][:16], y=df_ageDistribution[year3][16:], name="Percentage distribution of international migrants", yaxis="y2",  hoverinfo="y"))
        age_distribution.update_layout(
            title="International migrants distribution by age",
            xaxis=dict(title="Age Groups"),
            yaxis=dict(title="Number of international migrants", showgrid=False),
            yaxis2=dict(title="Percentage distribution", overlaying="y", side="right"),
            template="gridon",
            legend=dict(x=1, y=1.1)
        )
        st.plotly_chart(age_distribution, use_container_width=True)

    st.divider()


    ### Bar and line chart ###
    # world migrants and percentage of population
    st.subheader("International migrants and international migrants as a percentage of population")
    filepath = os.path.join("data", "international-migrant-stock", "migrants_percentage.xlsx")
    col7, col8 = st.columns([0.2, 0.8])
    with col7:
        region_sheet = {
            "World": 0,
            "Asia": 1,
            "Africa": 2,
            "Latin America and the Caribbean": 3,
            "North America": 4,
            "Oceania": 5,
            "Europe": 6
        }
        region = st.selectbox("Enter region: ", ("World", "North America", "Europe", "Asia", "Africa", "Oceania", "Latin America and the Caribbean"), key="migrants_percentage")
        try:
            df_migrantsPercentage = load_file(filepath, region_sheet[region])
        except Exception as e:
            print(f"Error while reading {filepath}: {e}")
    with col8:
        migrantsPercentage_plot = go.Figure()
        migrantsPercentage_plot.add_trace(go.Bar(x=df_migrantsPercentage["year"], y=df_migrantsPercentage["migrants"], name="Number of international migrants")) 
        migrantsPercentage_plot.add_trace(go.Scatter(x=df_migrantsPercentage["year"], y=df_migrantsPercentage["percentage"], yaxis="y2", name="Migrants as a percentage of total population", mode="lines"))
        migrantsPercentage_plot.update_layout(
            title="Number of international migrants and as a percentage of total population",
            xaxis=dict(title="Year"),
            yaxis=dict(title="Number of international migrants", showgrid=False),
            yaxis2=dict(title="Percentage of total population", overlaying="y", side="right"),
            template="gridon",
            legend=dict(x=1, y=1.1)
        )
        st.plotly_chart(migrantsPercentage_plot, use_container_width=True)

    st.divider()


    ### Line Chart ###
    # Annual growth rate of top countries
    st.subheader("Annual growth rate in the 9 leading countries (1990-2025)")
    col23, col24, col25 = st.columns([0.15, 0.7, 0.15])
    with col24:
        fig, ax = plt.subplots(figsize=(12, 8))
        grow, ax = plt.subplots()
        data = {
            'Year': [1990, 1995, 2000, 2005, 2010, 2015, 2020,2025],
            'China': [3.3,3.3,3.4,3.4,3.4,3.4,3.7,3.9],
            'India': [-1.1,-1.3,-1.5,-1.0,-1.1,-1.1,-1.2,-1.1],
            'UK': [2.6,2.6,4.6,5.1,3.2,3.0,3.0,3.0],
            'France': [0.2,0.7,2.2,1.4,1.7,1.5,1.6,1.4],
            'Germany': [6.5,1.5,1.5,0.8,1.6,3.4,2.7,1.6],
            'USA': [4.1,4.0,2.6,2.1,1.7,1.0,0.9,1.0],
            'Canada': [2.7,2.6,1.9,2.9,2.7,0.7,1.4,1.4],
            'Australia': [1.1,0.8,2.1,3.7,2.7,2.4,1.6,2.0],
            'New Zealand': [2.4,2.8,4.4,2.3,3.4,3.4,2.2,2.4]
        }
        df = pd.DataFrame(data)
        df_plot = df.set_index('Year')
        for column in df_plot.columns:
            ax.plot(df_plot.index, df_plot[column], marker='o', linewidth=2, label=column)
        ax.set_title('Annual Growth Rate of Country (1990-2025)', fontsize=16)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Annual Growth Rate (%)', fontsize=14)
        ax.set_xticks(data['Year'])
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Countries', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(grow, use_container_width=True)

    st.divider()
    

    ### Heatmap ###
    # Annual growth rate
    st.subheader("Annual Growth Rate of migrants based on Continents (1990-2025)")
    filepath = os.path.join("data", "international-migrant-stock", "growth_rate.xlsx")
    col11, col12, col13 = st.columns([0.15, 0.7, 0.15])
    with col12:
        try:
            df_growth = load_file(filepath)
        except Exception as e:
            print(f"Error while reading {filepath}: {e}")
        regions = ["World", "Africa", "Asia", "Europe", "Oceania", "North America", "Latin America and Caribbean"]
        df_continents = df_growth[df_growth['region'].isin(regions)]
        df_continents = df_continents.melt(
            id_vars="region",
            var_name="Year",
            value_name="Growth Rate"
        )
        df_continents = df_continents.pivot(
            index="region",
            columns="Year",
            values="Growth Rate"
        )
        plt.figure(figsize=(12, 7)) 
        fig, ax = plt.subplots()
        sns.heatmap(df_continents, fmt=".2f", annot=True, cmap=sns.cubehelix_palette(as_cmap=True), center=0, ax=ax)
        plt.title("Annual Growth Rate of migrants based on Continents (%)", pad=20)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.divider()

    ### Bar graph ###
    # Net Migration 
    st.subheader("Net number of migrants based on continents")
    filepath = os.path.join("data", "international-migrant-stock", "migrants.xlsx")
    try:
        df_migrants = load_file(filepath)
    except Exception as e:
        print(f"Error while reading {filepath}: {e}")
    col5, col6, col = st.columns([0.15, 0.7, 0.15])
    with col6:
        migrants_plot = go.Figure()
        migrants_plot.add_trace(go.Bar(x=df_migrants["Continent"], y=df_migrants["Net number of migrants"]*1000, name="Net number of migrants", hoverinfo="y"))
        migrants_plot.update_layout(
            title="Net number of migrants by continents",
            autosize=True,
            width=600,
            height=500,
            xaxis=dict(title="Continents"),
            yaxis=dict(title="Net number of migrants", showgrid=False),
            template="gridon",
            legend=dict(x=1, y=1.1)
        )
        st.plotly_chart(migrants_plot, use_container_width=True)

    st.divider()

    ### Pie Chart ###
    # Indicators of international migration
    st.subheader("Factors for migration")
    fig = px.pie(names=['Job Opportunity','Education','War/Refugee','Seeking Asylum','Natural Disasters','Environmental Changes', 'Other'], values=[169,6.1,32.5,21.2,32.6,22.3, 10])
    st.plotly_chart(fig, use_container_width=True)


elif page == "Data Sources":
    st.subheader("References")
    st.link_button("United Nations International migrant stock", url="https://www.un.org/development/desa/pd/content/international-migrant-stock")
    st.link_button("World Population Prospects", url="https://population.un.org/wpp/")
    st.link_button("World's human migration patterns in 2000–2019", url="https://www.nature.com/articles/s41562-023-01689-4/metrics")
    st.link_button("Our World in data migration", url="https://ourworldindata.org/migration")
    st.link_button("Migration Data Portal", url="https://www.migrationdataportal.org/")