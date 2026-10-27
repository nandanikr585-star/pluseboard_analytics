
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="PulseBoard", layout="wide")
st.title("PulseBoard: Life Expectancy Dashboard")
st.write("Explore life expectancy and health indicators.")

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/processed/cleaned_life_expectancy.csv"
    )

df = load_data()
df.columns = [
    c.strip().replace(" ", "_").replace("-", "_")
    for c in df.columns
]

country = "Country"
year = "Year"
life = "Life_expectancy"
status = "Status"

# Match actual column names without case sensitivity
lookup = {c.lower(): c for c in df.columns}
country = lookup.get("country", country)
year = lookup.get("year", year)
life = lookup.get("life_expectancy", life)
status = lookup.get("status", status)

if country not in df.columns or year not in df.columns or life not in df.columns:
    st.error("Required columns missing. Available columns:")
    st.write(df.columns.tolist())
    st.stop()

df[year] = pd.to_numeric(df[year], errors="coerce")
df[life] = pd.to_numeric(df[life], errors="coerce")
df = df.dropna(subset=[country, year, life])

st.sidebar.header("Dashboard Filters")
countries = sorted(df[country].dropna().unique())
selected = st.sidebar.multiselect(
    "Select countries", countries,
    default=countries[:min(5, len(countries))]
)

yr_min, yr_max = int(df[year].min()), int(df[year].max())
years = st.sidebar.slider(
    "Select year range", yr_min, yr_max, (yr_min, yr_max)
)

filtered = df[
    df[country].isin(selected) &
    df[year].between(years[0], years[1])
]

if status in df.columns:
    options = sorted(df[status].dropna().unique())
    chosen_status = st.sidebar.multiselect(
        "Development status", options, default=options
    )
    filtered = filtered[filtered[status].isin(chosen_status)]

if filtered.empty:
    st.warning("No data. Change the filters.")
    st.stop()

latest = int(filtered[year].max())
latest_df = filtered[filtered[year] == latest]
top_country = latest_df.groupby(country)[life].mean().idxmax()

a, b, c = st.columns(3)
a.metric("Average Life Expectancy", f"{filtered[life].mean():.2f} years")
b.metric("Top Country (selected data)", top_country)
c.metric("Latest Year", latest)

st.subheader("1. Life Expectancy Trend")
trend = filtered.groupby(year, as_index=False)[life].mean()
st.plotly_chart(
    px.line(trend, x=year, y=life, markers=True),
    use_container_width=True
)

st.subheader("2. Country Comparison")
comparison = filtered.groupby(country, as_index=False)[life].mean()
st.plotly_chart(
    px.bar(comparison, x=country, y=life),
    use_container_width=True
)

st.subheader("3. Health Indicator Relationship")
numeric = [
    col for col in filtered.select_dtypes("number").columns
    if col not in [year, life]
]
if numeric:
    indicator = st.selectbox("Choose indicator", numeric)
    st.plotly_chart(
        px.scatter(
            filtered, x=indicator, y=life,
            color=status if status in filtered.columns else None,
            hover_name=country
        ),
        use_container_width=True
    )
else:
    st.info("No additional numeric indicators available.")

st.subheader("4. Correlation Heatmap")
numbers = filtered.select_dtypes("number")
if numbers.shape[1] >= 2:
    st.plotly_chart(
        px.imshow(numbers.corr(), text_auto=".2f",
                  color_continuous_scale="RdBu_r"),
        use_container_width=True
    )
else:
    st.info("Not enough numeric columns for a heatmap.")

st.header("Key Insights")
st.write(f"- Average life expectancy: {filtered[life].mean():.2f} years.")
st.write(f"- Highest average life expectancy in year {latest}: {top_country}.")
st.write(f"- Selected countries: {filtered[country].nunique()}.")

if numeric:
    corr = filtered[numeric + [life]].corr()[life].drop(life).dropna()
    if not corr.empty:
        strongest = corr.abs().idxmax()
        st.write(
            f"- Strongest absolute correlation with life expectancy: "
            f"{strongest} ({corr[strongest]:.2f}). "
            "Correlation does not prove causation."
        )
