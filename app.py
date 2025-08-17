import streamlit as st
import os
import requests
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
from datetime import datetime
import tempfile
from geopy.geocoders import Nominatim
import time

# Set page config
st.set_page_config(
    page_title="Satellite Data Viewer",
    page_icon="🛰️",
    layout="wide"
)


# Your existing functions (unchanged)
def closest_valid_hour(requested_hour):
    valid_hours = [0, 3, 6, 9, 12, 15, 18, 21]
    closest_hour = min(valid_hours, key=lambda x: abs(x - requested_hour))
    return closest_hour


def download_file(file_url, save_path):
    response = requests.get(file_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        return True
    else:
        raise Exception(f"Failed to download the file. HTTP Status code: {response.status_code}")


def rbtop3():
    newcmp = mcolors.LinearSegmentedColormap.from_list("", [
        (0 / 140, "#000000"),
        (60 / 140, "#fffdfd"),
        (60 / 140, "#05fcfe"),
        (70 / 140, "#010071"),
        (80 / 140, "#00fe24"),
        (90 / 140, "#fbff2d"),
        (100 / 140, "#fd1917"),
        (110 / 140, "#000300"),
        (120 / 140, "#e1e4e5"),
        (120 / 140, "#eb6fc0"),
        (130 / 140, "#9b1f94"),
        (140 / 140, "#330f2f")
    ])
    return newcmp.reversed(), 40, -100


def handle_and_plot_irwin_cdr(nc_file, center_coord, date_time):
    center_lon, center_lat = center_coord

    if abs(center_lon - 180) <= 16.01:
        projection = ccrs.PlateCarree(central_longitude=180)
    else:
        projection = ccrs.PlateCarree()

    irwin_cdr = nc_file.variables['irwin_cdr']
    irwin_vza_adj = nc_file.variables['irwin_vza_adj']

    raw_irwin_cdr = irwin_cdr[0, :, :]
    vza_adjustment = irwin_vza_adj[0, :, :]
    lats = nc_file.variables['lat'][:]
    lons = nc_file.variables['lon'][:]

    original_temperature = raw_irwin_cdr - vza_adjustment

    fill_value = getattr(irwin_cdr, '_FillValue', None)
    scale_factor = getattr(irwin_cdr, 'scale_factor', 1)
    add_offset = getattr(irwin_cdr, 'add_offset', 0)

    if fill_value is not None:
        original_temperature = np.where(original_temperature == fill_value, np.nan, original_temperature)

    processed_data = original_temperature * scale_factor + add_offset

    processed_data_celsius = processed_data - 273.15
    processed_data_celsius = np.ma.masked_invalid(processed_data_celsius)

    smoothed_data = gaussian_filter(processed_data_celsius, sigma=1)

    cmap, vmax, vmin = rbtop3()

    fig, ax = plt.subplots(figsize=(18, 10), subplot_kw={'projection': projection})

    # Set the extent to zoom in to a 24x24 degree square around the center coordinate
    ax.set_extent([center_lon - 12, center_lon + 12, center_lat - 12, center_lat + 12], crs=ccrs.PlateCarree())

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')

    im = ax.pcolormesh(lons, lats, smoothed_data, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, orientation='vertical', label='Temperature (°C)')

    dt = datetime.strptime(date_time, '%Y%m%d%H')
    title = dt.strftime('GRIDSAT B1 Imagery for %B %d, %Y at %H:00 UTC')
    plt.title(title, fontsize=18, weight='bold', pad=10)

    fig.text(0.5, 0.085, 'Plotted by Sekai Chandra (@Sekai_WX)', ha='center', fontsize=15, weight='bold')

    return fig


def get_coordinates(location_str):
    """Geocode location string to lat/lon coordinates"""
    try:
        geolocator = Nominatim(user_agent="satellite_viewer")
        time.sleep(1)  # Rate limiting
        location = geolocator.geocode(location_str, timeout=10)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        st.error(f"Geocoding error: {str(e)}")
        return None, None


def process_gridsat_data(year, month, day, hour, center_coord):
    DATASET_START_DATE = datetime(1980, 1, 1, 0, 0)
    DATASET_END_DATE = datetime(2024, 3, 31, 21, 0)

    month_str = f"{int(month):02}"
    day_str = f"{int(day):02}"
    hour_int = int(hour)

    requested_hour = hour_int
    valid_hours = [0, 3, 6, 9, 12, 15, 18, 21]
    if requested_hour not in valid_hours:
        closest_hour = closest_valid_hour(requested_hour)
        raise Exception(
            f"The satellite dataset is only available every three hours! The closest valid hour is {closest_hour:02d}.")

    current_date = datetime(year, int(month), int(day), int(requested_hour))

    if current_date < DATASET_START_DATE or current_date > DATASET_END_DATE:
        raise Exception(
            "The requested date is out of this dataset's period of coverage (1980-01-01 to 2024-03-31)!")

    file_name = f"GRIDSAT-B1.{year}.{month_str}.{day_str}.{requested_hour:02d}.v02r01.nc"
    url = f"https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/access/{year}/{file_name}"

    # Use temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
        save_path = tmp_file.name

    try:
        download_file(url, save_path)
        date_time = f"{year}{month_str}{day_str}{requested_hour:02d}"

        with Dataset(save_path, mode='r') as nc_file:
            fig = handle_and_plot_irwin_cdr(nc_file, center_coord, date_time)
            return fig
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


# Streamlit UI
st.title("🛰️ Satellite Data Viewer")
st.markdown("### GRIDSAT-B1 Infrared Satellite Imagery")

st.markdown("""
This tool allows you to view historical satellite imagery from the GRIDSAT-B1 dataset. 
The data covers **January 1980 to March 2024** and is available every 3 hours.
""")

# Create columns for better layout
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📅 Select Date & Time")

    # Date input
    date_input = st.date_input(
        "Date",
        value=datetime(2024, 1, 1),
        min_value=datetime(1980, 1, 1),
        max_value=datetime(2024, 3, 31)
    )

    # Hour selection (only valid hours)
    hour_input = st.selectbox(
        "Hour (UTC)",
        options=[0, 3, 6, 9, 12, 15, 18, 21],
        format_func=lambda x: f"{x:02d}:00"
    )

    st.subheader("🌍 Location")

    # Location input method
    location_method = st.radio(
        "How would you like to specify the location?",
        ["City/Place Name", "Coordinates (Lat, Lon)"]
    )

    if location_method == "City/Place Name":
        location_input = st.text_input(
            "Enter city or place name",
            placeholder="e.g., New York, London, Tokyo"
        )
        lat, lon = None, None
        if location_input:
            with st.spinner("Geocoding location..."):
                lat, lon = get_coordinates(location_input)
                if lat and lon:
                    st.success(f"📍 Found: {lat:.4f}°, {lon:.4f}°")
                else:
                    st.error("Location not found. Please try a different name or use coordinates.")
    else:
        col_lat, col_lon = st.columns(2)
        with col_lat:
            lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=40.7589, step=0.1)
        with col_lon:
            lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-73.9851, step=0.1)

    # Generate button
    generate_button = st.button("🚀 Generate Satellite Image", type="primary")

with col2:
    st.subheader("📊 Satellite Image")

    if generate_button:
        if lat is not None and lon is not None:
            try:
                with st.spinner("Downloading and processing satellite data... This may take a moment."):
                    fig = process_gridsat_data(
                        date_input.year,
                        date_input.month,
                        date_input.day,
                        hour_input,
                        (lon, lat)
                    )

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)  # Clean up memory

                st.success("✅ Image generated successfully!")
                st.info("💡 Right-click on the image to save it to your device.")

            except Exception as e:
                st.error(f"❌ Error generating image: {str(e)}")
        else:
            st.warning("⚠️ Please provide a valid location.")

# Information section
with st.expander("ℹ️ About this data"):
    st.markdown("""
    **GRIDSAT-B1** is a satellite dataset that provides global infrared brightness temperature data.

    - **Coverage**: January 1980 to March 2024
    - **Temporal Resolution**: Every 3 hours (00, 03, 06, 09, 12, 15, 18, 21 UTC)
    - **Spatial Resolution**: ~4km at nadir
    - **Data Source**: NOAA/NCEI
    - **Color Scale**: Temperature in degrees Celsius

    The images show cloud-top temperatures and surface temperatures where clouds are absent. 
    Colder temperatures (blues/purples) typically indicate higher cloud tops, while warmer 
    temperatures (reds/yellows) indicate lower clouds or surface features.
    """)

st.markdown("---")
st.markdown("*Created by Sekai Chandra (@Sekai_WX)*")