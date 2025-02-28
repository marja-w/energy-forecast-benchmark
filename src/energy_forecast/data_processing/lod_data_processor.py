import glob
import xml.etree.ElementTree as ET
import csv
import numpy as np
from shapely.geometry import Polygon
import polars as pl

from src.energy_forecast.config import DATA_DIR, RAW_DATA_DIR


def polygon_area_3d(coords):
    """
    Compute the area of a polygon in 3D space using Newell's method.

    :param coords: List of (x, y, z) tuples representing the polygon's vertices.
                   The first and last coordinates must be the same.
    :return: Absolute area of the 3D polygon.
    """
    n = len(coords) - 1  # The last point is the same as the first
    normal = np.array([0.0, 0.0, 0.0])  # Initialize normal vector

    for i in range(n):
        x1, y1, z1 = coords[i]
        x2, y2, z2 = coords[i + 1]
        normal[0] += (y1 - y2) * (z1 + z2)
        normal[1] += (z1 - z2) * (x1 + x2)
        normal[2] += (x1 - x2) * (y1 + y2)

    # Compute normal vector magnitude
    normal_magnitude = np.linalg.norm(normal)

    # Projected 2D area is half the magnitude of the normal vector
    return abs(normal_magnitude) / 2


# Define file paths
xml_file = DATA_DIR / "lod2" / 'LoD2_32_565_5952_1_SH.xml'
xml_file_folder = DATA_DIR / "lod2"
csv_file = DATA_DIR / "lod2" / 'building_data.csv'

# Define the namespace mappings
namespaces = {
    'core': 'http://www.opengis.net/citygml/1.0',
    'bldg': 'http://www.opengis.net/citygml/building/1.0',
    'gml': 'http://www.opengis.net/gml',
    'xal': 'urn:oasis:names:tc:ciq:xsdschema:xAL:2.0'
}

file_list = glob.glob(f"{xml_file_folder}/*.xml")
# Open CSV file for writing
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(
            ["Building ID", "Creation Date", "Country", "Address", "postal_code", "Function", "Height (m)", "Storeys Above Ground", "ground_surface"])
    for xml_file in file_list:
        print(f" Processing {xml_file}")
        # Parse the XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Iterate through each building in the XML
        for building in root.findall(".//bldg:Building", namespaces):
            building_id = building.attrib.get('{http://www.opengis.net/gml}id', 'N/A')

            # Extract building function
            function = building.find("bldg:function", namespaces)
            function_text = function.text if function is not None else 'N/A'

            # Extract height
            height = building.find("bldg:measuredHeight", namespaces)
            height_value = height.text if height is not None else 'N/A'

            # Extract number of storeys
            storeys = building.find("bldg:storeysAboveGround", namespaces)
            storeys_value = storeys.text if storeys is not None else 'N/A'

            # Extract creation date
            creation_date = building.find('core:creationDate', namespaces).text if building.find('core:creationDate', namespaces) is not None else None

            # Extract address
            country_element = building.find(
                "bldg:address/core:Address/core:xalAddress/xal:AddressDetails/xal:Country/xal:CountryName",
                namespaces)
            country = country_element.text if country_element is not None else 'N/A'

            address_elem = building.find(
                "bldg:address/core:Address/core:xalAddress/xal:AddressDetails/xal:Country/xal:Locality/xal:Thoroughfare/xal:ThoroughfareName",
                namespaces)
            number_elem = building.find(
                "bldg:address/core:Address/core:xalAddress/xal:AddressDetails/xal:Country/xal:Locality/xal:Thoroughfare/xal:ThoroughfareNumber",
                namespaces)
            address = f"{address_elem.text} {number_elem.text}" if address_elem is not None else None
            if address is None:
                continue

            # Extract postal code
            postal_code_elem = building.find(
                "bldg:address/core:Address/core:xalAddress/xal:AddressDetails/xal:Country/xal:Locality/xal:PostalCode/xal:PostalCodeNumber",
                namespaces)
            postal_code = postal_code_elem.text if postal_code_elem is not None else 'N/A'

            # extract ground surface
            coordinates_elem = building.find("bldg:boundedBy/bldg:GroundSurface/bldg:lod2MultiSurface/gml:MultiSurface/gml:surfaceMember/gml:Polygon/gml:exterior/gml:LinearRing/gml:posList", namespaces)
            # coordinate_list = coordinates_elem.text.replace("\n", "").split() if coordinates_elem is not None else None
            # formatted_list = [tuple(map(float, coordinate_list[i:i+3])) for i in range(0, len(coordinate_list), 3)]
            # ground_surface = polygon_area_3d(formatted_list)
            pos_list_text = coordinates_elem.text.strip()
            coords_flat = list(map(float, pos_list_text.split()))
            coords = [(coords_flat[i], coords_flat[i + 1], coords_flat[i + 2]) for i in range(0, len(coords_flat), 3)]
            polygon = Polygon(coords)
            ground_surface = polygon.area

            # Write to CSV
            writer.writerow([building_id, creation_date, country, address, postal_code, function_text, height_value, storeys_value, ground_surface])

print(f"CSV file saved at {csv_file}")

dh_meta = pl.read_csv(RAW_DATA_DIR / "district_heating_meta.csv").with_columns(
    pl.lit("Mehrfamilienhaus").alias("typ"),
    pl.lit(75).alias("min_vorlauf_temp"),
    pl.lit(90).alias("max_vorlauf_temp")
).rename({"eco_u_id": "id"}).select(
    ["id", "city", "address", "postal_code", "typ", "min_vorlauf_temp", "max_vorlauf_temp"])
dh_meta = dh_meta.rename({"address": "adresse", "postal_code": "plz", "city": "ort"}).select(
    ["id", "adresse", "ort", "plz", "typ", "min_vorlauf_temp", "max_vorlauf_temp"]).cast(
    {"plz": pl.String, "min_vorlauf_temp": pl.String, "max_vorlauf_temp": pl.String}).with_columns(
    pl.lit("dh").alias("source"),
    pl.lit("district heating").alias("primary_energy")
)
df_buildings = pl.read_csv(DATA_DIR / "lod2" / "building_data.csv", null_values="N/A").drop_nulls().rename({"Address": "adresse"})
dh_meta_lod = dh_meta.join(df_buildings, on="adresse", how="left").unique(["adresse"]).sort(pl.col("adresse"))
dh_meta_lod.write_csv(RAW_DATA_DIR / "dh_meta_lod.csv")
