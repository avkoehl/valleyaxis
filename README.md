# Valley Axis

Implementation of glacier centerline extraction algorithm from [Kienholz et al. (2014)](https://tc.copernicus.org/articles/8/503/2014/).

![Example](./img/example.png)

## TODO

- [ ] Remove any detected inlets points that are not on the valley floor
- [ ] Add option to remove inlets points that are not near any boundary

## Installation

### Prerequisites

- Python 3.10 or higher
- [Poetry (package manager)](https://python-poetry.org/)

### Installing from Github

1. Clone the repository
```bash
git clone git@github.com:avkoehl/valleyaxis.git
cd valleyaxis
```

2. Install dependencies using Poetry
```bash
poetry install
```

## Input Data Requirements

- **DEM**: Digital elevation model 
- **Flowlines**: GeoDataFrame of linestrings 
- **Valley Floor**: Polygon of valley floor extent

## Usage

Here's a complete example showing how to extract valley centerlines:

```python
import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray as rxr

# Load input data
dem = rxr.open_rasterio("./sample_data/1805000202-dem.tif", masked=True).squeeze()
flowlines = gpd.read_file("./sample_data/1805000202-flowlines.shp")
floor = gpd.read_file("./sample_data/floor.shp").geometry[0].buffer(0.01) 
centerlines = valley_centerlines(dem, flowlines, floor, maxarea=0)
```

## References

Kienholz, C., Rich, J. L., Arendt, A. A., and Hock, R.: A new method for deriving glacier centerlines applied to glaciers in Alaska and northwest Canada, The Cryosphere, 8, 503–519, https://doi.org/10.5194/tc-8-503-2014, 2014.
