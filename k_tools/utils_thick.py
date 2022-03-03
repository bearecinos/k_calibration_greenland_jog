import os
import logging
import numpy as np
import rasterio
import salem
import pickle
import pyproj
from affine import Affine
from salem import wgs84
import xarray as xr
import geopandas as gpd
from oggm import utils
from k_tools import misc
from IPython import embed

# Module logger
log = logging.getLogger(__name__)

def open_thick_raster(tiff_path):
    """
    Opens a tiff file from Greenland thickness observations
    and puts it in a oggm ready format (right corner etc).
    Returns a xarray.Dataset for easy processing
    :param
        tiff_path: path to the data
    :return
        ds: xarray object with data already scaled
    """

    # Processing raster data
    src = rasterio.open(tiff_path)

    # Retrieve the affine transformation
    if isinstance(src.transform, Affine):
        transform = src.transform
    else:
        transform = src.affine

    dy = transform.e
    ds = salem.open_xr_dataset(tiff_path)
    data = ds.data.values

    # Read the image data, flip upside down if necessary
    data_in = data
    if dy < 0:
        data_in = np.flip(data_in, 0)

    # save the correct direction for the data.
    ds.data.values = np.flip(data_in, 0)

    return ds


def crop_thick_data_to_glacier_grid(gdir, thick_f, error_f):
    """
    Crop thickness data and uncertainty to the glacier grid
    :param
        gdir: Glacier Directory
        thick_f: xarray data containing thickness
        error_f: xarray data containing the errors
    :return
        ds_array: an array of thickness cropped to the glacier grid
        dr_array: an array of thickness errors cropped to the glacier grid
    """

    # Crop to glacier grid
    ds_glacier = thick_f.salem.subset(grid=gdir.grid, margin=2)
    dr_glacier = error_f.salem.subset(grid=gdir.grid, margin=2)

    return ds_glacier, dr_glacier

def crop_thick_data_to_flowline(thick, error, shp):
    """
    Crop thickness data and uncertainty to the glacier flowlines
    :param
        thick: xarray data containing thickness
        error: xarray data containing the errors
        shp: Shape file containing the glacier flowlines
    :return
        ds_array: an array of thickness croped to the glacier main flowline .
        dr_array: an array of thickness erros croped to the glacier
                  main flowline.
    """

    # Crop to flowline
    ds_fls = thick.salem.roi(shape=shp.iloc[[0]])
    dr_fls = error.salem.roi(shape=shp.iloc[[0]])
    return ds_fls, dr_fls

def crop_thick_data_to_flowline_width(thick, error, shp, last_one_third=False):
    """
    Crop thickness and uncertainty to the glacier flowlines
    :param
        thick: xarray data containing thickness
        error: xarray data containing the errors
        shp: Shape file containing the glacier flowlines catchment widths
    :return
        ds_width: an array of thickness croped to the glacier catchment widths
            of the main flowline .
        dr_width: an array of thickness erros
            croped to the glacier catchment widths of the main flowline .
    """

    # Crop to main flowline and catchment
    shp_main = shp.loc[shp.MAIN == 1]

    if last_one_third is True:
        shp_main_end = shp_main.iloc[-np.int(len(shp_main) / 3):]
        shp_main = shp_main_end

    ds_width = thick.salem.roi(shape=shp_main)
    dr_width = error.salem.roi(shape=shp_main)

    return ds_width, dr_width

def dropnan_values_from_xarray(array,
                              namedim_x=str,
                              namedim_y=str,
                              get_xy=False):
    """
    Get rid of nan's inside an xarray data set of specific dimensions, basically returns
    data points for every coordinate with a valid value. Returns an array of points, and if
    get_xy is True gives back the coordinates of those valid points.

    :array xarray to clean
    :namedim_x: string with the x dimension name of the data array
    :namedim_y: string with the y dimension name of the data array
    :get_xy bool set to true if you want also coordinates
    """

    array_stack = array.stack(s=[namedim_x, namedim_y])
    array_nonan = array_stack[array_stack.notnull()]
    # Convert to pandas to drop nans
    ds = array_nonan.to_dataframe()

    if get_xy:
        nx = ds.index.get_level_values(0)
        ny = ds.index.get_level_values(1)
        return array_nonan.data, nx, ny
    else:
        return array_nonan.data

def calculate_observation_thickness(gdir, ds_fls, dr_fls):
    """
    Calculate observation thickness and error along the flowline profile.
    gdir: Glacier directory
    ds_fls: Thickness data along the flowline as a xarray.Dataset
    dr_fls: Thickness error along the flowline as a xarray.Dataset
    Returns
    -------
    h: numpy array with thickness data
    error_h: numpy array with error data
    x_coord: list of longitudes
    y_coord: list of latitudes
    """

    coords = misc._get_flowline_lonlat(gdir)
    x, y = coords[0].geometry[3].coords.xy

    raster_proj = pyproj.Proj(ds_fls.attrs['pyproj_srs'])

    # We will also get data for the entire flowline
    x_all, y_all = salem.gis.transform_proj(wgs84, raster_proj, x, y)
    H_fls = ds_fls.interp(x=x_all, y=y_all, method='nearest')
    H_err_fls = dr_fls.interp(x=x_all, y=y_all, method='nearest')

    # Lets save lon and lat for an easier plotting
    H_fls['lon'] = x
    H_fls['lat'] = y
    H_err_fls['lon'] = x
    H_err_fls['lat'] = y
    # And save the same data with new lat and lon dimensions
    new_data = xr.DataArray(H_fls.data, dims=("lat", "lon"), coords={"lat": y, "lon": x})
    new_error = xr.DataArray(H_err_fls.data, dims=("lat", "lon"), coords={"lat": y, "lon": x})
    H_fls['h_new'] = new_data
    H_err_fls['h_error'] = new_error

    # Drop nan and get the one value per flowline coordinate
    h, x_coord, y_coord = dropnan_values_from_xarray(H_fls.h_new,
                                                     namedim_x='lon',
                                                     namedim_y='lat',
                                                     get_xy=True)

    error_h, x_coord, y_coord = dropnan_values_from_xarray(H_err_fls.h_error,
                                                           namedim_x='lon',
                                                           namedim_y='lat',
                                                           get_xy=True)


    return h, error_h, x_coord, y_coord

@utils.entity_task(log)
def thick_data_to_gdir(gdir, ds=None, dr=None):
    """
    Transforms shapefiles into flowlines, crops the thick data into
    the flowline outline and transform this into an array of points
    Returns only the last thickness or the thickness observation
    form Millan, et.al 2022 along the main flowline.

    gdir: Glacier directory
    ds: Xarray.Dataframe with thickness data
    dr: Xarray.Dataframe with thickness error
    """

    misc.write_flowlines_to_shape(gdir, path=gdir.dir)
    shp_path = os.path.join(gdir.dir, 'glacier_centerlines.shp')
    shp = gpd.read_file(shp_path)

    ds_fls, dr_fls = crop_thick_data_to_flowline(ds, dr, shp)


    thick, error, lon, lat = calculate_observation_thickness(gdir,
                                                             ds_fls,
                                                             dr_fls)

    out = {"h": thick,
           "error": error,
           "lon": lon,
           "lat": lat}

    fp = os.path.join(gdir.dir, 'thickness_data' + '.pkl')
    with open(fp, 'wb') as f:
        pickle.dump(out, f, protocol=-1)
