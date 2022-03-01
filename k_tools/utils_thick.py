import logging
import numpy as np
import rasterio
from affine import Affine
import salem
from k_tools.misc import _get_flowline_lonlat
import pyproj
from salem import wgs84
import xarray as xr
from oggm import entity_task

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

def calculate_observation_thickness(gdir, ds_fls, dr_fls, return_profile=False):
    """
    Calculate observation thickness and error at the end of the flowline and
    along the flowline profile.
    if return_profile is True returns thickness data for the entire flowline as
     an array of values: thick, error, lon, lat
    if return_profile is False only returns data at the last pixel of the flowline
    as values: thick, error at last pixel
    """

    coords = _get_flowline_lonlat(gdir)
    x, y = coords[0].geometry[3].coords.xy

    # We only want the last pixel of the main centerline as
    # thickness changes a lot along a profile so probably
    # the best its to just pick the last pixel
    x_2 = x[-1]
    y_2 = y[-1]

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

    # For the end of the glacier
    x_end, y_end = salem.gis.transform_proj(wgs84, raster_proj, x_2, y_2)
    H_end = ds_fls.interp(x=x_end, y=y_end, method='nearest')
    H_err_end = dr_fls.interp(x=x_end, y=y_end, method='nearest')

    # Calculating means
    ds_mean_end = H_end.mean(skipna=True).data.values
    dr_mean_end = H_err_end.mean(skipna=True).data.values

    # Drop nan and get the one value per flowline coordinate
    h, x_coord, y_coord = dropnan_values_from_xarray(H_fls.h_new,
                                                     namedim_x='lon',
                                                     namedim_y='lat',
                                                     get_xy=True)

    error_h, x_coord, y_coord = dropnan_values_from_xarray(H_err_fls.h_error,
                                                           namedim_x='lon',
                                                           namedim_y='lat',
                                                           get_xy=True)

    if return_profile:
        return h, error_h, x_coord, y_coord
    else:
        return ds_mean_end, dr_mean_end