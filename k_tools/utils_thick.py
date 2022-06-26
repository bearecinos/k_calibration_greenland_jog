import os
import logging
import numpy as np
import pandas as pd
import rasterio
import salem
import pickle
import pyproj
from affine import Affine
from salem import wgs84
import xarray as xr
import geopandas as gpd
from oggm import utils
from oggm.utils._workflow import _get_centerline_lonlat
from k_tools import misc

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

    coords = _get_centerline_lonlat(gdir,
                                   flowlines_output=True)[-1]['geometry'].coords

    lon = coords.xy[0]
    lat = coords.xy[1]

    raster_proj = pyproj.Proj(ds_fls.attrs['pyproj_srs'])

    x_all, y_all = salem.gis.transform_proj(wgs84, raster_proj, lon, lat)
    lon_xr = xr.DataArray(x_all, dims="z")
    lat_xr = xr.DataArray(y_all, dims="z")

    H_fls = ds_fls.interp(x=lon_xr, y=lat_xr, method='nearest')
    H_err_fls = dr_fls.interp(x=lon_xr, y=lat_xr, method='nearest')

    assert len(lon) == len(H_fls)
    assert len(lat) == len(H_err_fls)

    return H_fls.data, H_err_fls.data, lon, lat

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

    # This is faster if we crop to the flowline shape file
    misc.write_flowlines_to_shape(gdir, path=gdir.dir)
    shp_path = os.path.join(gdir.dir, 'glacier_centerlines.shp')
    shp = gpd.read_file(shp_path)

    ds_fls, dr_fls = crop_thick_data_to_flowline(ds, dr, shp)

    thick, err, lon, lat = calculate_observation_thickness(gdir,
                                                           ds_fls,
                                                           dr_fls)

    if np.isnan(thick).all():
        return {}

    else:
        out = {"h": thick,
               "error": err,
               "lon": lon,
               "lat": lat}

        fp = os.path.join(gdir.dir, 'thickness_data' + '.pkl')
        with open(fp, 'wb') as f:
            pickle.dump(out, f, protocol=-1)

@utils.entity_task(log, writes=['gridded_data'])
def millan_data_to_gdir(gdir, ds=None, dr=None, plot_dir=None):
    """
    Project Millan et al. 2022 thickness rasters
    to the glacier grid and store under
    griddata.nc oggm file

    gdir: Glacier directory
    ds: tif file with thickness data
    dr: tif file with thickness error
    """

    # OGGM should download the right tiff here
    dh = salem.GeoTiff(ds)
    dh_r = salem.GeoTiff(dr)

    grid_gla = gdir.grid.center_grid
    proj_h = dh.grid.proj

    x0, x1, y0, y1 = grid_gla.extent_in_crs(proj_h)

    try:
        dh.set_subset(corners=((x0, y0), (x1, y1)), crs=proj_h, margin=4)
        dh_r.set_subset(corners=((x0, y0), (x1, y1)), crs=proj_h, margin=4)
    except RuntimeError:
        log.info("There is no data for this glacier")
        return {}

    grid_h = dh.grid.center_grid

    # Get dat for each raster file
    h = utils.clip_min(dh.get_vardata(), 0)
    h_err = utils.clip_min(dh_r.get_vardata(), 0)

    h_new = gdir.grid.map_gridded_data(h, grid_h, interp='linear')
    h_err_new = gdir.grid.map_gridded_data(h_err, grid_h, interp='linear')

    h_new = utils.clip_min(h_new.filled(0), 0)
    h_err_new = utils.clip_min(h_err_new.filled(0), 0)

    # We mask zero ice as nodata as in bedtopo.py#L56
    h_new = np.where(h_new == 0, np.NaN, h_new)
    h_err_new = np.where(h_err_new == 0, np.NaN, h_err_new)

    #Base url below should come from oggm downloads...

    with utils.ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:
        vn = 'millan_ice_thickness'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x',), zlib=True)
        v.units = 'm'
        ln = 'Ice thickness from Millan, et al. 2022'
        v.long_name = ln
        v.base_url = 'https://www.sedoo.fr/theia-publication-products/?uuid=55acbdd5-3982-4eac-89b2-46703557938c'
        v[:] = h_new

        vn = 'millan_ice_thickness_error'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x',), zlib=True)
        v.units = 'm'
        ln = 'Ice thickness error from Millan, et al. 2022'
        v.long_name = ln
        v.base_url = 'https://www.sedoo.fr/theia-publication-products/?uuid=55acbdd5-3982-4eac-89b2-46703557938c'
        v[:] = h_err_new

    misc.write_flowlines_to_shape(gdir, path=gdir.dir)
    shp_path = os.path.join(gdir.dir, 'glacier_centerlines.shp')
    shp = gpd.read_file(shp_path)

    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        ds = ds.load()
    ds.millan_ice_thickness.attrs['pyproj_srs'] = ds.attrs['pyproj_srs']
    ds.millan_ice_thickness_error.attrs['pyproj_srs'] = ds.attrs['pyproj_srs']

    import matplotlib.pyplot as plt
    cmap_thick = plt.cm.get_cmap('YlGnBu')

    fig, ax = plt.subplots()
    mp = ds.salem.get_map();
    mp.set_shapefile(gdir.read_shapefile('outlines'))
    mp.set_shapefile(shp, color='r')
    mp.set_data(ds.millan_ice_thickness)
    mp.set_cmap(cmap_thick)
    mp.set_levels(np.arange(0, 500, 10))
    mp.set_extend('both')
    mp.visualize(ax=ax, cbar_title='thickness m');
    plt.savefig(os.path.join(plot_dir, gdir.rgi_id + '.png'))
    plt.clf()


def combined_model_thickness_and_observations(file_path):
    """
    Reads a csv containing model and observations thickness
    averages the thickness and uncertainty in the last five
    pixels of the flowline
    """

    base = os.path.basename(file_path)
    rgi_id = os.path.splitext(base)[0]


    df = pd.read_csv(file_path, index_col=0)

    thick_oggm = np.round(np.nanmean(df['thick_end_fls'].iloc[-5:]), decimals = 4)

    try:
        thick_obs = np.round(np.nanmean(df['H_flowline'].iloc[-5:]), decimals = 4)
        error_obs = np.round(np.nanmean(df['H_flowline_error'].iloc[-5:]), decimals = 4)
        return rgi_id, thick_oggm, thick_obs, error_obs
    except KeyError as e:
        print('There is no data for glacier ', rgi_id)
        return {}
