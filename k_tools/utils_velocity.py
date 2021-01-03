import numpy as np
import logging
import pyproj
import pandas as pd
import rasterio
import salem
from affine import Affine
from salem import wgs84
from collections import defaultdict
from k_tools.misc import _get_flowline_lonlat
from oggm import utils
# Module logger
log = logging.getLogger(__name__)


def open_vel_raster(tiff_path):
    """
    Opens a tiff file from Greenland velocity observations
    and calculates a raster of velocities or uncertainties with the
    corresponding color bar
    :param
        tiff_path: path to the data
    :return
        ds: xarray object with data already scaled
    """

    # Processing vel data
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

    # Scale the velocities by the log of the data.
    d = np.log(np.clip(data_in, 1, 3000))
    data_scale = (255 * (d - np.amin(d)) / np.ptp(d)).astype(np.uint8)

    ds.data.values = np.flip(data_scale, 0)

    return ds


def crop_vel_data_to_glacier_grid(gdir, vel, error):
    """
    Crop velocity data and uncertainty to the glacier grid
    for plotting only!
    :param
        gdir: Glacier Directory
        vel: xarray data containing vel or vel errors from
                the whole Greenland
        error: xarray data containing the errors from
                the whole Greenland
    :return
        ds_array: an array of velocity croped to the glacier grid
        dr_array: an array of velocity erros croped to the glacier grid
    """

    # Crop to glacier grid
    ds_glacier = vel.salem.subset(grid=gdir.grid, margin=2)
    dr_glacier = error.salem.subset(grid=gdir.grid, margin=2)

    return ds_glacier, dr_glacier


def crop_vel_data_to_flowline(vel, error, shp):
    """
    Crop velocity data and uncertainty to the glacier flowlines
    :param
        vel: xarray data containing vel or vel errors from
             the whole Greenland
        error: xarray data containing the errors from
               the whole Greenland
        shp: Shape file containing the glacier flowlines
    :return
        ds_array: an array of velocity croped to the glacier main flowline .
        dr_array: an array of velocity erros croped to the glacier
                  main flowline.
    """

    # Crop to flowline
    ds_fls = vel.salem.roi(shape=shp.iloc[[0]])
    dr_fls = error.salem.roi(shape=shp.iloc[[0]])

    return ds_fls, dr_fls


def crop_vel_data_to_flowline_width(vel, error, shp, last_one_third=False):
    """
    Crop velocity data and uncertainty to the glacier flowlines
    :param
        vel: xarray data containing vel or vel errors from
             the whole Greenland
        error: xarray data containing the errors from
               the whole Greenland
        shp: Shape file containing the glacier flowlines catchment widths
    :return
        ds_width: an array of velocity croped to the glacier catchment widths
            of the main flowline .
        dr_width: an array of velocity erros
            croped to the glacier catchment widths of the main flowline .
    """

    # Crop to main flowline and catchment
    shp_main = shp.loc[shp.MAIN == 1]

    if last_one_third is True:
        shp_main_end = shp_main.iloc[-np.int(len(shp_main) / 3):]
        shp_main = shp_main_end

    ds_width = vel.salem.roi(shape=shp_main)
    dr_width = error.salem.roi(shape=shp_main)

    return ds_width, dr_width


def calculate_observation_vel(gdir, ds_fls, dr_fls):
    """
    Calculates the mean velocity and error at the end of the flowline
    exactly 5 pixels upstream of the last part of the glacier that contains
    a velocity measurements
    :param
        gdir: Glacier directory
        ds_flowline: xarray data containing vel observations from the main
                     lowline
        dr_flowline: xarray data containing errors in vel observations from
                     the main flowline
    :return
        ds_mean: a mean velocity value over the last parts of the flowline.
        dr_mean: a mean error of the velocity over the last parts of the
                 main flowline.
    """

    coords = _get_flowline_lonlat(gdir)

    x, y = coords[0].geometry[3].coords.xy

    # We only want one third of the main centerline! kind of the end of the
    # glacier. For very long glaciers this might not be that ideal

    x_2 = x[-np.int(len(x) / 3):]
    y_2 = y[-np.int(len(x) / 3):]

    raster_proj = pyproj.Proj(ds_fls.attrs['pyproj_srs'])

    # For the entire flowline
    x_all, y_all = salem.gis.transform_proj(wgs84, raster_proj, x, y)
    vel_fls = ds_fls.interp(x=x_all, y=y_all, method='nearest')
    err_fls = dr_fls.interp(x=x_all, y=y_all, method='nearest')
    # Calculating means
    ds_mean = vel_fls.mean(skipna=True).data.values
    dr_mean = err_fls.mean(skipna=True).data.values

    # For the end of the glacier
    x_end, y_end = salem.gis.transform_proj(wgs84, raster_proj, x_2, y_2)
    vel_end = ds_fls.interp(x=x_end, y=y_end, method='nearest')
    err_end = dr_fls.interp(x=x_end, y=y_end, method='nearest')
    # Calculating means
    ds_mean_end = vel_end.mean(skipna=True).data.values
    dr_mean_end = err_end.mean(skipna=True).data.values

    vel_fls_all = np.around(ds_mean, decimals=2)
    err_fls_all = np.around(dr_mean, decimals=2)
    vel_fls_end = np.around(ds_mean_end, decimals=2)
    err_fls_end = np.around(dr_mean_end, decimals=2)

    return vel_fls_all, err_fls_all, vel_fls_end, err_fls_end, len(x)


def its_live_to_gdir(gdir, dsx, dsy, dex, dey, fx):
    """ Re-project its_live files to a given glacier directory.
        based on the function from oggm_shop:
        https://github.com/OGGM/oggm/blob/master/oggm/shop/its_live.py#L79
        Variables are added to the gridded_data nc file.
        Re-projecting velocities from one map proj to another is done
        re-projecting the vector distances.
        In this process, absolute velocities might change as well because
        map projections do not always preserve
        distances -> we scale them back to the original velocities as per the
        ITS_LIVE documentation. Which states that velocities are given in
        ground units, i.e. absolute velocities.
        We use bi-linear interpolation to re-project the velocities to
        the local glacier map.

        Parameters
        ----------
        gdir : :py:class:`oggm.GlacierDirectory`
            where to write the data
        dsx: :salem.Geotiff: velocity in the x direction for Greenland
        dsy: :salem.Geotiff: velocity in the y direction for Greenland
        dex: :salem.Geotiff: velocity error in the x direction Greenland
        dey: :salem.Geotiff: velocity error in the y direction Greenland
        fx: path directory to original velocity data (x direction)
        """

    # subset its live data to our glacier map
    grid_gla = gdir.grid.center_grid
    proj_vel = dsx.grid.proj

    x0, x1, y0, y1 = grid_gla.extent_in_crs(proj_vel)

    # Same projection for all the itslive data
    dsx.set_subset(corners=((x0, y0), (x1, y1)), crs=proj_vel, margin=4)
    dsy.set_subset(corners=((x0, y0), (x1, y1)), crs=proj_vel, margin=4)

    dex.set_subset(corners=((x0, y0), (x1, y1)), crs=proj_vel, margin=4)
    dey.set_subset(corners=((x0, y0), (x1, y1)), crs=proj_vel, margin=4)

    grid_vel = dsx.grid.center_grid

    # TODO: this should be taken care of by salem
    # https://github.com/fmaussion/salem/issues/171
    with rasterio.Env():
        with rasterio.open(fx) as src:
            nodata = getattr(src, 'nodata', -32767.0)

    # Get the coords at t0
    xx0, yy0 = grid_vel.center_grid.xy_coordinates

    # Compute coords at t1
    xx1 = dsx.get_vardata()
    yy1 = dsy.get_vardata()

    ex1 = dex.get_vardata()
    ey1 = dey.get_vardata()

    non_valid = (xx1 == nodata) | (yy1 == nodata)
    non_valid_e = (ex1 == nodata) | (ey1 == nodata)

    xx1[non_valid] = np.NaN
    yy1[non_valid] = np.NaN

    ex1[non_valid_e] = np.NaN
    ey1[non_valid_e] = np.NaN

    orig_vel = np.sqrt(xx1 ** 2 + yy1 ** 2)
    orig_vel_e = np.sqrt(ex1 ** 2 + ey1 ** 2)

    xx1 += xx0
    yy1 += yy0

    ex1 += xx0
    ey1 += yy0

    # Transform both to glacier proj
    xx0, yy0 = salem.transform_proj(proj_vel, grid_gla.proj, xx0, yy0)
    xx1, yy1 = salem.transform_proj(proj_vel, grid_gla.proj, xx1, yy1)

    ex0, ey0 = salem.transform_proj(proj_vel, grid_gla.proj, xx0, yy0)
    ex1, ey1 = salem.transform_proj(proj_vel, grid_gla.proj, ex1, ey1)

    # Correct no data after proj as well (inf)
    xx1[non_valid] = np.NaN
    yy1[non_valid] = np.NaN

    ex1[non_valid_e] = np.NaN
    ey1[non_valid_e] = np.NaN

    # Compute velocities from there
    vx = xx1 - xx0
    vy = yy1 - yy0

    ex = ex1 - ex0
    ey = ey1 - ey0

    # Scale back velocities - https://github.com/OGGM/oggm/issues/1014
    new_vel = np.sqrt(vx ** 2 + vy ** 2)
    new_vel_e = np.sqrt(ex ** 2 + ey ** 2)

    p_ok = new_vel > 1e-5  # avoid div by zero
    vx[p_ok] = vx[p_ok] * orig_vel[p_ok] / new_vel[p_ok]
    vy[p_ok] = vy[p_ok] * orig_vel[p_ok] / new_vel[p_ok]

    p_ok_e = new_vel_e > 1e-5  # avoid div by zero
    ex[p_ok_e] = ex[p_ok_e] * orig_vel_e[p_ok_e] / new_vel_e[p_ok_e]
    ey[p_ok_e] = ey[p_ok_e] * orig_vel_e[p_ok_e] / new_vel_e[p_ok_e]

    # And transform to local map
    vx = grid_gla.map_gridded_data(vx, grid=grid_vel, interp='linear')
    vy = grid_gla.map_gridded_data(vy, grid=grid_vel, interp='linear')

    ex = grid_gla.map_gridded_data(ex, grid=grid_vel, interp='linear')
    ey = grid_gla.map_gridded_data(ey, grid=grid_vel, interp='linear')

    # Write
    with utils.ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:
        vn = 'obs_icevel_x'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x',), zlib=True)
        v.units = 'm yr-1'
        v.long_name = 'ITS LIVE velocity data in x map direction'
        v[:] = vx

        vn = 'obs_icevel_y'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x',), zlib=True)
        v.units = 'm yr-1'
        v.long_name = 'ITS LIVE velocity data in y map direction'
        v[:] = vy

        vn = 'obs_icevel_x_error'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x',), zlib=True)
        v.units = 'm yr-1'
        v.long_name = 'ITS LIVE error velocity data in x map direction'
        v[:] = ex

        vn = 'obs_icevel_y_error'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x',), zlib=True)
        v.units = 'm yr-1'
        v.long_name = 'ITS LIVE error velocity data in y map direction'
        v[:] = ey


def calculate_itslive_vel(gdir, ds_fls, dr_fls):
    """
    Calculates the mean velocity and error at the end of the flowline
    exactly 5 pixels upstream of the last part of the glacier that contains
    a velocity measurements from Itslive
    :param
        gdir: Glacier directory
        ds_flowline: xarray data containing vel observations from the main
                     lowline
        dr_flowline: xarray data containing errors in vel observations from
                     the main flowline
    :return
        ds_mean: a mean velocity value over the last parts of the flowline.
        dr_mean: a mean error of the velocity over the last parts of the
                 main flowline.
    """

    coords = _get_flowline_lonlat(gdir)

    x, y = coords[0].geometry[3].coords.xy

    # We only want one third of the main centerline! kind of the end of the
    # glacier. For very long glaciers this might not be that ideal

    x_2 = x[-np.int(len(x) / 3):]
    y_2 = y[-np.int(len(x) / 3):]

    raster_proj = pyproj.Proj(ds_fls.attrs['pyproj_srs'])

    # For the entire flowline
    x_all, y_all = salem.gis.transform_proj(wgs84, raster_proj, x, y)
    vel_fls = ds_fls.interp(x=x_all, y=y_all, method='nearest')
    err_fls = dr_fls.interp(x=x_all, y=y_all, method='nearest')
    # Calculating means
    ds_mean = vel_fls.mean(skipna=True).data
    dr_mean = err_fls.mean(skipna=True).data

    # For the end of the glacier
    x_end, y_end = salem.gis.transform_proj(wgs84, raster_proj, x_2, y_2)
    vel_end = ds_fls.interp(x=x_end, y=y_end, method='nearest')
    err_end = dr_fls.interp(x=x_end, y=y_end, method='nearest')
    # Calculating means
    ds_mean_end = vel_end.mean(skipna=True).data
    dr_mean_end = err_end.mean(skipna=True).data

    vel_fls_all = np.around(ds_mean.flatten()[0], decimals=2)
    err_fls_all = np.around(dr_mean.flatten()[0], decimals=2)
    vel_fls_end = np.around(ds_mean_end.flatten()[0], decimals=2)
    err_fls_end = np.around(dr_mean_end.flatten()[0], decimals=2)

    return vel_fls_all, err_fls_all, vel_fls_end, err_fls_end, len(x)


def calculate_model_vel(gdir, filesuffix=''):
    """ Calculates the average velocity along the main flowline
    in different parts and at the last one third region upstream
    of the calving front
    :param
        gdir: Glacier directory
        filesuffix: any string to be added to the file name
    :return
        surf_fls_vel: surface velocity velocity along all the flowline (m/yr)
        cross_fls_vel: velocity along all the flowline (m/yr)
        surf_calving_front: surface velocity at the calving front (m/yr)
        cross_final: cross-section velocity at the calving front (m/yr)
    """

    if filesuffix is None:
        vel = gdir.read_pickle('inversion_output')[-1]
    else:
        vel = gdir.read_pickle('inversion_output', filesuffix=filesuffix)[-1]

    vel_surf_data = vel['u_surface']
    vel_cross_data = vel['u_integrated']

    length_fls = len(vel_surf_data)/3

    surf_fls_vel = np.nanmean(vel_surf_data)
    cross_fls_vel = np.nanmean(vel_cross_data)

    surf_calving_front = np.nanmean(vel_surf_data[-np.int(length_fls):])
    cross_final = np.nanmean(vel_cross_data[-np.int(length_fls):])

    return surf_fls_vel, cross_fls_vel, surf_calving_front, cross_final


def find_k_values_within_vel_range(df_oggm, df_vel):
    """
    Finds all k values and OGGM velocity data that is within range of the
    velocity observation and its error. In the case that no OGGM vel is within
    range flags if OGGM overestimates or underestimates velocities.
    :param
        df_oggm: OGGM data from k sensitivity experiment
        df_vel: observations from MEaSUREs v.1.0
    :return
        out: dictionary with the OGGM data frame crop to observations values or
             with a flag in case there is over estimation or under estimation
    """

    obs_vel = df_vel.vel_calving_front.values
    error_vel = df_vel.error_calving_front.values

    r_tol = error_vel/obs_vel
    if r_tol < 0.1:
        r_tol = 0.1

    first_oggm_value = df_oggm.iloc[0].velocity_surf
    last_oggm_value = df_oggm.iloc[-1].velocity_surf

    low_lim = obs_vel - error_vel
    up_lim = obs_vel + error_vel

    index = df_oggm.index[np.isclose(df_oggm.velocity_surf,
                                     obs_vel,
                                     rtol=r_tol,
                                     atol=0)].tolist()
    if not index and (last_oggm_value < low_lim):
        df_oggm_new = df_oggm
        message = 'OGGM underestimates velocity'
    elif not index and (first_oggm_value > up_lim):
        df_oggm_new = df_oggm.iloc[0]
        message = 'OGGM overestimates velocity'
    elif not index:
        df_oggm_new = df_oggm
        message = 'k factor step too big'
    else:
        df_oggm_new = df_oggm.loc[index]
        mu_stars = df_oggm_new.mu_star
        if mu_stars.iloc[-1] == 0:
            df_oggm_new = df_oggm_new.iloc[-2]
            message = 'OGGM is within range but mu_star ' \
                      'does not allows more calving'
        else:
            df_oggm_new = df_oggm_new
            message = 'OGGM is within range'

    if isinstance(df_oggm_new, pd.Series):
        df_oggm_new = df_oggm_new.to_frame().T
    else:
        df_oggm_new = df_oggm_new

    df_oggm_new = df_oggm_new.reset_index(drop=True)

    out = defaultdict(list)
    out['oggm_vel'].append(df_oggm_new)
    out['vel_message'].append(message)
    out['obs_vel'].append(df_vel)
    out['low_lim_vel'].append(low_lim)
    out['up_lim_vel'].append(up_lim)

    return out


def merge_vel_calibration_results_with_glac_stats(calibration_path,
                                                  glac_stats_path,
                                                  volume_bsl_path,
                                                  df_vel,
                                                  exp_name):
    """

    :param calibration_path: path to velocity calibration results csv file
        from performing a linear fits to the input data and model output.
    :param glac_stats_path: path to OGGM glacier stats csv file after running
    the model with a specific k configuration and vel data.
    (e.g itslive lowbound, value and upbound)
    :param exp_name: name in str of the k configuration
    :return: dataframe merged
    """

    # Read OGGM stats output
    oggm_stats = pd.read_csv(glac_stats_path)

    # Read calibration output and crop the file to the right k configuration
    d_calibration = pd.read_csv(calibration_path, index_col='Unnamed: 0')
    d_calibration.rename(columns={'RGIId': 'rgi_id'}, inplace=True)

    df_vel = df_vel[['rgi_id', 'velocity_cross', 'velocity_surf']]

    # Read volume below sea level output
    oggm_vbsl = pd.read_csv(volume_bsl_path, index_col='Unnamed: 0')

    oggm_vbsl.rename(columns={
        'RGIId': 'rgi_id',
        'volume bsl': 'vbsl',
        'volume bsl with calving': 'vbsl_c'},
        inplace=True)

    if "lowbound" in exp_name:
        d_calibration = d_calibration[['rgi_id',
                                       'method',
                                       'surface_vel_obs',
                                       'obs_low_bound',
                                       'obs_up_bound',
                                       'k_for_lw_bound']]
        if "itslive" in exp_name:
            d_calibration.rename(columns={
                                'method': 'method_itslive',
                                'surface_vel_obs': 'surface_vel_obs_itslive',
                                'obs_low_bound': 'obs_low_bound_itslive',
                                'obs_up_bound': 'obs_up_bound_itslive',
                                'k_for_lw_bound': 'k_for_lw_bound_itslive'
            }, inplace=True)

        elif "measures" in exp_name:
            d_calibration.rename(columns={
                                'method': 'method_measures',
                                'surface_vel_obs': 'surface_vel_obs_measures',
                                'obs_low_bound': 'obs_low_bound_measures',
                                'obs_up_bound': 'obs_up_bound_measures',
                                'k_for_lw_bound': 'k_for_lw_bound_measures'
            }, inplace=True)

    if "upbound" in exp_name:
        d_calibration = d_calibration[['rgi_id',
                                       'method',
                                       'surface_vel_obs',
                                       'obs_low_bound',
                                       'obs_up_bound',
                                       'k_for_up_bound']]

        if "itslive" in exp_name:
            d_calibration.rename(columns={
                                'method': 'method_itslive',
                                'surface_vel_obs': 'surface_vel_obs_itslive',
                                'obs_low_bound': 'obs_low_bound_itslive',
                                'obs_up_bound': 'obs_up_bound_itslive',
                                'k_for_up_bound': 'k_for_up_bound_itslive'
            }, inplace=True)

        elif "measures" in exp_name:
            d_calibration.rename(columns={
                                'method': 'method_measures',
                                'surface_vel_obs': 'surface_vel_obs_measures',
                                'obs_low_bound': 'obs_low_bound_measures',
                                'obs_up_bound': 'obs_up_bound_measures',
                                'k_for_up_bound': 'k_for_up_bound_measures'
            }, inplace=True)

    if "value" in exp_name:
        d_calibration = d_calibration[['rgi_id',
                                       'method',
                                       'surface_vel_obs',
                                       'obs_low_bound',
                                       'obs_up_bound',
                                       'k_for_obs_value']]

        if "itslive" in exp_name:
            d_calibration.rename(columns={
                                'method': 'method_itslive',
                                'surface_vel_obs': 'surface_vel_obs_itslive',
                                'obs_low_bound': 'obs_low_bound_itslive',
                                'obs_up_bound': 'obs_up_bound_itslive',
                                'k_for_obs_value': 'k_for_obs_value_itslive',
            }, inplace=True)

        elif "measures" in exp_name:
            d_calibration.rename(columns={
                                'method': 'method_measures',
                                'surface_vel_obs': 'surface_vel_obs_measures',
                                'obs_low_bound': 'obs_low_bound_measures',
                                'obs_up_bound': 'obs_up_bound_measures',
                                'k_for_obs_value': 'k_for_obs_value_measures'
            }, inplace=True)

    df_merge = pd.merge(left=oggm_stats,
                        right=d_calibration,
                        how='inner',
                        left_on = 'rgi_id',
                        right_on='rgi_id')

    df_merge_all = pd.merge(left=df_merge,
                            right=oggm_vbsl,
                            how='left',
                            left_on='rgi_id',
                            right_on='rgi_id')

    df_merge_with_vel = pd.merge(left=df_merge_all,
                            right=df_vel,
                            how='left',
                            left_on='rgi_id',
                            right_on='rgi_id')

    return df_merge_with_vel