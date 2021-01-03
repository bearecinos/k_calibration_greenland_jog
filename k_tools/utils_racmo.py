#  Tools to process RACMO data
import pandas as pd
import numpy as np
import logging
import os
import pyproj
import xarray as xr
from collections import defaultdict
from oggm import cfg
from oggm.utils._workflow import ncDataset

# Module logger
log = logging.getLogger(__name__)


def calving_flux_km3yr(gdir, smb):
    """
    Converts SMB (in MB equivalent) to a frontal ablation flux (in Km^3/yr).
    This is necessary to find k values with RACMO data.
    :param
        gdir: Glacier Directory
        smb: Surface Mass balance from RACMO in  mm. w.e a-1
    :return
        q_calving: smb converted to calving flux
    """
    if not gdir.is_tidewater:
        return 0.
    # Original units: mm. w.e a-1, to change to km3 a-1 (units of specific MB)
    rho = cfg.PARAMS['ice_density']
    q_calving = (smb * gdir.rgi_area_m2) / (1e9 * rho)

    return q_calving


def open_racmo(netcdf_path, netcdf_mask_path=None):
    """Opens a netcdf from RACMO with a format PROJ (x, y, projection)
    and DATUM (lon, lat, time)
    :param
        netcdf_path: path to the data
        netcdf_mask_path: Must be given when opening SMB data else needs to be
                          None.
    :return
        out: xarray object with projection and coordinates in order
    """

    # RACMO open variable file
    ds = xr.open_dataset(netcdf_path, decode_times=False)

    if netcdf_mask_path is not None:
        # open RACMO mask
        ds_geo = xr.open_dataset(netcdf_mask_path, decode_times=False)

        try:
            ds['x'] = ds_geo['x']
            ds['y'] = ds_geo['y']
            ds_geo.close()
        except KeyError as e:
            pass

    # Add the proj info to all variables
    proj = pyproj.Proj('EPSG:3413')
    ds.attrs['pyproj_srs'] = proj.srs
    for v in ds.variables:
        ds[v].attrs['pyproj_srs'] = proj.srs

    # Fix the time stamp
    ds['time'] = np.append(
        pd.period_range(start='2018.01.01', end='2018.12.01',
                        freq='M').to_timestamp(),
        pd.period_range(start='1958.01.01', end='2017.12.01',
                        freq='M').to_timestamp())

    out = ds
    ds.close()

    return out


def crop_racmo_to_glacier_grid(gdir, ds):
    """ Crops the RACMO data to the glacier grid
    :param
        gdir: `oggm.GlacierDirectory`
        ds: xarray object
    :return
        ds_sel_roi: xarray with the data cropped to the glacier outline
    """
    try:
        ds_sel = ds.salem.subset(grid=gdir.grid, margin=2)
    except ValueError:
        ds_sel = None

    if ds_sel is None:
        ds_sel_roi = None
    else:
        ds_sel = ds_sel.load().sortby('time')
        ds_sel_roi = ds_sel.salem.roi(shape=gdir.read_shapefile('outlines'))

    return ds_sel_roi


def get_racmo_time_series(ds_sel_roi,
                          var_name,
                          dim_one,
                          dim_two,
                          dim_three,
                          time_start=None, time_end=None, alias=None):
    """ Generates RACMO time series for a time period
     with the data already cropped to the glacier outline.
    :param
        ds_sel_roi: xarray obj already cropped to the glacier outline
        var_name: the variable name to extract the time series from
        dim_one : 'x' or 'lon'
        dim_two: 'y' or 'lat'
        dim_three: 'time'
        time_start: a time where the RACMO time series should begin
        time_end: a time where the RACMO time series should end
        alias: time stamp for the resample
    :return
        ts_31: xarray object with a time series of the RACMO variable, monthly
        data for a reference period.
    """
    if ds_sel_roi is None:
        ts_31 = None
    elif ds_sel_roi[var_name].isnull().all():
        ts_31 = None
    else:
        t_s = ds_sel_roi[var_name].mean(dim=[dim_one,
                                             dim_two],
                                        skipna=True)

        ts = t_s.resample(time=alias).mean(dim=dim_three, skipna=True)

        if time_start is None:
            ts_31 = ts
        else:
            ts_31 = ts.sel(time=slice(time_start, time_end))

    return ts_31


def get_racmo_std_from_moving_avg(ds_sel_roi,
                                  var_name,
                                  dim_one,
                                  dim_two):
    """ Generates RACMO time series for yearly averages and computes
    the std of the variable to analyse.
    :param
        ds_sel_roi: xarray obj already cropped to the glacier outline
        var_name: the variable name to extract the time series from the given
                    array
        dim_one : 'x' or 'lon'
        dim_two: 'y' or 'lat'
    :return
        std: standard deviation of the variable cropped to the glacier outline
    """
    if ds_sel_roi is None:
        std = None
    elif ds_sel_roi[var_name].isnull().all():
        std = None
    else:
        ts = ds_sel_roi[var_name].mean(dim=[dim_one, dim_two], skipna=True)
        mean_yr = ts.rolling(time=12).mean()

        std = mean_yr.std()

    return std


def process_racmo_data(gdir,
                       racmo_path,
                       time_start=None, time_end=None, alias=None):
    """Processes and writes RACMO data in each glacier directory. Computing
    time series of the data for a reference period
    :param
        gdir: `oggm.GlacierDirectory`
        racmo_path: the main path to the RACMO data (see config.ini)
        time_start: a time where the RACMO time series should begin
                    e.g '1961-01-01'
        time_end: a time where the RACMO time series should end
                    e.g '1990-12-01'
        alias: string of time stamp for resample e.g AS, M, D
    :return
        writes an nc file in each glacier directory with the RACMO data
        time series of SMB, precipitation, run off and melt for any reference
        period.
    """

    mask_file = os.path.join(racmo_path,
                             'Icemask_Topo_Iceclasses_lon_lat_average_1km.nc')

    smb_file = os.path.join(racmo_path,
                        'smb_rec.1958-2018.BN_RACMO2.3p2_FGRN055_GrIS.MM.nc')

    prcp_file = os.path.join(racmo_path,
                        'precip.1958-2018.BN_RACMO2.3p2_FGRN055_GrIS.MM.nc')

    run_off_file = os.path.join(racmo_path,
                        'runoff.1958-2018.BN_RACMO2.3p2_FGRN055_GrIS.MM.nc')

    melt_file = os.path.join(racmo_path,
                        'snowmelt.1958-2018.BN_RACMO2.3p2_FGRN055_GrIS.MM.nc')

    fall_file = os.path.join(racmo_path,
                        'snowfall.1958-2018.BN_RACMO2.3p2_FGRN055_GrIS.MM.nc')

    # Get files as xarray all units mm. w.e.
    # Surface Mass Balance
    ds_smb = open_racmo(smb_file, mask_file)
    # Total precipitation: solid + Liquid
    ds_prcp = open_racmo(prcp_file)
    # Run off
    ds_run_off = open_racmo(run_off_file)
    # water that result from snow and ice melting
    ds_melt = open_racmo(melt_file)
    # Solid precipitation
    ds_fall = open_racmo(fall_file)

    # crop the data to glacier outline
    smb_sel = crop_racmo_to_glacier_grid(gdir, ds_smb)
    prcp_sel = crop_racmo_to_glacier_grid(gdir, ds_prcp)
    run_off_sel = crop_racmo_to_glacier_grid(gdir, ds_run_off)
    melt_sel = crop_racmo_to_glacier_grid(gdir, ds_melt)
    fall_sel = crop_racmo_to_glacier_grid(gdir, ds_fall)

    # get RACMO time series in 31 year period centered in t*
    smb_31 = get_racmo_time_series(smb_sel,
                                   var_name='SMB_rec',
                                   dim_one='x',
                                   dim_two='y',
                                   dim_three='time',
                                   time_start=time_start, time_end=time_end,
                                   alias=alias)

    smb_std = get_racmo_std_from_moving_avg(smb_sel,
                                            var_name='SMB_rec',
                                            dim_one='x',
                                            dim_two='y')

    prcp_31 = get_racmo_time_series(prcp_sel,
                                    var_name='precipcorr',
                                    dim_one='lon',
                                    dim_two='lat',
                                    dim_three='time',
                                    time_start=time_start, time_end=time_end,
                                    alias=alias)

    run_off_31 = get_racmo_time_series(run_off_sel,
                                       var_name='runoffcorr',
                                       dim_one='lon',
                                       dim_two='lat',
                                       dim_three='time',
                                       time_start=time_start,
                                       time_end=time_end,
                                       alias=alias)

    melt_31 = get_racmo_time_series(melt_sel,
                                    var_name='snowmeltcorr',
                                    dim_one='lon',
                                    dim_two='lat',
                                    dim_three='time',
                                    time_start=time_start, time_end=time_end,
                                    alias=alias)

    fall_31 = get_racmo_time_series(fall_sel,
                                    var_name='snowfallcorr',
                                    dim_one='lon',
                                    dim_two='lat',
                                    dim_three='time',
                                    time_start=time_start, time_end=time_end,
                                    alias=alias)

    fpath = gdir.dir + '/racmo_data.nc'
    if os.path.exists(fpath):
        os.remove(fpath)

    if smb_31 is None:
        return print('There is no RACMO file for this glacier ' + gdir.rgi_id)
    else:
        with ncDataset(fpath,
                       'w', format='NETCDF4') as nc:

            nc.createDimension('time', None)

            nc.author = 'B.M Recinos'
            nc.author_info = 'Open Global Glacier Model'

            timev = nc.createVariable('time', 'i4', ('time',))

            tatts = {'units': 'year'}

            calendar = 'standard'

            tatts['calendar'] = calendar

            timev.setncatts(tatts)
            timev[:] = smb_31.time

            v = nc.createVariable('smb', 'f4', ('time',))
            v.units = 'mm w.e.'
            v.long_name = 'surface mass balance'
            v[:] = smb_31

            v = nc.createVariable('std', 'f4', ('time',))
            v.units = 'mm w.e.'
            v.long_name = 'std surface mass balance'
            v[:] = smb_std

            v = nc.createVariable('prcp', 'f4', ('time',))
            v.units = 'mm w.e.'
            v.long_name = 'total yearly precipitation amount'
            v[:] = prcp_31

            v = nc.createVariable('run_off', 'f4', ('time',))
            v.units = 'mm w.e.'
            v.long_name = 'total yearly run off amount'
            v[:] = run_off_31

            v = nc.createVariable('snow_melt', 'f4', ('time',))
            v.units = 'mm w.e.'
            v.long_name = 'total yearly snowmelt amount'
            v[:] = melt_31

            v = nc.createVariable('snow_fall', 'f4', ('time',))
            v.units = 'mm w.e.'
            v.long_name = 'total yearly snowfall amount'
            v[:] = fall_31


def get_smb31_from_glacier(gdir):
    """ Reads RACMO data and takes a mean over a reference period for the
        surface mass balance and adds an uncertainty based on the std
        over the entire data period.
    :param
        gdir: `oggm.GlacierDirectory`
    :return
        out_dic: a dictionary with averages and cumulative estimates of smb in
                 original units and in frontal ablation units.
                 It also includes uncertainty.
    """
    fpath = gdir.dir + '/racmo_data.nc'

    if os.path.exists(fpath):
        with ncDataset(fpath, mode='r') as nc:
            smb = nc.variables['smb'][:]
            std = nc.variables['std'][:]
            if smb.all() == 0:
                smb_mean = None
                smb_std = None
                smb_cum = None
                smb_calving_mean = None
                smb_calving_std = None
                smb_calving_cum = None
                print('This glacier has no racmo data ' + gdir.rgi_id)
            else:
                smb_mean = np.nanmean(smb)
                smb_std = np.nanmean(std)
                smb_cum = np.nansum(smb)
                smb_calving_mean = calving_flux_km3yr(gdir, smb_mean)
                smb_calving_std = calving_flux_km3yr(gdir, smb_std)
                smb_calving_cum = calving_flux_km3yr(gdir, smb_cum)
    else:
        print('This glacier has no racmo data ' + gdir.rgi_id)
        smb_mean = None
        smb_std = None
        smb_cum = None
        smb_calving_mean = None
        smb_calving_std = None
        smb_calving_cum = None

    out_dic = dict(smb_mean=smb_mean,
                   smb_std=smb_std,
                   smb_cum=smb_cum,
                   smb_calving_mean=smb_calving_mean,
                   smb_calving_std=smb_calving_std,
                   smb_calving_cum=smb_calving_cum)

    return out_dic


def get_mu_star_from_glacier(gdir):
    """ Reads RACMO data and calculates the mean temperature sensitivity
    from RACMO SMB data and snow melt estimates.
    In a glacier wide average and a mean value of the entire RACMO time series.
    Based on the method described in Oerlemans, J., and Reichert, B. (2000).
    :param
        gdir: `oggm.GlacierDirectory`
    :return
        mean_mu: Mu_star from RACMO, mean value in mm.w.e /K-1
    """

    fpath = gdir.dir + '/racmo_data.nc'

    if os.path.exists(fpath):
        with ncDataset(fpath, mode='r') as nc:
            smb = nc.variables['smb'][:]
            melt = nc.variables['snow_melt'][:]
            mu = smb / melt
            mean_mu = np.average(mu, weights=np.repeat(gdir.rgi_area_km2,
                                                       len(mu)))
    else:
        print('This glacier has no racmo data ' + gdir.rgi_id)
        mean_mu = None

    return mean_mu


def find_k_values_within_racmo_range(df_oggm, df_racmo):
    """
    Finds all k values and OGGM velocity data that is within range of the
    velocity observation and its error. In the case that no OGGM vel is within
    range flags if OGGM overestimates or underestimates velocities.
    :param
        df_oggm: OGGM data from k sensitivity experiment
        df_racmo: observations from MEaSUREs v.1.0
    :return
        out: dictionary with the OGGM data frame crop to observations values or
             with a flag in case there is over estimation or under estimation
    """

    fa_racmo = df_racmo.q_calving_RACMO_mean.values
    error_racmo = df_racmo.q_calving_RACMO_mean_std.values
    r_tol = error_racmo / fa_racmo

    if r_tol < 0.1:
        r_tol = 0.1

    first_oggm_value = df_oggm.iloc[0].calving_flux
    last_oggm_value = df_oggm.iloc[-1].calving_flux

    # Our first conditions is for glaciers that have a racmo flux from a
    # negative range to a positive range.
    # We can use positive racmo values to calibrate OGGM but we have to care
    # about mu_star to don't over do it!
    if (fa_racmo - error_racmo <= 0) and (fa_racmo + error_racmo > 0):
        low_lim = np.zeros(1)
        up_lim = fa_racmo + error_racmo

        index = df_oggm.index[np.isclose(df_oggm.calving_flux,
                                         fa_racmo,
                                         rtol=r_tol, atol=0)].tolist()
        if not index and (first_oggm_value > up_lim):
            df_oggm_new = df_oggm.iloc[0]
            message = 'OGGM overestimates Fa'
        elif not index and (first_oggm_value < up_lim):
            mask = df_oggm['calving_flux'].between(0, up_lim[0])
            df_oggm_new = df_oggm[mask]
            message = 'OGGM is within range, lower RACMO bound is negative' \
                      'but upper is positive and mu_star is positive'
        else:
            df_oggm_new = df_oggm.loc[index]
            mu_stars = df_oggm_new.mu_star
            if mu_stars.iloc[-1] == 0:
                df_oggm_new = df_oggm_new.iloc[-2]
                message = 'OGGM is within range but ' \
                          'mu_star does not allows more calving'
            else:
                df_oggm_new = df_oggm_new
                message = 'OGGM is within range'
    # Some glaciers will always have a postive racmo smb range
    elif fa_racmo - error_racmo > 0:
        low_lim = fa_racmo - error_racmo
        up_lim = fa_racmo + error_racmo

        index = df_oggm.index[np.isclose(df_oggm.calving_flux,
                                         fa_racmo,
                                         rtol=r_tol, atol=0)].tolist()

        if not index and (last_oggm_value < low_lim):
            df_oggm_new = df_oggm
            message = 'OGGM underestimates Fa'
        elif not index and (first_oggm_value > up_lim):
            df_oggm_new = df_oggm.iloc[0]
            message = 'OGGM overestimates Fa'
        else:
            # print(index)
            df_oggm_new = df_oggm.loc[index]
            message = 'OGGM is within range'
    # And some glaciers will have a RACMO smb + error = negative they should
    # not have a calving flux!
    else:
        assert fa_racmo + error_racmo < 0
        df_oggm_new = None
        message = 'This glacier should not calve'
        low_lim = fa_racmo - error_racmo
        up_lim = fa_racmo + error_racmo

    # Make sure we write a warning if the glacier should not calve in terms
    # of RACMO smb being negative
    if df_oggm_new is None and message == 'This glacier should not calve':
        message = message
    elif df_oggm_new.empty and message != 'This glacier should not calve':
        message = 'We need to repeat k experiment'
    else:
        message = message

    if isinstance(df_oggm_new, pd.Series):
        df_oggm_new = df_oggm_new.to_frame().T

    if df_oggm_new is not None:
        df_oggm_new = df_oggm_new.reset_index(drop=True)

    out = defaultdict(list)
    out['oggm_racmo'].append(df_oggm_new)
    out['racmo_message'].append(message)
    out['obs_racmo'].append(df_racmo)
    out['low_lim_racmo'].append(low_lim)
    out['up_lim_racmo'].append(up_lim)

    return out


def merge_racmo_calibration_results_with_glac_stats(calibration_path,
                                                    glac_stats_path,
                                                    volume_bsl_path,
                                                    df_vel,
                                                    exp_name):
    """

    :param calibration_path: path to racmo calibration results csv file
        from performing a linear fits to the input data and model output.
    :param glac_stats_path: path to OGGM glacier stats csv file after running
    the model with a specific k configuration and racmo data.
    (e.g racmo lowbound, value and upbound)
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
                                       'fa_racmo',
                                       'racmo_low_bound',
                                       'racmo_up_bound',
                                       'k_for_lw_bound']]

        d_calibration.rename(columns={
                            'method': 'method_racmo',
                            'k_for_lw_bound': 'k_for_lw_bound_racmo'
        }, inplace=True)

    if "upbound" in exp_name:
        d_calibration = d_calibration[['rgi_id',
                                       'method',
                                       'fa_racmo',
                                       'racmo_low_bound',
                                       'racmo_up_bound',
                                       'k_for_up_bound']]

        d_calibration.rename(columns={
            'method': 'method_racmo',
            'k_for_up_bound': 'k_for_up_bound_racmo'
        }, inplace=True)

    if "value" in exp_name:
        d_calibration = d_calibration[['rgi_id',
                                       'method',
                                       'fa_racmo',
                                       'racmo_low_bound',
                                       'racmo_up_bound',
                                       'k_for_racmo_value']]

        d_calibration.rename(columns={
            'method': 'method_racmo'
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