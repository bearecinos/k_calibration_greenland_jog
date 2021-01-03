import numpy as np
import logging
import geopandas as gpd
import pandas as pd
import os
import xarray as xr
import pickle
from salem import wgs84
from scipy import stats
from shapely.ops import transform as shp_trafo
import shapely.geometry as shpg
from functools import partial
from collections import OrderedDict
from oggm import cfg, graphics, utils
import netCDF4
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import salem

# Module logger
log = logging.getLogger(__name__)


def splitall(path):
    """
    split a path into all its components
    :param path
    :return allparts
    """
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:
            # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:
            # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def get_study_area(rgi, main_path, ice_cap_prepro_path):
    """
    Get study area sum
    :param
    RGI: RGI as a geopandas
    MAIN_PATH: repository path
    ice_cap_prepro_path: ice cap pre-processing to get ica cap areas
    :return
    study_area: Study area
    """
    rgidf = rgi.sort_values('RGIId', ascending=True)

    # Read Areas for the ice-cap computed in OGGM during
    # the pre-processing runs
    df_prepro_ic = pd.read_csv(os.path.join(main_path,
                                            ice_cap_prepro_path))
    df_prepro_ic = df_prepro_ic.sort_values('rgi_id', ascending=True)

    # Assign an area to the ice cap from OGGM to avoid errors
    rgidf.loc[rgidf['RGIId'].str.match('RGI60-05.10315'),
              'Area'] = df_prepro_ic.rgi_area_km2.values

    # Get rgi only for Lake Terminating and Marine Terminating
    glac_type = ['0']
    keep_glactype = [(i not in glac_type) for i in rgidf.TermType]
    rgidf = rgidf.iloc[keep_glactype]

    # Get rgi only for glaciers that have a week connection or are
    # not connected to the ice-sheet
    connection = [2]
    keep_connection = [(i not in connection) for i in rgidf.Connect]
    rgidf = rgidf.iloc[keep_connection]

    study_area = rgidf.Area.astype(float).sum()
    return study_area


def normalised(value):
    """Normalised value
        :params
        value : value to normalise
        :returns
        n_value: value normalised
        """
    value_min = min(value)
    value_max = max(value)
    n_value = (value - value_min) / (value_max - value_min)

    return n_value


def num_of_zeros(n):
    """
    Count the number of zeros after decimal point
    :param n: number
    :return: number of zeros after decimal point
    """
    s = '{:.16f}'.format(n).split('.')[1]
    return len(s) - len(s.lstrip('0'))


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def calculate_volume_percentage(volume_one, volume_two):
    return np.around((volume_two * 100) / volume_one, 2) - 100


def calculate_sea_level_equivalent(value):
    """
    Calculates sea level equivalent of a volume
    of ice in km^3
    taken from: http://www.antarcticglaciers.org
    :param value: glacier volume
    :return: glacier volume in s.l.e
    """
    # Convert density of ice to Gt/km^3
    rho_ice = 900 * 1e-3  # Gt/km^3

    area_ocean = 3.618e8  # km^2
    height_ocean = 1e-6  # km (1 mm)

    # Volume of water required to raise global sea levels by 1 mm
    vol_water = area_ocean * height_ocean  # km^3 of water

    mass_ice = value * rho_ice  # Gt
    return mass_ice * (1 / vol_water)


def write_pickle_file(gdir, var, filename, filesuffix=''):
    """ Writes a variable to a pickle on disk.
    Parameters
    ----------
    gdir: Glacier directory
    var : object the variable to write to disk
    filename : str file name (must be listed in cfg.BASENAME)
    filesuffix : str append a suffix to the filename.
    """
    if filesuffix:
        filename = filename.split('.')
        assert len(filename) == 2
        filename = filename[0] + filesuffix + '.' + filename[1]

    fp = os.path.join(gdir.dir, filename)

    with open(fp, 'wb') as f:
        pickle.dump(var, f, protocol=-1)


def read_pickle_file(gdir, filename, filesuffix=''):
    """ Reads a variable to a pickle on disk.
    Parameters
    ----------
    gdir: Glacier directory
    filename : str file name
    filesuffix : str append a suffix to the filename.
    """
    if filesuffix:
        filename = filename.split('.')
        assert len(filename) == 2
        filename = filename[0] + filesuffix + '.' + filename[1]

    fp = os.path.join(gdir.dir, filename)

    with open(fp, 'rb') as f:
        out = pickle.load(f)

    return out


def read_rgi_ids_from_csv(file_path):
    """
    Function to read a csv file and get the glaciers ID's in that dataframe
    """
    data = pd.read_csv(file_path)
    rgi_ids = data.RGIId.values

    return rgi_ids


def area_percentage(gdir):
    """ Calculates the lowest 5% of the glacier area from the rgi area
    (this is used in the velocity estimation)
    :param gdir: Glacier directory
    :return: area percentage and the index along the main flowline array
    where that lowest 5% is located.
    """
    rgi_area = gdir.rgi_area_m2
    area_percent = 0.05 * rgi_area

    inv = gdir.read_pickle('inversion_output')[-1]

    # volume in m3 and dx in m
    section = inv['volume'] / inv['dx']

    # Find the index where the lowest 5% percent of the rgi area is located
    index = (np.cumsum(section) <= area_percent).argmin()

    return area_percent, index


def _get_flowline_lonlat(gdir):
    """Quick n dirty solution to write the flowlines as a shapefile"""

    cls = gdir.read_pickle('inversion_flowlines')
    olist = []
    for j, cl in enumerate(cls[::-1]):
        mm = 1 if j == 0 else 0
        gs = gpd.GeoSeries()
        gs['RGIID'] = gdir.rgi_id
        gs['LE_SEGMENT'] = np.rint(np.max(cl.dis_on_line) * gdir.grid.dx)
        gs['MAIN'] = mm
        tra_func = partial(gdir.grid.ij_to_crs, crs=wgs84)
        gs['geometry'] = shp_trafo(tra_func, cl.line)
        olist.append(gs)

    return olist


def _get_catchment_widths_lonlat(gdir):
    """Quick n dirty solution to write the flowlines catchment widths
     as a shapefile"""
    cls = gdir.read_pickle('inversion_flowlines')
    olist = []
    for j, cl in enumerate(cls[::-1]):
        for wi, cur, (n1, n2) in zip(cl.widths, cl.line.coords, cl.normals):
            _l = shpg.LineString([shpg.Point(cur + wi / 2. * n1),
                                  shpg.Point(cur + wi / 2. * n2)])

            mm = 1 if j == 0 else 0
            gs = gpd.GeoSeries()
            gs['RGIID'] = gdir.rgi_id
            gs['LE_SEGMENT'] = np.rint(np.max(cl.dis_on_line) * gdir.grid.dx)
            gs['MAIN'] = mm
            tra_func = partial(gdir.grid.ij_to_crs, crs=wgs84)
            gs['geometry'] = shp_trafo(tra_func, _l)
            olist.append(gs)

    return olist


def write_flowlines_to_shape(gdir, filesuffix='', path=''):
    """Write the centerlines in a shapefile.

    Parameters
    ----------
    gdir: Glacier directory
    filesuffix : str add suffix to output file
    path:
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location
    """

    if path == '':
        path = os.path.join(cfg.PATHS['working_dir'],
                            'glacier_centerlines' + filesuffix + '.shp')
    else:
        path = os.path.join(path, 'glacier_centerlines' + filesuffix + '.shp')

    olist = []

    olist.extend(_get_flowline_lonlat(gdir))

    odf = gpd.GeoDataFrame(olist)

    shema = dict()
    props = OrderedDict()
    props['RGIID'] = 'str:14'
    props['LE_SEGMENT'] = 'int:9'
    props['MAIN'] = 'int:9'
    shema['geometry'] = 'LineString'
    shema['properties'] = props

    crs = {'init': 'epsg:4326'}

    # some writing function from geopandas rep
    from shapely.geometry import mapping
    import fiona

    def feature(i, row):
        return {
            'id': str(i),
            'type': 'Feature',
            'properties':
                dict((k, v) for k, v in row.items() if k != 'geometry'),
            'geometry': mapping(row['geometry'])}

    with fiona.open(path, 'w', driver='ESRI Shapefile',
                    crs=crs, schema=shema) as c:
        for i, row in odf.iterrows():
            c.write(feature(i, row))


def write_catchments_to_shape(gdir, filesuffix='', path=''):
    """Write the centerlines in a shapefile.

    Parameters
    ----------
    gdir: Glacier directory
    filesuffix : str add suffix to output file
    path:
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location
    """

    if path == '':
        path = os.path.join(cfg.PATHS['working_dir'],
                            'glacier_catchments' + filesuffix + '.shp')
    else:
        path = os.path.join(path, 'glacier_catchments' + filesuffix + '.shp')

    olist = []

    olist.extend(_get_catchment_widths_lonlat(gdir))

    odf = gpd.GeoDataFrame(olist)

    shema = dict()
    props = OrderedDict()
    props['RGIID'] = 'str:14'
    props['LE_SEGMENT'] = 'int:9'
    props['MAIN'] = 'int:9'
    shema['geometry'] = 'LineString'
    shema['properties'] = props

    crs = {'init': 'epsg:4326'}

    # some writing function from geopandas rep
    from shapely.geometry import mapping
    import fiona

    def feature(i, row):
        return {
            'id': str(i),
            'type': 'Feature',
            'properties':
                dict((k, v) for k, v in row.items() if k != 'geometry'),
            'geometry': mapping(row['geometry'])}

    with fiona.open(path, 'w', driver='ESRI Shapefile',
                    crs=crs, schema=shema) as c:
        for i, row in odf.iterrows():
            c.write(feature(i, row))


def calculate_pdm(gdir):
    """Calculates the Positive degree month sum; is the total sum,
    of monthly averages temperatures above 0°C in a 31 yr period
    centered in t_star year and with a reference height at the free board.

    Parameters
    ----------
    gdir: Glacier directory
    """
    # First we get the years to analise
    # Parameters
    temp_all_solid = cfg.PARAMS['temp_all_solid']
    temp_all_liq = cfg.PARAMS['temp_all_liq']
    # temp_melt = cfg.PARAMS['temp_melt']
    prcp_fac = cfg.PARAMS['prcp_scaling_factor']
    # default_grad = cfg.PARAMS['temp_default_gradient']

    df = gdir.read_json('local_mustar')
    tstar = df['t_star']
    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
    # Year range
    yr = [tstar - mu_hp, tstar + mu_hp]

    # Then the heights
    heights = gdir.get_inversion_flowline_hw()[0]

    # Then the climate data
    ds = xr.open_dataset(gdir.get_filepath('climate_historical'))
    # We only select the years tha we need
    new_ds = ds.sel(time=slice(str(yr[0])+'-01-01',
                               str(yr[1])+'-12-31'))
    # we make it a data frame
    df = new_ds.to_dataframe()

    # We create the new matrix
    igrad = df.temp * 0 + cfg.PARAMS['temp_default_gradient']
    iprcp = df.prcp
    iprcp *= prcp_fac

    npix = len(heights)

    # We now estimate the temperature gradient
    grad_temp = np.atleast_2d(igrad).repeat(npix, 0)
    grad_temp *= (heights.repeat(len(df.index)).reshape(
        grad_temp.shape) - new_ds.ref_hgt)
    temp2d = np.atleast_2d(df.temp).repeat(npix, 0) + grad_temp

    # Precipitation
    prcpsol = np.atleast_2d(iprcp).repeat(npix, 0)
    fac = 1 - (temp2d - temp_all_solid) / (temp_all_liq - temp_all_solid)
    fac = np.clip(fac, 0, 1)
    prcpsol = prcpsol * fac

    data_temp = pd.DataFrame(temp2d,
                             columns=[df.index],
                             index=heights)
    data_prcp = pd.DataFrame(prcpsol,
                             columns=[df.index],
                             index=heights)

    temp_free_board = data_temp.iloc[-1]
    solid_prcp_top = data_prcp.iloc[0].sum()

    pdm_temp = temp_free_board[temp_free_board > 0].sum()
    pdm_number = temp_free_board[temp_free_board > 0].count()

    return pdm_temp, pdm_number, solid_prcp_top


def solve_linear_equation(a1, b1, a2, b2):
    """
    Solve linear equation

    Parameters
    ----------
        a1: Observation slope (either from the
                lower bound, value, upper bound)
        b1: Observation intercept. (either from the
                lower bound, value, upper bound)
        a2: Linear fit slope to the model data.
        b2: Linear fit intercept to the model data
    Returns
    -------
        z: Intercepts (x, y) x will be k and y velocity.
    """

    a = np.array([[-a1, 1], [-a2, 1]], dtype='float')
    b = np.array([b1, b2], dtype='float')

    z = np.linalg.solve(a, b)
    return z


def get_core_data(df):
    """
    df: dataframe of glacier statistics from which
    we will extract the core data that does not change
    with k values
    return: df_core: core of the data set
    """

    df_core = df[['rgi_id', 'rgi_region', 'rgi_subregion', 'name',
                  'cenlon', 'cenlat', 'rgi_area_km2', 'glacier_type',
                  'terminus_type', 'status', 'dem_source',
                  'dem_needed_interpolation', 'dem_invalid_perc',
                  'dem_invalid_perc_in_mask', 'n_orig_centerlines',
                  'volume_before_calving', 'calving_front_width',
                 'calving_front_slope', 'calving_front_free_board',
                 'perc_invalid_flowline', 'inversion_glen_a', 'inversion_fs',
                 'dem_needed_extrapolation', 'dem_extrapol_perc']]

    df_core['volume_before_calving'] = df_core.volume_before_calving*1e-9

    return df_core


def get_k_dependent(df, df_vel, exp_name):
    """
    df: dataframe of glacier statistics from which
    we will extract the data that changes with k values
    returns: df_dep: k dependent variables per run
    """

    # Get dependent data according to exp name
    df_dep_core = df[['rgi_id', 'inv_volume_km3', 'inv_thickness_m',
                 'vas_volume_km3', 'vas_thickness_m', 'calving_flux',
                 'calving_mu_star', 'calving_law_flux', 'calving_water_level',
                 'calving_inversion_k', 'calving_front_water_depth',
                 'calving_front_thick', 'vbsl', 'vbsl_c']]

    df_vel = df_vel[['rgi_id', 'velocity_cross', 'velocity_surf']]

    df_merge_core = pd.merge(left=df_dep_core,
                        right=df_vel,
                        how='left',
                        left_on='rgi_id',
                        right_on='rgi_id')

    df_merge_core.columns = [col+'_'+exp_name for col in df_merge_core.columns]

    df_merge_core = df_merge_core.rename(columns={'rgi_id'+'_'+exp_name: 'rgi_id'})

    if 'itslive' in exp_name:
        extra = df[['method_itslive', 'surface_vel_obs_itslive',
                'obs_low_bound_itslive', 'obs_up_bound_itslive']]
    if 'measures' in exp_name:
        extra = df[['method_measures', 'surface_vel_obs_measures',
                    'obs_low_bound_measures', 'obs_up_bound_measures']]
    if 'racmo' in exp_name:
        extra = df[['method_racmo', 'fa_racmo', 'racmo_low_bound',
                    'racmo_up_bound',]]

    df_dep = pd.concat([df_merge_core, extra], axis=1)

    return df_dep


def summarize_exp(df):
    """
    df: dataframe of glacier statistics from which
    we will extract the calving fluxes and volumes per experiment
    return: df_results: results of each exp.
    """

    df_results = df[['rgi_id',
                     'rgi_area_km2',
                     'volume_before_calving',
                     'inv_volume_km3',
                     'calving_flux',
                     'vbsl',
                     'vbsl_c']]

    df_results['volume_before_calving'] = df_results.volume_before_calving*1e-9

    return df_results


def calculate_study_area(ids, geo_df):
    """ Calculates the area for a selection of ids in a shapefile
    """
    keep_ids = [(i in ids) for i in geo_df.RGIId]
    rgi_ids = geo_df.iloc[keep_ids]
    area_sel = rgi_ids.Area.sum()

    return area_sel


def calculate_statistics(obs, model, area_coverage, z):
    """
    Calculates statistics between velocity and racmo observations and
    estimates done by oggm after k calibration
    obs: measures, itslive or racmo observations
    model: model resuts after k calibration
    area_coverage: study area percentage represented in the data
    z: an array of the same length as the data frames
    returns
    -------
    test: a box of the statistics to plot in a figure
    zline: fitted line
    wline: fitted line
    """

    RMSD = utils.rmsd(obs, model)

    mean_dev = utils.md(obs, model)

    slope, intercept, r_value, p_value, std_err = stats.linregress(obs, model)

    test = AnchoredText(' Area % = ' + str(format(area_coverage, ".2f")) +
                        '\n slope = ' + str(format(slope, ".2f")) +
                        '\n intercept = ' + str(format(intercept, ".2f")) +
                        '\n r$^2$ = ' + str(format(r_value, ".2f")) +
                        '\n RMSD = ' + str(format(RMSD, ".2f")) +
                        ' m$yr^{-1}$' +
                        '\n Bias = ' + str(format(mean_dev, ".2f")) +
                        ' m$yr^{-1}$',
                        prop=dict(size=12), frameon=True, loc=1)

    zline = slope * z + intercept
    wline = 1 * z + 0

    return test, zline, wline


@graphics._plot_map
def plot_inversion_diff(gdirs, ax=None, smap=None,
                        linewidth=3,
                        vmax=None, k=None):
    """Plots inversion differences before and after calving for
    a glacier directory."""

    gdir = gdirs[0]
    with netCDF4.Dataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    # Dirty optim
    try:
        smap.set_topography(topo)
    except ValueError:
        pass

    toplot_th = np.array([])
    toplot_lines = []
    toplot_crs = []
    vol = []
    for gdir in gdirs:
        crs = gdir.grid.center_grid
        geom = gdir.read_pickle('geometries')
        inv = gdir.read_pickle('inversion_output',
                               filesuffix='_without_calving_')

        inv_c = gdir.read_pickle('inversion_output')

        # Plot boundaries
        poly_pix = geom['polygon_pix']
        smap.set_geometry(poly_pix, crs=crs, fc='none', zorder=2,
                          linewidth=.2)
        for l in poly_pix.interiors:
            smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)

        # plot Centerlines
        cls = gdir.read_pickle('inversion_flowlines')
        for l, c, cc in zip(cls, inv, inv_c):

            smap.set_geometry(l.line, crs=crs, color='gray',
                              linewidth=1.2, zorder=50)
            toplot_th = np.append(toplot_th, cc['thick'] - c['thick'])
            for wi, cur, (n1, n2) in zip(l.widths, l.line.coords, l.normals):
                l = shpg.LineString([shpg.Point(cur + wi / 2. * n1),
                                     shpg.Point(cur + wi / 2. * n2)])
                toplot_lines.append(l)
                toplot_crs.append(crs)
            vol.extend(cc['volume'] - c['volume'])

    cm = plt.cm.get_cmap('YlOrRd')
    dl = salem.DataLevels(cmap=cm, nlevels=256, data=toplot_th,
                          vmin=0, vmax=vmax)
    colors = dl.to_rgb()
    for l, c, crs in zip(toplot_lines, colors, toplot_crs):
        smap.set_geometry(l, crs=crs, color=c,
                          linewidth=linewidth, zorder=50)

    smap.plot(ax)
    return dict(cbar_label='delta thickness \n [m]',
                cbar_primitive=dl,
                title_comment=' ({:.2f} km3)'.format(np.nansum(vol) * 1e-9))