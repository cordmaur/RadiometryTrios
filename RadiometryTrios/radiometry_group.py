from pathlib import Path
import pandas as pd
import numpy as np
import math
from datetime import timedelta
import configparser
import logging
from plotly import subplots
import plotly.io as pio

from .common import listify, apply_subplot
from .radiometry import BaseRadiometry, Radiometry


# #############  Radiometry Class  ##################
class RadiometryGroup:

    labels = {'Ed': {'y_axis': "Irradiance (mW/(m^2))",
                     'title': 'Irradiance (Ed)'},
              'Ld': {'y_axis': 'Radiance (mW/(m^2 sr))',
                     'title': 'Radiance (Ld)'},
              'Lu': {'y_axis': 'Radiance (mW/(m^2 sr))',
                     'title': 'Radiance (Lu)'},
              'Rrs': {'y_axis': 'Reflectance (sr^-1)',
                      'title': 'Reflectance (Rrs)'}}

    default_prefixes = ['Rrs', 'Ed', 'Ld', 'Lu']

    # define the order for the radiances to be plotted
    plot_order = {'Rrs': 0, 'Ed': 1, 'Lu': 2, 'Ld': 3}

    # ----------  INITIALIZATION METHODS  ----------- #
    def __init__(self, radiometries, metadata, folder=None, logger=None, base_interpolated_name='_interpolated'):
        self.radiometries = radiometries
        self.metadata = metadata
        self.folder = folder
        self.logger = logger
        self.base_interpolated_name = base_interpolated_name
        self.subset = None

    @classmethod
    def open_folder(cls, folder, prefixes=None, base_name='_spectrum_LO', read_backup=False,
                    load_interpolated=False, base_interpolated_name='_interpolated', logger_level=logging.INFO):
        """Open the radiometry on a specific folder. If prefixes are not specified, open the basic [Lu, Ld, Ed, Rrs]"""

        # Create a logger
        logger = logging.getLogger('TriosRadiometry.RadiometryGroup')
        logger.setLevel(logger_level)

        # set the prefixes to be loaded. If not informed, use the default prefixes
        prefixes = RadiometryGroup.default_prefixes if prefixes is None else prefixes

        # convert the folder into a Path object
        folder = Path(folder)

        # check if the folder exists
        if not folder.exists() or not folder.is_dir():
            logger.error(f'Folder {str(folder)} does not exists')
            return

        # get the names of the raw files
        raw_suffixes = ['.bak', '.txt', '.mlb', '.csv'] if read_backup else ['.txt', '.mlb', '.csv']
        raw_names = [(prefix, base_name, raw_suffixes, 'raw') for prefix in prefixes]

        # get the name of the interpolated files
        interp_suffixes = ['.bak', '.csv'] if read_backup else ['.csv']
        interp_names = [(prefix, base_interpolated_name, interp_suffixes, 'interpolated') for
                        prefix in prefixes if load_interpolated]

        # create a dictionary to store the radiometries
        radiometries = {'raw': {}, 'interpolated': {}}

        # Loop through the files to load
        for prefix, base_name, suffixes, rtype in raw_names + interp_names:
            name = prefix + base_name
            fn = BaseRadiometry.get_file_by_suffix(folder, stem=name, suffixes=suffixes)

            # if a file is found, try to open a Radiometry. If it is OK (not None) append it to the list
            if fn:
                logger.debug(f'Found {fn.name}')
                rdmtry = Radiometry.open(fn, r_type=prefix)

                if rdmtry is not None:
                    radiometries[rtype][prefix] = rdmtry

            else:
                logger.debug(f'File {prefix}:{rtype} not found.')

        if len(radiometries['raw']) == 0 and len(radiometries['interpolated']) == 0:
            logger.error(f'Error creating a RadiometryGroup')
            return None
        else:
            metadata = BaseRadiometry.open_metadata(folder, logger)
            return cls(radiometries, metadata, folder, logger, base_interpolated_name)

    def update_metadata(self):
        if self.metadata is None:
            config = configparser.ConfigParser()
            config.add_section('Metadata')
            # config.add_section('MeanReflectance')

            self.metadata = config
        else:
            config = self.metadata

        config.set('Metadata', 'Ed_device', self.get_radiometry('Ed').metadata.set_index('key').loc['IDDevice'].value)
        config.set('Metadata', 'Ld_device', self.get_radiometry('Ld').metadata.set_index('key').loc['IDDevice'].value)
        config.set('Metadata', 'Lu_device', self.get_radiometry('Lu').metadata.set_index('key').loc['IDDevice'].value)

        for rd in self.all_radiometries:
            name = rd.r_type + '_Measurements'
            config.set('Metadata', name, str(len(rd._data)))

        dates = self.times_range(string=False)
        config.set('Metadata', 'Start_Date', str(dates[0]))
        config.set('Metadata', 'End_Date', str(dates[1]))
        config.set('Metadata', 'Duration', str(dates[1] - dates[0]))

        with open(self.folder / 'metadata.txt', 'w') as configfile:
            config.write(configfile)

    # ----------  RADIOMETRY METHODS  ----------- #
    def interpolate_radiometries(self, r_types=None, step=1, min_wl=320, max_wl=950):
        r_types = ['Ed', 'Lu', 'Ld'] if r_types is None else listify(r_types)

        for r_type in r_types:
            if r_type not in self.radiometries['raw']:
                self.logger.info(f'{r_type} not found to be interpolated')
                continue

            rd = self.radiometries['raw'][r_type]

            # Interpolate data
            data = rd.interpolate(step=step, min_wl=min_wl, max_wl=max_wl, inplace=False)

            # Create a brand new interpolated Radiometry
            interp = Radiometry(self.folder/(r_type + self.base_interpolated_name + '.csv'),
                                r_type=r_type,
                                data=data,
                                metadata=None)

            # if there is a subset in the group, set it in the new radiometry
            interp.subset = self.subset

            self.radiometries['interpolated'][r_type] = interp

    def create_reflectance(self, ro=0.028, step=1, min_wl=320, max_wl=950, ed='Ed', lu='Lu', ld='Ld', rrs='Rrs',
                           interpolate=True):

        r_types = [ed, lu, ld]

        if interpolate:
            self.interpolate_radiometries(r_types=r_types, step=step, min_wl=min_wl, max_wl=max_wl)

        ed = self.radiometries['interpolated'][ed]._data
        lu = self.radiometries['interpolated'][lu]._data
        ld = self.radiometries['interpolated'][ld]._data if ld is not None else 0

        # Calculate the reflectance
        r_rs_data = (lu - ro * ld) / ed
        r_rs_data.dropna(axis=0, how='all', inplace=True)

        # Create the interpolated Radiometry
        r_rs = Radiometry(path=self.folder/(rrs + self.base_interpolated_name + '.csv'),
                          r_type=rrs,
                          data=r_rs_data,
                          metadata=None)

        # set the subset accordingly
        r_rs.subset = self.subset

        self.radiometries['interpolated'][rrs] = r_rs

    def get_radiometry(self, r_type, interpolated=False):
        interpolation = 'interpolated' if interpolated else 'raw'

        # First, let's check if the radiometry is already loaded
        if r_type not in self.radiometries[interpolation]:
            print(f"Radiometry {r_type}:{interpolation} not available."
                  f"Loaded: {list(self.radiometries[interpolation].keys())}")
            return None

        else:
            return self.radiometries[interpolation][r_type]

    def get_radiometry_data(self, r_type, use_subset=False, interpolated=False):
        rd = self.get_radiometry(r_type, interpolated)

        if rd is not None:
            return rd.data if use_subset else rd._data
        else:
            return None

    # ----------  PLOTTING METHODS  ----------- #
    def get_title_summary(self, use_subset):
        if use_subset and self.subset is not None:
            s = f'{len(self.subset)} measures from subset: {self.subset[0]}-{self.subset[-1]}'
        else:
            s = f'{len(self.times)} measures from {self.times_range(string=True)}'

        s = f'SPM={self.spm} mg/l | Chl-a={self.chla} ug/l :  ' + s
        return s

    def get_area_location_measurement(self):
        measurement = self.folder.stem
        location = self.folder.parent.stem
        area = self.folder.parent.parent.stem
        return area, location, measurement

    def plot_radiometry(self, r_type='Rrs', use_subset=True, interpolated=True, mean=False, std_delta=1., **kwargs):

        rd = self.get_radiometry(r_type, interpolated=interpolated)

        if rd is None:
            return None

        fig = rd.plot(use_subset=use_subset, mean=mean, std_delta=std_delta, **kwargs)

        if r_type in self.labels:
            fig.update_layout(title=self.labels[r_type]['title'],
                              yaxis_title=self.labels[r_type]['y_axis'])

        return fig

    def plot_radiometries(self, r_types=None, cols=2, base_height=400, use_subset=True, interpolated=True,
                          mean=False, **kwargs):
        """
        Plot all the radiometries that are loaded.
        :param r_types: The radiometry names ex. ['reflectance', 'Ed']. If None plot all loaded radiometries.
        :param cols: Number of columns
        :param base_height: Height for each figure row
        :param use_subset:
        :param interpolated: Indicate if it should plot Interpolated (default) or Raw radiometries.
        :return: Multi-axis figure
        """

        interpolation = 'interpolated' if interpolated else 'raw'
        r_types = self.radiometries[interpolation].keys() if r_types is None else listify(r_types)

        r_types = sorted(r_types, key=lambda x: RadiometryGroup.plot_order[x])

        if len(r_types) == 0:
            print(f"No {interpolation} radiances were found to plot.")
            return

        # get the number of rows
        n = len(r_types)
        rows = math.ceil(n/cols)

        # get the titles of the graphs
        titles = list(r_types)

        # create the main figure
        fig = subplots.make_subplots(rows=rows, cols=cols,
                                     subplot_titles=titles)

        for idx, name in enumerate(r_types):
            position = ((idx // cols) + 1, (idx % cols) + 1)
            subplot = self.plot_radiometry(name, interpolated=interpolated, use_subset=use_subset, mean=mean, **kwargs)
            if subplot is not None:
                apply_subplot(fig, subplot, position)

        # Adjust the title
        title = ' '.join(self.get_area_location_measurement())
        title += "<span style='font-size: 12px;'>"
        title += f'   ({self.folder})<br>'
        title += self.get_title_summary(use_subset=use_subset)
        title += '</span>'

        fig.update_layout(title=title, height=base_height * rows)
        return fig

    @property
    def spm(self):
        return self.get_metadata('Metadata', 'spm', float)

    @property
    def chla(self):
        return self.get_metadata('Metadata', 'chl-a', float)

    def get_metadata(self, section, key, data_type=float):
        if self.metadata is None:
            return
        else:
            value = self.metadata.get(section, key, fallback=None)
            return None if (value is None) or (value == '') else data_type(value)

    # ----------  DATETIME METHODS  ----------- #
    def adjust_time(self, t_delta):
        """Adjust the datetime of the radiances by a timedelta. It is necessary to recreate reflectance afterwards"""
        for rd in self.all_radiometries:
            rd.adjust_time(t_delta)

        new_folder = self.folder.with_name(self.create_measurement_name())
        self.folder.rename(new_folder)
        self.folder = new_folder

    @property
    def subset(self):
        subset = None
        for rd in self.all_radiometries:
            if rd.subset is not None:
                subset = rd.subset if subset is None else subset.intersection(rd.subset)
        return subset

    @subset.setter
    def subset(self, values):
        for rd in self.all_radiometries:
            rd.subset = values

    @property
    def times(self):
        times = None
        for rd in self.all_radiometries:
            times = rd._data.index if times is None else times.union(rd._data.index)
        return times

    def times_range(self, string=True):
        times = self.times

        if string:
            return str(times.min()), str(times.max())
        else:
            return times.min(), times.max()

    @property
    def all_radiometries(self):
        raw_list = list(self.radiometries['raw'].values())
        interpolated_list = list(self.radiometries['interpolated'].values())
        return raw_list + interpolated_list

    # ----------  FILTERING METHODS  ----------- #
    def apply_time_filter(self, start, end=None, interval=timedelta(minutes=5), accumulate=False):
        """
        Creates a subset of times that can be used for displaying/exporting the measurements (through the
        flag `use_subset`).
        :param start: time window start datetime in 'yyyy-mm-dd hh:mm:ss' (str) format
        :param end: time window end datetime in 'yyyy-mm-dd hh:mm:ss' (str) format.
        If None, a interval will be applied to the start time.
        :param interval: A timedelta parameter that's used when end is not passed.
        :param accumulate: If accumulate is False, the filter is always applied to the entire times range,
        otherwise, it is being accumulated every time the method is called. This can be used to fine-tune
        the curves to be exported.
        :return: The resulting datetimes found in the database.
        """

        for rd in self.all_radiometries:
            rd.apply_time_filter(start, end, interval, accumulate)

    def apply_value_filter(self, r_type, min_value=-np.inf, max_value=np.inf, accumulate=False, interpolated=False,
                           wl_window=None):

        rd = self.get_radiometry(r_type, interpolated=interpolated)

        if rd is None:
            return

        subset = rd.apply_value_filter(min_value, max_value, accumulate, wl_window)

        for rd in self.all_radiometries:
            rd.subset = subset

        return self.subset

    def reset_filter(self):
        for rd in self.all_radiometries:
            rd.reset_filter()

    # ----------  IN/OUT METHODS  ----------- #
    def create_measurement_name(self):
        return self.times_range()[0].replace('-', '').replace(' ', '-').replace(':', '')[:13]

    def save_radiometries(self, folder=None, use_subset=True, save_interpolated=True, save_backup=True, csv_sep=';'):

        # Set the output folder
        folder = self.folder if folder is None else Path(folder)

        # if everything is Ok, change current folder to the new one
        if folder.exists() and folder.is_dir():
            self.folder = folder
        else:
            self.logger.error(f'Folder {folder} does not exist.')

        for rd in self.all_radiometries:
            # skip the interpolated radiometries if they are not meant to be saved
            if not save_interpolated and rd.interpolated:
                continue

            fn = (folder/rd.path.stem).with_suffix('.csv' if rd.interpolated else '.txt')

            rd.save(fn, use_subset=use_subset, save_backup=save_backup, csv_sep=csv_sep)

    def save_radiometries_graph(self, folder=None, use_subset=True, mean=False, name=True, **kwargs):
        folder = Path(folder) if folder is not None else self.folder

        if not folder.exists() or not folder.is_dir():
            self.logger.error(f'Folder {folder} does not exist.')
            return

        fig = self.plot_radiometries(use_subset=use_subset, mean=mean, **kwargs)

        if name:
            fn = '_'.join(self.get_area_location_measurement()) + '.png'
        else:
            fn = f'Fig_{self.create_measurement_name()}.png'

        print(f'Saving image {fn} into: {folder}')
        pio.write_image(fig, str(folder/fn), width=2000, height=1200, validate=False)#, engine='kaleido')

    def save_mean_radiometry(self, use_subset=True):
        """This function will create a MEAN spectrum LO.txt, with all the radiances available"""
        data = {}
        for r_type in list(self.radiometries['interpolated'].keys()):
            rdmtry = self.get_radiometry_data(r_type, use_subset=use_subset, interpolated=True)
            rdmtry = rdmtry.median(skipna=False) if len(rdmtry) > 10 else rdmtry.mean()
            data.update({r_type: rdmtry})

        df = pd.DataFrame(data, index=data['Rrs'].index).T.dropna(how='all', axis=1)
        df.insert(0, 'DateTime', str(self.times.mean())[:19])
        df.index.name = 'r_type'

        # create the filename
        fn = self.folder / 'Mean_interpolated.csv'

        # backup
        BaseRadiometry.backup(fn)

        df.to_csv(fn, sep=';')

        return df

    # ##########  SPECIAL METHODS  #############
    def __getitem__(self, r_type):
        """
        Return a specific radiometry. By default, the interpolated is returned,
        if it is not found, return the Raw.
        :param r_type: name of the radiometry to be obtained
        :return: DataFrame with the desired radiometry
        """
        if r_type in self.radiometries['interpolated'].keys():
            return self.get_radiometry_data(r_type, interpolated=True)
        elif r_type in self.radiometries['raw'].keys():
            return self.get_radiometry_data(r_type, interpolated=False)
        else:
            print(f'No radiometry {r_type} found.')

    def __repr__(self):
        s = f'Class Radiometry\n'
        s += f"Raw radiometries: {list(self.radiometries['raw'].keys())} \n"
        s += f"Interpolated radiometries: {list(self.radiometries['interpolated'].keys())} \n"
        s += f'Metadata: {bool(self.metadata is not None)}\n'
        s += f'Folder: {self.folder} \n'
        # s += f'Date Range: {self.times_range()} \n'
        # s += f'Subset: '
        # s += f'{len(self.subset)} items' if isinstance(self.subset, list) else f'{self.subset}'
        return s
