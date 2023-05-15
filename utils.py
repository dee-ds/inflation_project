"""Utility functions."""

import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

from scipy.stats import norm


def roman_arabic_transform(df: pd.DataFrame, col: str, q: bool = False):
    """
    Transform month notation from roman into arabic numerals 
    for chosen data frame column (works inplace).

    Parameters:
        - df: The data frame.
        - col: The name of column containing the dates.
        - q: Apply the function for dates given as quarters; False by default.

    Returns:
        None.
    """
    
    if not q:
        roman = ['I', 'II', 'III', 'IV', 'V', 'VI', 
                 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']
        roman_re = [' ' + num + '$' for num in roman]  # regex

        arabic = list(range(1, 13))
        arabic_str = ['-' + str(num) for num in arabic]  # for months
        
        df[col].replace(roman_re, arabic_str, regex=True, inplace=True)
    else:
        roman_q = ['I-III', 'IV-VI', 'VII-IX', 'X-XII']
        roman_q_re = [' ' + num + '$' for num in roman_q]  # regex

        arabic_q = list(range(3, 13, 3))
        arabic_q_str = ['-' + str(num) for num in arabic_q]  # for quarters
        
        df[col].replace(roman_q_re, arabic_q_str, regex=True, inplace=True)
    
    # change the dates into datetime format
    df[col] = pd.to_datetime(df[col], format='%Y-%m')


def calculate_indices(df: pd.DataFrame, group_cols: list[str], *, 
    base_col: str = 'inflation', base_tf: str = 'm2m', 
    new_tfs: str | list[str] = ['y2y', 'cumul'], 
    rename_base: bool = True):
    """
    Calculate additional indices within chosen time frames.
    
    Parameters:
        - df: The data frame.
        - group_cols: The list of columns to group by.
        - base_col: The base column in the original data 
          ('inflation' or 'wage'); 'inflation' by default.
        - base_tf: The time frame of the base column ('m2m' or 'q2q'); 
          'm2m' by default.
        - new_tfs: The time frames of new indices 
          ('y2y' and/or 'cumul'); ['y2y', 'cumul'] by default.
        - rename_base: Rename the base column with the 'base_tf' ending; 
          True by default.
          
    Returns:
        The pandas `DataFrame` object with additional columns.
    """
    
    if isinstance(new_tfs, str):
        new_tfs = [new_tfs]
    btf_n = 12 if (base_tf == 'm2m') else 4
    
    df = df.sort_index().assign(rate_col=lambda x: x[base_col] / 100.0)
    df[base_col] = df[base_col] - 100.0
    
    if 'y2y' in new_tfs:
        df = df.assign(
            **dict([(base_col + '_y2y', 
                lambda x: x.groupby(group_cols)\
                .rate_col.transform(lambda y: y.rolling(btf_n, 
                min_periods=btf_n).agg(np.product)) * 100.0 - 100.0)])
        )
    if 'cumul' in new_tfs:
        df = df.assign(
            **dict([(base_col + '_cumul', 
                lambda x: x.groupby(group_cols)\
                .rate_col.cumprod() * 100.0 - 100.0)])
        )
    df.drop('rate_col', axis=1, inplace=True)
    
    if rename_base:
        df.rename(columns={base_col: base_col + '_' + base_tf}, inplace=True)
    
    return df


def label_shorthands(df: pd.DataFrame, labels: list[str] | str, 
    col_label: str | None = None, level: int | None = None):
    """
    Select labels with ease, providing only their shorthands.

    Parameters:
        - df: The data frame.
        - labels: The shorthands of the labels to find.
        - col_label: The name of the column with labels.
        - level: The level of the columns MultiIndex with labels 
          (use in the case of pivoted data frame).
          
    Note: Exactly one argument `col_label` or `level` has to be provided.
    
    Returns:
        The list of labels with full names.
    """

    if (col_label is None and level is None) or\
       (col_label is not None and level is not None):
        raise TypeError('Exactly one argument "col_label" or "level" '
                        'has to be provided.')

    if not isinstance(labels, list):
        labels = [labels]
    
    # get all the labels
    if level is not None:
        labels_full = df.columns.get_level_values(level=level).unique()
    else:
        labels_full = df[col_label].unique()
    
    # find the labels using shorthands
    labels_list = []
    for label in labels:
        if label in labels_full:
            # exclude ambiguity (short labels contained in longer ones)
            labels_list.append(label)
        else:
            for _label in labels_full:
                if label in _label:
                    labels_list.append(_label)
                    break
    
    if not labels_list:
        raise ValueError('The decription of the label(s) is unknown, '
              'please correct it.')
    
    return labels_list


def inflation_plot(df: pd.DataFrame, starttime: str = '2010-01', 
    endtime: str = '2023-03', categories: str | list[str] = 'GRAND TOTAL', 
    goods: str | list[str] = 'GRAND TOTAL', quarterly: bool = False):
    """
    Plot the inflation graphs for chosen household(s)/voivodeship(s) 
    and good(s).
    
    Parameters:
        - df: The data frame.
        - starttime: The starting date to consider; '2010-01' by default.
        - endtime: The ending date to consider; '2023-03' by default.
        - categories: The types of households or voivodeships to use; 
          'GRAND TOTAL' by default.
        - goods: The types of goods to use; 'GRAND TOTAL' by default.
        - quarterly: Set True if you want to use the quarterly inflation data; 
          False by default.
    
    Returns:
        The matplotlib `Axes` objects.
    """
    
    if quarterly and categories == 'GRAND TOTAL':
        categories = 'POLAND'
    
    if isinstance(categories, list) and isinstance(goods, list)\
        and len(categories) > 1 and len(goods) > 1:
        raise ValueError('At least one list (categories or goods) ' 
            'has to be singular.')
    
    if quarterly:
        # change the column names in the original data frame 
        # for the universality of further code
        df.rename(columns={'voivodeship': 'household', 
            'inflation_q2q': 'inflation_m2m'}, inplace=True)
        # change default starttime/endtime into quarters
        starttime = '2010-Q1' if starttime == '2010-01' else starttime
        endtime = '2022-Q3' if endtime == '2023-03' else endtime

    # grab the full labels from the shorthands
    categories_iter = label_shorthands(df, categories, 'household')
    goods_iter = label_shorthands(df, goods, 'good')
    
    # prepare the data frame
    _df = df[df.household.isin(categories_iter) & df.good.isin(goods_iter)]\
        .loc[pd.Period(starttime):pd.Period(endtime)]
        
    # re-calculate the cumulative inflation if necessary
    if starttime != '2010-01' and starttime != '2010-Q1':
        _df = _df.sort_index().assign(
            inf_rate=lambda x: x.inflation_m2m.add(100.0) / 100.0, 
            inflation_cumul=lambda x: x.groupby(['household', 'good'])\
                .inf_rate.cumprod() * 100.0 - 100.0
        ).drop('inf_rate', axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    def _df_plot(_df: pd.DataFrame, labels: list, m2m: bool = False):
        """
        The private tool function for creating the plots.
        """
        
        infl_cols = ['inflation_cumul', 'inflation_y2y']
        infl_names = ['cumulative', 'year-to-year']
        
        if not m2m:
            # only the line plots
            for ax, col, name in zip(axes, infl_cols, infl_names):
                _df[col].plot(
                    title=f'The dynamics of {name} inflation:\n{labels[0]}', 
                    ylabel='inflation [%]', ax=ax
                )
        else:
            # the line plots and the bar plot
            for col in infl_cols:
                _df[col].plot(
                    title='The dynamics of inflation:\n'
                          f'{labels[0]} for {labels[1]}', 
                    ylabel='inflation [%]', ax=axes[0]
                )
            axes[0].legend(infl_names)
            _df['inflation_m2m'].plot(
                kind='bar', 
                title='The dynamics of month-to-month inflation:\n'
                      f'{labels[0]} for {labels[1]}', 
                ylabel='inflation [%]', 
                color=np.where(_df['inflation_m2m'] > 0, 'r', 'g'), 
                ax=axes[1]
            )
            axes[1].xaxis.set_major_locator(plt.MaxNLocator('auto'))
            axes[1].xaxis.set_tick_params(rotation=30)

    # plot the graphs
    if len(categories_iter) > 1:
        _df_plot(_df.pivot(columns='household'), goods_iter)
    elif len(goods_iter) > 1:
        _df_plot(_df.pivot(columns='good'), categories_iter)
    else:
        _df_plot(_df, [*goods_iter, *categories_iter], True)
    
    # restore proper column names in the original data frame
    if quarterly:
        df.rename(columns={'household': 'voivodeship', 
            'inflation_m2m': 'inflation_q2q'}, inplace=True)

    return axes


def voivs_gdf():
    """
    Use the geopandas module to load the spatial data of Polish voivodeships.
    
    Returns:
        The geopandas `GeoDataFrame` object.
    """

    # load the data and extract the necessary columns
    voiv_map_gdf = gpd.read_file('voivodeships_map_data/voivodeships.shp', 
        encoding='utf-8').iloc[:, [3, -1]]\
        .rename(columns={'JPT_NAZWA_': 'voivodeship'})

    # improve Polish accents
    acc = ['Ś', 'Ą', 'Ł', 'Ó', 'Ę', 'Ń']
    no_acc = ['S', 'A', 'L', 'O', 'E', 'N']
    voiv_map_gdf['voivodeship'] = voiv_map_gdf['voivodeship'].str.upper()\
        .replace(acc, no_acc, regex=True)
    
    return voiv_map_gdf


def inflation_map(geo_df: gpd.GeoDataFrame, quarter: str = '2022-Q3', 
    good: str = 'GRAND TOTAL', infl_type: str = 'q2q', **kwargs):
    """
    Plot the inflation map of Poland.
    
    Parameters:
        - geo_df: The geopandas geo data frame to use.
        - quarter: The quarter to consider; '2022-Q3' by default.
        - good: The type of good to use for comparison; 
          'GRAND TOTAL' by default.
        - infl_type: The type of inflation to use for comparison ('q2q', 'y2y' 
          or 'cumul'); 'q2q' by default.
        - kwargs: Keyword arguments to provide to geopandas plotting function.
          
    Returns:
        The matplotlib `Axes` object.
    """
    
    # get the inflation type for comparison
    match infl_type:
        case 'q2q': infl_str = 'quarter-to-quarter'
        case 'y2y': infl_str = 'year-to-year'
        case 'cumul': infl_str = 'cumulative'
    
    infl_type = 'inflation_' + infl_type
    
    # seleect the good
    good = label_shorthands(geo_df, good, 'good')[0]
    
    # grab the data
    geo_df = geo_df.query(f'good == "{good}"').loc[quarter]
    geo_df[infl_type] = geo_df[infl_type].round(2)  # for clear annotations
    
    # plot the map
    with plt.style.context({'ytick.direction': 'out'}):
        ax = geo_df.plot(
            column=infl_type, cmap='YlOrRd', linewidth=0.5, edgecolor='k', 
            legend=True, legend_kwds = {"shrink": 0.4}, **kwargs
        )
    
    # annotate the inflation values in the map
    for _, geo_voiv in geo_df.iterrows():
        ax.annotate(text=geo_voiv[infl_type], 
            xy=geo_voiv.geometry.centroid.coords[0], ha='center')
    
    ax.axis('off')
    ax.set_title(f'The inflation of {good}:\n{infl_str} [%] in {quarter}')
    
    return ax


def voivs_hypothesis(df: pd.DataFrame, good: str = 'GRAND TOTAL'):
    """
    Calculate the matrix of p-values between the voivodeships 
    for the null hypothesis (uses the Wald test).
    
    Parameters:
        - df: The data frame with columns corresponding 
          to goods and voivodeships (MultiIndex).
        - good: The type of good to examine for the differences 
          between the voivodeships.
    
    Returns:
        - The pandas `DataFrame` object.
    """

    good = label_shorthands(df, good, level=0)[0]

    df_good = df[good]
    
    # get the squared sample size and the standard normal gen using scipy.stats
    n_sqrt = np.sqrt(df_good.shape[0])
    norm_var = norm(0, 1)
    
    # declare the matrix
    cols = np.sort(df_good.columns)
    df_p_values = pd.DataFrame(
        index=cols, 
        columns=cols
    )
    
    # calculate p-values
    for voiv1, voiv2 in itertools.combinations(cols, 2):
        data1 = df_good[voiv1].values
        data2 = df_good[voiv2].values
        data_diff = np.subtract(data1, data2)

        data_mean = np.mean(data_diff)
        data_std = np.std(data_diff) / n_sqrt
        wald_stat = data_mean / data_std  # the Wald statistic
        p_value = 2 * norm_var.cdf(-abs(wald_stat))  # the p-value
        
        # fill the matrix (lower triangular part)
        df_p_values.at[voiv2, voiv1] = p_value
    
    return df_p_values


def infl_wag_plot(df: pd.DataFrame, starttime: str = '2011-01', 
    endtime: str = '2023-03', category: str | None = None, 
    good: str = 'GRAND TOTAL', section: str = 'total'):
    """
    Plot the year-to-year graphs to compare the inflation and the wages.
    
    Parameters:
        - df: The data frame.
        - starttime: The starting date to consider; '2010-01' by default.
        - endtime: The ending date to consider; '2023-03' by default.
        - category: The type of category to use - None or chosen voivodeship 
          in the case of quarterly data; None by default.
        - good: The type of good to use; 'GRAND TOTAL' by default.
        - section: The type of section to use; 
          'total' enterprise sector by default.
    
    Returns:
        The matplotlib `Axes` objects.
    """
    
    if category:
        starttime = '2011-Q1' if starttime == '2011-01' else starttime
        endtime = '2022-Q3' if endtime == '2023-03' else endtime
    
    # grab the full labels from the shorthands
    good = label_shorthands(df, good, 'good')[0]
    section = label_shorthands(df, section, 'section')[0]
    
    # select the data
    cat_q = f'voivodeship == "{category}"' if category is not None else 'True'
    df = df.query(f'{cat_q} and good == "{good}" and section == "{section}"')\
        .loc[pd.Period(starttime):pd.Period(endtime)]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # build the plots
    df.plot(
        y=['wage_y2y', 'inflation_y2y'], 
        label=['wages', 'inflation'], 
        ylabel='year-to-year changes [%]', 
        ax=axes[0]
    )
    df.plot(
        kind='scatter', 
        x='wage_y2y', y='inflation_y2y', 
        xlabel='y2y wages [%]', ylabel='y2y inflation [%]', color='k', 
        ax=axes[1]
    )
    axes[1].set_box_aspect(1)
    
    # set the title
    if category:
        cat_q = cat_q.replace(' ==', ':').replace('\"', '')
        fig.suptitle('The dynamics of year-to-year wages and inflation\n'
                     f'({cat_q}, good: {good}, section: {section})')
    else:
        fig.suptitle('The dynamics of year-to-year wages and inflation\n'
                     f'(good: {good}, section: {section})')
    
    return axes


def infl_wag_diff_plot(df: pd.DataFrame, period : str, 
    voivodeship: str | None = None, section: str | None = None, 
    good: str | None = None):
    """
    Plot the differences of the year-to-year wages and inflation indices, based
    on the quarterly data. The function returns a map or a box plots.

    Parameters:
        - df: The data frame.
        - period: The date(s) to consider. For the map, choose one single 
          quarter. For the box plots, use interval of quarters, e.g.,
          '2011-Q1:2020-Q4'. To consider the whole time interval, select 'all'.
        - voivodeship: The voivodeship to use for the comparison.
        - section: The section to use for the comparison.
        - good: The good to use for the comparison.
    
    Note: For the map, use one fixed category (voivodeship, section or good). 
    For the box plots, fix two categories.
        
    Returns:
        The matplotlib `Axes` object.
    """
    
    if voivodeship is None and section is None and good is None:
        raise Exception('At least one label has to be provided.')
    
    # get the date or the date slice
    if period == 'all':
        period = slice('2011-Q1', '2022-Q3')
    else:
        period = period if not ':' in period else slice(*period.split(':'))
    
    df = df.loc[period]
    
    # get the labels and query the frame
    pivot_labels = {'voivodeship': voivodeship, 'section': section, 
        'good': good}
    non_NA_labels = {}
    
    if voivodeship is not None:
        voivodeship = label_shorthands(df, voivodeship, 'voivodeship')[0]
        df = df.query(f'voivodeship == "{voivodeship}"')
        pivot_labels.pop('voivodeship')
        non_NA_labels['voivodeship'] = voivodeship
    if section is not None:
        section = label_shorthands(df, section, 'section')[0]
        df = df.query(f'section == "{section}"')
        pivot_labels.pop('section')
        non_NA_labels['section'] = section
    if good is not None:
        good = label_shorthands(df, good, 'good')[0]
        df = df.query(f'good == "{good}"')
        pivot_labels.pop('good')
        non_NA_labels['good'] = good
    
    pivot_labels = list(pivot_labels)
    non_NA_labels_vals = list(non_NA_labels.values())
    non_NA_labels = list(non_NA_labels)
    
    # plot the results
    if isinstance(period, str) and len(pivot_labels) == 2:
        df = df.pivot(
            index=pivot_labels[1], columns=pivot_labels[0], 
            values='wage_infl_diff'
        )
        plot = sns.heatmap(
            df, 
            center=0, square=True, 
            cmap='seismic', cbar_kws={'fraction': 0.025}, 
            linewidths=0.4, linecolor='w', clip_on=False
        ).set(xlabel='', ylabel='', 
            title='The y2y (wages - inflation) [%] for {}\nand \'{}\' {}'\
            .format(period, 
                    non_NA_labels_vals[0], 
                    non_NA_labels[0])
        )
    elif isinstance(period, slice) and len(pivot_labels) == 1:
        plot = df.groupby(pivot_labels[0]).boxplot(
            column='wage_infl_diff', subplots=False, figsize=(16, 8), 
            ylabel='y2y (wages - inflation) [%]', rot=90
        )
        plot.set_title(
            'The distribution of year-to-year (wages - inflation) [%] '
            'from {0} to {1};\n\'{2}\' {3} and \'{4}\' {5}'.format(
                period.start, period.stop, 
                non_NA_labels_vals[0], non_NA_labels[0], 
                non_NA_labels_vals[1], non_NA_labels[1]
            ))
        plot.grid(visible=False, axis='x')
        
        plot_x_labels = [label.get_text()[1:-17] for 
            label in plot.get_xticklabels()]
        plot.set_xticklabels(plot_x_labels)
    else:
        raise Exception('Please provide correct period(s) and label(s); '
            'see the function description for details.')
    
    return plot


def wealth_plot(df: pd.DataFrame, good: str, 
    starttime: str = '2010-01', endtime: str = '2023-03', 
    sections: list[str] | str = 'total', 
    voivodeships: list[str] | str = ['POLAND', 'MAZOWIECKIE']):
    """
    Plot relations between the wages and the prices of chosen product.
    
    Parameters:
        - df: The data frame.
        - good: The type of good to use; must be explicitly provided.
        - starttime: The starting date to consider; '2010-01' by default.
        - endtime: The ending date to consider; '2023-03' by default.
        - section: The type of section(s) to use; 'total' enterprise sector 
          by default. Please provide no more than two sections.
        - voivodeships: The type of voivodeship(s) to use; 'POLAND' and 
          'MAZOWIECKIE' by default.
          Please Provide no more than two voivodeships.
    
    Note: The `sections` and `voivodeships` parameters cannot be both 
    multi-element lists. Compare two sections for one chosen voivodeship, 
    or two voivodeships for one chosen section.
    
    Returns:
        The matplotlib `Axes` objects.
    """
    
    if (isinstance(sections, list) and len(sections) > 1 
        and isinstance(voivodeships, list) and len(voivodeships) > 1):
        raise Exception('Plase select only one section or voivodeship.')
    
    if isinstance(sections, str): sections = [sections]
    if isinstance(voivodeships, str): voivodeships = [voivodeships]
    
    if len(sections) > 2 or len(voivodeships) > 2:
        raise Exception('Plase select no more than two sections '
            'or voivodeships.')
    
    # grab the full labels from the shorthands
    good = label_shorthands(df, good, 'good')[0]
    sections = label_shorthands(df, sections, 'section')
    voivodeships = label_shorthands(df, voivodeships, 'voivodeship')
    
    # query the data
    df = df.loc[pd.Period(starttime):pd.Period(endtime)].query(
        f'good == "{good}" and section.isin({sections}) '
        f'and voivodeship.isin({voivodeships})'
    )
    
    if df.empty:
        raise KeyError('Some label is missing; '
            'possibly the product is not available for chosen viovodeship(s).')
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    many = 'section' if len(sections) == 2 else 'voivodeship'  
        
    # build the line plots
    df_line = df.pivot(columns=many, values=['wage', 'wealth'])
    
    for ax, df_line_w, y_lab in zip([axes[0], axes[1]], 
        [df_line['wage'], df_line['wealth']], ['wage [PLN]', 'wealth']):
            df_line_w.plot(lw=2.5, ylabel=y_lab, ax=ax)
            ax.fill_between(df_line_w.index, 
                df_line_w.iloc[:, 0], df_line_w.iloc[:, 1], 
                hatch='||', facecolor='w'
            )
    
    # build the 2D plots
    df_2D = df.sort_values(['date', many]).groupby(level=0)\
        .apply(lambda x: pd.Series({
            'wages_ratio': x['wage'][1] / x['wage'][0], 
            'product_price': x['price'][0]
            })
        )
    
    df_2D.plot(kind='scatter', x='product_price', y='wages_ratio', 
        xlabel='price', ylabel=f'ratio between {many}s wages', 
        alpha=0.7, ax=axes[2]
    )
    axes[2].set_box_aspect(1)
    
    axes[3].hexbin(x=df_2D.product_price, y=df_2D.wages_ratio, 
        cmap='gray_r', gridsize=20)
    axes[3].set(xlabel='price', ylabel=f'ratio between {many}s wages')
    axes[3].set_box_aspect(1)

    plt.suptitle(f'The comparison for "{good}"')
    plt.tight_layout()
    
    return axes


def basket_plot(df: pd.DataFrame, 
    voivodeship: str, section: str, 
    goods_dict: dict):
    """
    Plot the evolution of wealth in relation to chosen basket of products.
    
    Parameters:
        - df: The data frame.
        - voivodeship: The type of voivodeship to use.
        - section: The type of section to use.
        - goods_dict: The dictionary of products forming the basket 
          (the keys as products types and the values as products amounts).
    
    Returns:
        The matplotlib `Axes` objects.
    """
    
    # grab the full labels from the shorthands
    section = label_shorthands(df, section, 'section')[0]
    voivodeship = label_shorthands(df, voivodeship, 'voivodeship')[0]
    goods = label_shorthands(df, list(goods_dict.keys()), 'good')
    
    goods_dict = {k: v for k, v in zip(goods, goods_dict.values())}
    
    # grab the labels of non-foodstuffs
    n_food = []
    for good in goods:
        if df.query(f'voivodeship == "{voivodeship}" '
            f'and good == "{good}"').empty:
                n_food.append(good)
    
    # if there are no foodstuffs in the basket, grab this one for further use
    if n_food == goods:
        df_apple = df.query(
            f'section == "{section}" and voivodeship == "{voivodeship}" '
            'and good == "Apple juice - per l"'
        )
    
    # query the original data frame for selected categories
    df = df.query(
        f'section == "{section}" and good.isin({goods}) '
        f'and voivodeship.isin(["{voivodeship}", "POLAND"])'
    )

    # add the prices of non-foodstuffs in the voivodeship using the POLAND data
    if n_food:
        
        # grab the frame with non-foodstuffs
        df_nf = df.query(
            f'voivodeship == "POLAND" and good.isin({n_food})'
        )[:]
        
        # assign chosen voivodeship and erase the POLAND wages
        df_nf.voivodeship.replace('POLAND', voivodeship, inplace=True)
        df_nf.wage = np.NaN
        
        # correct the wages in the combined frame
        if n_food != goods:
            df = pd.concat([df, df_nf])\
                .sort_values(['date', 'voivodeship', 'wage'])\
                .fillna(method='ffill')
        else:
            df = pd.concat([df, df_nf, df_apple])\
                .sort_values(['date', 'voivodeship', 'wage'])\
                .fillna(method='ffill')
            df = df[df.good != 'Apple juice - per l']

    # the utility function for calculating the basket price
    def basket_price(df_g):
        df_g_1 = df_g[df_g.name]
        df_comb = df_g_1[goods[0]] * goods_dict[goods[0]]
        for good in goods[1:]:
            df_comb = df_comb + df_g_1[good] * goods_dict[good]
        return df_comb
    
    # calculate the basket price
    df_prices = df.pivot(columns=['voivodeship', 'good'], values='price')\
        .reindex(pd.period_range('2010-01', '2023-03', freq='1M'))\
        .groupby(axis=1, level=0).apply(lambda x: basket_price(x))
    
    # get the wages and join the frames
    df_wages = df.drop(columns=['good', 'price', 'wealth']).drop_duplicates()\
        .pivot(columns='voivodeship', values='wage')
    df_comb = df_prices.join(df_wages, how='left', rsuffix='_w')
    
    # get the wealth for the combined products
    df_final = pd.DataFrame({
        voivodeship: df_comb[voivodeship + '_w'].div(df_comb[voivodeship]), 
        'POLAND': df_comb['POLAND_w'].div(df_comb['POLAND'])
    })
    
    # plot the final frame
    ax = df_final.plot(
        figsize=(8, 4), xlabel='date', ylabel='wealth', 
        title='The evolution of wealth in '
            f'{voivodeship} within \'{section}\' section'
    )
    
    # add the list of products to the plot
    goods_str = 'the basket:\n\n'
    for k, v in goods_dict.items():
        goods_str += k + ': ' + str(v) + '\n'
    
    ax.text(x=1.05 , y=0.5, s=goods_str, verticalalignment='center', 
        transform=ax.transAxes)
    
    return ax
