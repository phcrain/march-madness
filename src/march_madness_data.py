import polars as pl
import requests
from bs4 import BeautifulSoup
import re
import time
import os
import random
import numpy as np
import rapidfuzz as rf
import datetime
from src.config import ROUND_NAMES


def round_regex(round_col: str) -> pl.Expr:
    return (
        pl.when(pl.col(round_col).str.contains('64')).then(pl.lit(ROUND_NAMES[0]))
        .when(pl.col(round_col).str.contains('32')).then(pl.lit(ROUND_NAMES[1]))
        .when(pl.col(round_col).str.contains(r'(?i)sixteen|16')).then(pl.lit(ROUND_NAMES[2]))
        .when(pl.col(round_col).str.contains(r'(?i)eight|8')).then(pl.lit(ROUND_NAMES[3]))
        .when(pl.col(round_col).str.contains(r'(?i)final four|final 4')).then(pl.lit(ROUND_NAMES[4]))
        .when(pl.col(round_col).str.contains(r'(?i)champion')).then(pl.lit(ROUND_NAMES[5]))
        .otherwise(pl.col(round_col))
    )


def res_mean(df: pl.DataFrame | pl.LazyFrame,
             col: str,
             lower: float = 0.10,
             upper: float = 0.90) -> pl.DataFrame | pl.LazyFrame | pl.Null:
    """ Calculate an outlier-resilient mean and standard deviation

    Col values are filtered to include only values between the given percentiles before calculating mean and std

    Parameters
    ----------
    df: pl.DataFrame | pl.LazyFrame
        Tabular data containing column of interest
    col: str
        Name of column of interest
    lower: float, default = 0.10
        Lower percentile to cut off data below
    upper: float, default = 0.90
        Upper percentile to cut off data above

    Returns
    -------
    float
        Mean of data within the inter-percentile range of the specified percentiles
    """
    numeric_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                     pl.Float32, pl.Float64, pl.Decimal,
                     # Boolean included since it is converted to int
                     pl.Boolean]
    # Select col of interest
    x = df.select(col)
    # Get data type
    dtype = x.collect_schema()[col]
    # Return Null if type is not numeric
    if dtype not in numeric_types:
        empty_frame = pl.LazyFrame([], schema=[col])
        if isinstance(df, pl.LazyFrame):
            return empty_frame
        else:
            return empty_frame.collect()
    # Convert bool to int
    if dtype == pl.Boolean:
        x = x.with_columns(pl.col(col).cast(pl.UInt8))
    x = x.filter(
        (pl.col(col).ge(pl.col(col).quantile(lower, 'nearest'))) &
        (pl.col(col).le(pl.col(col).quantile(upper, 'nearest')))
    )
    # Return the mean and std of the clipped range
    return pl.concat([x.mean(), x.std().rename({col: f'{col}_std'})], how='horizontal')


def get_team_slug(team_name):
    """Convert a team name into the Sports-Reference slug."""
    team_name = re.sub(r'[^A-Za-z\s-]', '', team_name)
    team_name = team_name.lower().replace(" ", "-")
    team_name = re.sub(r'-st$', '-state', team_name)
    team_name = re.sub(r'^uc-', 'california-', team_name)
    team_name = re.sub(r'^c-', 'central-', team_name)
    team_name = re.sub(r'^e-', 'eastern-', team_name)
    alt_map = {
        'albany': 'albany-ny',
        'app-state': 'appalachian-state',
        'arkansaspine-bluff': 'arkansas-pine-bluff',
        'bonaventure': 'st-bonaventure',
        'byu': 'brigham-young',
        'cal-st-bakersfield': 'cal-state-bakersfield',
        'cal-st-fullerton': 'cal-state-fullerton',
        'centenary': 'centenary-la',
        'central-connecticut-state': 'central-connecticut',
        'charleston': 'college-of-charleston',
        'col-of-charleston': 'college-of-charleston',
        'detroit': 'detroit-mercy',
        'fau': 'florida-atlantic',
        'fdu': 'fairleigh-dickinson',
        'fiu': 'florida-international',
        'g-washington': 'george-washington',
        'grambling-state': 'grambling',
        'illinoischicago': 'illinois-chicago',
        'kansas-city': 'missouri-kansas-city',
        'little-rock': 'arkansas-little-rock',
        'liu': 'long-island-university',
        'liu-brooklyn': 'long-island-university',
        'long-island': 'long-island-university',
        'louisiana': 'louisiana-lafayette',
        'loyola-chicago': 'loyola-il',
        'lsu': 'louisiana-state',
        'mcneese': 'mcneese-state',
        'md-eastern': 'maryland-eastern-shore',
        'miami-fla': 'miami-fl',
        'miami-ohio': 'miami-oh',
        'morehead': 'morehead-state',
        'mt-st-marys': 'mount-st-marys',
        'mtsu': 'middle-tennessee',
        'n-carolina': 'north-carolina',
        'n-colorado': 'northern-colorado',
        'n-illinois': 'northern-illinois',
        'nc-at': 'north-carolina-at',
        'nc-central': 'north-carolina-central',
        'nc-state': 'north-carolina-state',
        'nm-state': 'new-mexico-state',
        'north-kentucky': 'northern-kentucky',
        'nwestern-state': 'northwestern-state',
        'ole-miss': 'mississippi',
        'omaha': 'nebraska-omaha',
        'penn': 'pennsylvania',
        'pitt': 'pittsburgh',
        'prairie-view-am': 'prairie-view',
        'purdue-fort-wayne': 'ipfw',
        'queens': 'queens-nc',
        's-dakota-state': 'south-dakota-state',
        'saint-francis': 'saint-francis-pa',
        'saint-marys': 'saint-marys-ca',
        'san-jos-state': 'san-jose-state',
        'sc-upstate': 'south-carolina-upstate',
        'se-missouri': 'southeast-missouri-state',
        'se-missouri-state': 'southeast-missouri-state',
        'se-missouri-st': 'southeast-missouri-state',
        'sf-austin': 'stephen-f-austin',
        'siu-edwardsville': 'southern-illinois-edwardsville',
        'smu': 'southern-methodist',
        'so-indiana': 'southern-indiana',
        'southern-miss': 'southern-mississippi',
        'st-francis-brookyln': 'st-francis-ny',
        'st-francis-pa': 'saint-francis-pa',
        'st-johns': 'st-johns-ny',
        'st-marys': 'saint-marys-ca',
        'st-marys-ca': 'saint-marys-ca',
        'st-peters': 'saint-peters',
        'tcu': 'texas-christian',
        'tex-am-cc': 'texas-am-corpus-christi',
        'texas-am-cc': 'texas-am-corpus-christi',
        'texas-am---cc': 'texas-am-corpus-christi',
        'texas-am-corpus-chris': 'texas-am-corpus-christi',
        'texaspan-american': 'texas-pan-american',
        'texasrio-grande-valley': 'texas-pan-american',
        'troy-state': 'troy',
        'uab': 'alabama-birmingham',
        'uc-irvine': 'california-irvine',
        'uc-santa-barbara': 'california-santa-barbara',
        'ucf': 'central-florida',
        'uconn': 'connecticut',
        'ucsb': 'california-santa-barbara',
        'umass': 'massachusetts',
        'umbc': 'maryland-baltimore-county',
        'unc-asheville': 'north-carolina-asheville',
        'unc-charlotte': 'charlotte',
        'unc-greensboro': 'north-carolina-greensboro',
        'unc-wilmington': 'north-carolina-wilmington',
        'unlv': 'nevada-las-vegas',
        'usc': 'southern-california',
        'ut-chattanooga': 'chattanooga',
        'ut-martin': 'tennessee-martin',
        'utep': 'texas-el-paso',
        'utsa': 'texas-san-antonio',
        'vcu': 'virginia-commonwealth',
        'w-illinois': 'western-illinois',
        'w-michigan': 'western-michigan',
        'william--mary': 'william-mary',
        'wisconsin-milwaukee': 'milwaukee',
    }
    return alt_map.get(team_name, team_name)



def get_selection_sunday(year: int | str) -> datetime.date:
    """
    Returns the date of Selection Sunday for a given NCAA tournament year.

    Parameters
    ----------
    year: int | str
        The year of the NCAA tournament

    Returns
    -------
    datetime.date
        The date of Selection Sunday for that year

    Examples
    --------
    ```{py}
    from src.player_stats import get_selection_sunday

    print(get_selection_sunday(2025))  # Output: 2025-03-16
    ```
    """
    # Start from March 1st of the given year
    march_first = datetime.date(int(year), 3, 1)
    # Find the weekday of March 1st (0 = Monday, 6 = Sunday)
    weekday = march_first.weekday()

    # Find the first Sunday in March
    days_until_sunday = (6 - weekday)
    second_sunday = march_first + datetime.timedelta(days=days_until_sunday + 7)

    # Heuristic: if March 1st is Wednesday or later, use third Sunday instead
    if weekday < 4:
        selection_sunday = second_sunday
    else:
        selection_sunday = second_sunday + datetime.timedelta(days=7)

    return selection_sunday


round_dict = {rnd: i+1 for i, rnd in enumerate(ROUND_NAMES)}

march_madness_raw = pl.scan_csv(
    'data/March_Madness_Dataset.csv',
    # March_Madness_Data.csv must be updated manually with teams at the start of MM and with results at the end of MM
    schema={'Year': pl.String,
            'Round': pl.String,
            'W_Region': pl.String,
            'W_Seed': pl.String,
            'W_Team': pl.String,
            'W_Score': pl.String,
            'L_Region': pl.String,
            'L_Seed': pl.String,
            'L_Team': pl.String,
            'L_Score': pl.String,
            'OT': pl.String}
)


def march_madness_data(df: pl.LazyFrame = march_madness_raw) -> pl.LazyFrame:
    """ Read and format march madness data

    Parameters
    ----------
    lazy: (bool)
        If set to True, returns a LazyFrame (default). Otherwise, returns a DataFrame.

    Returns
    -------

    """
    # Load dataset

    # Strip whitespace from entire df
    df = df.with_columns(pl.all().str.strip_chars())
    # Convert str cols to ints or bool
    df = df.with_columns(
        pl.col('Year').str.to_integer(),
        pl.col('W_Score').str.to_integer(),
        pl.col('L_Score').str.to_integer(),
        pl.col('OT') == "1"
    )
    # Remove any play-in games
    df = df.filter(~pl.col('Round').str.contains('(?i)opening'))

    # Standardize round names
    df = df.with_columns(round_regex('Round').alias('Round'))

    game_cols = ['Year', 'Round', 'OT']  # not team specific cols

    df_long = (
        pl.concat([
            df.rename(lambda x: x if x in game_cols else x.replace('W_', 'Self_').replace('L_', 'Opp_')),
            df.rename(lambda x: x if x in game_cols else x.replace('W_', 'Opp_').replace('L_', 'Self_'))
        ], how='diagonal')
        .sort(game_cols)
        .with_columns(
            pl.col('Self_Seed').cast(pl.Int16),  # convert seed to int
            pl.col('Opp_Seed').cast(pl.Int16),  # convert seed to int
            pl.col('Round').map_elements(lambda _: round_dict.get(_), pl.Int16),
            # convert names to standardized slugs
            pl.col('Self_Team').map_elements(get_team_slug, pl.String),  # convert names to standardized slugs
            pl.col('Opp_Team').map_elements(get_team_slug, pl.String),  # convert names to standardized slugs
        )
    )

    if 'GameID' in df_long.collect_schema():
        df_long = df_long.with_columns(pl.concat_str(['GameID', 'Self_Team'], separator='_').alias('GameID'))

    # Collect if DataFrame should be returned
    return df_long


class MarchMadnessData:
    def __init__(self, stats_path: str = "data/combined_stats.parquet"):
        self.stats_path = stats_path
        self.round_dict = round_dict
        self.data = march_madness_raw

    def load(self):
        self.data = march_madness_data(self.data)
        return self

    def transform(self):
        stats = pl.scan_parquet(self.stats_path)
        # Join season stats to march madness data
        mm = (
            self.data
            .join(stats.rename(lambda x: f'Self_{x}' if x != 'Year' else x), on=['Self_Team', 'Year'], how='inner')
            .join(stats.rename(lambda x: f'Opp_{x}' if x != 'Year' else x), on=['Opp_Team', 'Year'], how='inner')
            .drop('Self_Team', 'Opp_Team', 'Opp_Score', 'OT', 'Self_Region', 'Opp_Region')
            .rename({'Self_Score': 'Target_Score'})
        )

        mm = mm.rename(lambda x: x.replace('Self_', ''))

        # Fetch column names
        cnames = mm.collect_schema().names()

        # Feature eng: create diffs for team v opp stats
        for cname in cnames:
            if '_opp' in cname and cname.replace('_opp', '') in cnames:
                new_name = f'{cname}_diff'
                mm = mm.with_columns(pl.col(cname.replace('_opp', '')).sub(pl.col(cname)).alias(new_name))
                mm = mm.drop(cname)
                # Update field names list
                cnames.remove(cname)
                cnames.append(new_name)

        # Feature eng: create diffs and ratios for team v team matchups
        # Get shared A/B team stats
        shared_cnames = [cname for cname in cnames if f'Opp_{cname}' in cnames]
        # Calculate diffs and products
        mm = (
            mm
            .with_columns([pl.col(cname).sub(pl.col(f'Opp_{cname}')).alias(f'diff_{cname}') for cname in shared_cnames])
            .with_columns([pl.col(cname).pow(2).sub(pl.col(f'Opp_{cname}').pow(2)).alias(f'diffsq_{cname}') for cname in shared_cnames])
            .with_columns([pl.col(cname).mul(pl.col(f'Opp_{cname}')).alias(f'prod_{cname}') for cname in shared_cnames])
            # .drop([cname for cname in shared_cnames if cname != 'Seed'])
            # .drop([f'Opp_{cname}' for cname in shared_cnames if cname != 'Seed'])
        )

        # Save memory with Float32 instead of 64
        mm = mm.with_columns(pl.col(pl.Float64).cast(pl.Float32).name.keep())

        self.data = mm

        return self

    def collect(self):
        return self.data.collect()


team_years = march_madness_data().select('Self_Team', 'Year').rename({'Self_Team': 'Team'}).unique().collect()

years = team_years['Year'].unique()


def get_bart_stats(year: int, session: requests.Session = None) -> pl.LazyFrame:
    """ Get pre-tournament season stats for NCAA tournament teams in a given year.

    These stats are hosted and maintained by Bart Torvik on https://barttorvik.com
    include NCAA tournament games. For the use case of predicting spreads during the tournament, we
    should only have stats for non-NCAA tournament games to prevent data leak during model development.

    Parameters
    ---------
    year: int
        The year of the end of the season of interest, e.g., 2023-24 is referenced by `2024`
    session: requests.Session, optional
        Session to use with requests. If not provided, requests will use one-time use get

    Returns
    -------
    dict
        Average and resilient-average stats for regular season games

    Examples
    --------
    ```{py}
    from src.march_madness_data_score import get_basic_stats

    # Get TCU men's basketball season stats from 2024-2025:
    df = get_basic_stats('texas-christian', 2024)

    print(df.collect())
    ```
    """
    ss = get_selection_sunday(year).strftime('%Y%m%d')
    url = f'https://barttorvik.com/timemachine/team_results/{ss}_team_results.json.gz'
    # url = f'https://barttorvik.com/timemachine/team_results/{year}0314_team_results.json.gz'

    # Get HTML content (can also be from a local file)
    if session:
        res = session.get(url)
    else:
        res = requests.get(url)
    # Raise an error if status code is not 200
    res.raise_for_status()
    # Create polars lazyframe by parsing json
    df = pl.LazyFrame(res.json(),
                      schema=['rk', 'team', 'conf', 'record', 'adjoe', 'oe_rank', 'adjde', 'de_rank',
                              'barthag', 'barthag_rank', 'proj_w', 'proj_l', 'proj_conf_w', 'proj_conf_l',
                              'conf_rec', 'sos', 'nconf_sos', 'conf_sos', 'proj_sos', 'proj_nconf_sos',
                              'proj_conf_sos', 'elite_sos', 'elite_nconf_sos', 'opp_oe', 'opp_de',
                              'proj_opp_oe', 'proj_opp_de', 'conf_adj_oe', 'conf_adj_de', 'qual_o',
                              'qual_d', 'qual_barthag', 'qual_games', 'fun', 'conf_pf', 'conf_pa',
                              'conf_poss', 'conf_oe', 'conf_de', 'conf_sos_remain', 'conf_win%',
                              'wab', 'wab_rk', 'fun_rk', 'adjt'],
                      orient='row')

    df = (
        df
        # Set year as a field
        .with_columns(pl.lit(year).alias('Year'))
        # Get team name as slug
        .with_columns(pl.col('team').map_elements(get_team_slug, pl.String).alias('Team'))
        # Get Quality Games as a fraction of total games
        .with_columns(
            pl.col('record').str.extract_all(r'[0-9]+')
            .list.to_struct(fields=["w", "l"])
            .alias("record_struct")
        )
        .with_columns(
            pl.col('record_struct').struct.field('w').cast(pl.UInt8)
            .add(pl.col('record_struct').struct.field('l').cast(pl.UInt8))
            .alias('g_played')
        )
        .with_columns(pl.col('qual_games').truediv(pl.col('g_played')).alias('qual_games_ratio'))
        .drop('g_played', 'record_struct')
        # Drop unneeded cols
        .drop('conf_sos_remain', 'proj_w', 'proj_l', 'proj_conf_w',
              'proj_conf_l', 'conf_rec', 'record', 'team')
    )

    # Validate names based on sports-ref data
    vnames = combine('data/season_stats/basic_stats').filter(pl.col('Year').eq(year)).select('Team')
    vnames = vnames.collect()['Team'].to_numpy()
    df_names = df.select('Team').collect()['Team'].to_list()
    vnames_validation = np.isin(vnames, df_names)
    if not all(vnames_validation):
        nonames = vnames[~vnames_validation]
        # Match each DF1 slug to closest DF2 slug
        fmatches = {rf.process.extractOne(noname, df_names)[0]:   noname for noname in nonames}
        print(f'Fixing names with fuzzy matching:\n{fmatches}')
        df = df.with_columns(
            pl.Series([fmatches.get(df_name, df_name) for df_name in df_names]).alias('Team')
        )

    return df


def get_basic_stats(team: str,
                    year: int,
                    save_game: bool = False,
                    session: requests.Session = None) -> pl.LazyFrame:
    """ Scrapes game logs for a team in a given year and returns regular season averages.

    Scraping these stats from game data is necessary given provided season stats on Sports-Reference
    include NCAA tournament games. For the use case of predicting spreads during the tournament, we
    should only have stats for non-NCAA tournament games to prevent data leak during model development.

    Parameters
    ---------
    team: str
        Team name in format of valid sports-reference slug, e.g., "kansas"
    year: int
        The year of the end of the season of interest, e.g., 2023-24 is referenced by `2024`
    save_game: bool, default = False
        If True, saves the game files (before summarizing) in the game_stats directory
    session: requests.Session, optional
        Session to use with requests. If not provided, requests will use one-time use get

    Returns
    -------
    dict
        Average and resilient-average stats for regular season games

    Examples
    --------
    ```{py}
    from src.march_madness_data_score import get_basic_stats

    # Get TCU men's basketball season stats from 2024-2025:
    df = get_basic_stats('texas-christian', 2024)

    print(df.collect())
    ```
    """
    url = f"https://www.sports-reference.com/cbb/schools/{team}/men/{year}-gamelogs.html"
    # Set headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Connection': 'close'
    }

    # Get HTML content (can also be from a local file)
    if session:
        res = session.get(url, headers=headers)
    else:
        res = requests.get(url, headers=headers)
    # Raise an error if status code is not 200
    res.raise_for_status()
    # Get result text as parsed html
    soup = BeautifulSoup(res.text, 'html.parser')
    # Find the table
    table = soup.find("table", {"id": "team_game_log"})

    # Extract headers
    all_rows = table.find_all("tr")
    header_row = all_rows[1]
    headers_og = [th.get_text(strip=True) for th in header_row.find_all("th")]
    # Rename and filter the headers
    headers = []
    opp_count = 0
    for h in headers_og:
        h = h.strip()
        if h == "":
            h = "Location"
        elif h == "Rk":
            continue
        elif h == "Opp":
            if opp_count == 0:
                h = "Opp_name"
            else:
                h = "Opp_score"
            opp_count += 1
        elif h in headers:
            h += '_opp'
        headers.append(h)

    # Extract rows
    rows = []
    for tr in table.find_all("tr")[2:]:  # Skip header rows
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cells:
            rows.append(cells)

    # Create Polars LazyFrame
    df = pl.LazyFrame(rows, schema=headers, orient='row')

    # Filter out rows
    df = (
        df
        # Filter regular season and conference tournament games only
        .filter(pl.col("Type").is_in(["REG (Conf)", "REG (Non-Conf)", "CTOURN"]))
        # Filter games with no stats recorded
        .filter(pl.col('Gtm').ne(''))
    )

    if save_game:
        df.collect().write_csv(f'data/game_stats/basic_stats/{year}_{team}.csv')

    # Convert columns types
    bool_cols = ['Rslt', 'OT']
    other_cols = ['Gtm', 'Date', 'Location', 'Opp_name', 'Type']
    float_cols = ['FG%', '3P%', '2P%', 'eFG%', 'FT%',
                  'FG%_opp', '3P%_opp', '2P%_opp', 'eFG%_opp', 'FT%_opp']
    # The rest are int
    df = (
        df
        .drop(other_cols)
        .with_columns(
            # Float cols
            *[pl.col(col).replace('', '0').cast(pl.Float64) for col in float_cols],
            # Bool cols
            pl.col("Rslt").eq("W").cast(pl.Int8),
            pl.col("OT").str.contains("OT", literal=True).cast(pl.Int8),
            # Convert all other cols to int
            *[pl.col(col).cast(pl.Int16) for col in headers
              if col not in float_cols + bool_cols + other_cols]
        )
        # Add feature: score differential
        .with_columns((pl.col('Tm') - pl.col('Opp_score')).alias('Spread'))
    )

    # Compute outlier-resilient averages and variance
    res_stats = (
        pl.concat(
            [res_mean(df, col) for col in df.collect_schema().names()],
            how='horizontal'
        )
    )

    return res_stats


def get_advanced_stats(team: str,
                       year: int,
                       save_game: bool = False,
                       session: requests.Session = None) -> pl.LazyFrame:
    """ Scrapes game logs for a team in a given year and returns regular season averages.

    Scraping these stats from game data is necessary given provided season stats on Sports-Reference
    include NCAA tournament games. For the use case of predicting spreads during the tournament, we
    should only have stats for non-NCAA tournament games to prevent data leak during model development.

    Parameters
    ---------
    team: str
        Team name in format of valid sports-reference slug, e.g., "kansas"
    year: int
        The year of the end of the season of interest, e.g., 2023-24 is referenced by `2023`
    save_game: bool, default = False
    session: requests.Session, optional
        Session to use with requests. If not provided, requests will use one-time use get

    Returns
    -------
    dict
        Average and resilient-average stats for regular season games

    Examples
    --------
    ```{py}
    from src.march_madness_data_score import get_advanced_stats

    # Get TCU men's basketball season stats from 2024-2025:
    df = get_advanced_stats('texas-christian', 2024)

    print(df.collect())
    ```
    """
    url = f"https://www.sports-reference.com/cbb/schools/{team}/men/{year}-gamelogs-advanced.html"
    # Pretend we're a user
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Connection': 'close'
    }
    # Get HTML content (can also be from a local file)
    if session:
        res = session.get(url, headers=headers)
    else:
        res = requests.get(url, headers=headers)
    # Raise an error if status code is not 200
    res.raise_for_status()
    # Get result text as parsed html
    soup = BeautifulSoup(res.text, 'html.parser')
    # Find the table
    table = soup.find("table", {"id": "team_advanced_game_log"})

    # Extract headers
    all_rows = table.find_all("tr")
    header_row = all_rows[1]
    headers_og = [th.get_text(strip=True) for th in header_row.find_all("th")]
    # Rename and filter the headers
    headers = []
    opp_count = 0
    for h in headers_og:
        h = h.strip()
        if h == "":
            h = "Location"
        elif h == "Rk":
            continue
        elif h == "Opp":
            if opp_count == 0:
                h = "Opp_name"
            else:
                h = "Opp_score"
            opp_count += 1
        elif h in headers:
            h += '_opp'
        headers.append(h)

    # Extract rows
    rows = []
    for tr in table.find_all("tr")[2:]:  # Skip header rows
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cells:
            rows.append(cells)

    # Create Polars LazyFrame
    df = pl.LazyFrame(rows, schema=headers, orient='row')

    # Filter out rows
    df = (
        df
        # Filter regular season and conference tournament games only
        .filter(pl.col("Type").is_in(["REG (Conf)", "REG (Non-Conf)", "CTOURN"]))
        # Filter games with no stats recorded
        .filter(pl.col('Gtm').ne(''))
    )

    if save_game:
        df.collect().write_csv(f'data/game_stats/advanced_stats/{year}_{team}.csv')

    # Convert columns types
    int_cols = ["Tm", "Opp_score"]
    bool_cols = ['Rslt', 'OT']
    other_cols = ['Gtm', 'Date', 'Location', 'Opp_name', 'Type']
    # The rest are float
    df = (
        df
        .drop(other_cols)
        .with_columns(
            # Integer cols
            *[pl.col(col).cast(pl.Int16) for col in int_cols],
            # Bool cols
            pl.col("Rslt").eq("W").cast(pl.Int8),
            pl.col("OT").str.contains("OT", literal=True).cast(pl.Int8),
            # Convert all other cols to float
            *[pl.col(col).replace('', '0').cast(pl.Float64) for col in headers
              if col not in int_cols + bool_cols + other_cols]
        )
        # Add feature: score differential
        .with_columns((pl.col('Tm') - pl.col('Opp_score')).alias('Spread'))
    )

    # Compute outlier-resilient averages and variance
    res_stats = (
        pl.concat(
            [res_mean(df, col) for col in df.collect_schema().names()],
            how='horizontal'
        )
    )

    return res_stats


def update_barts():
    """Try to get Bart team stats for years not yet saved"""
    stats_fp = 'data/season_stats/bart_stats'
    # Create requests session to re-use in loop
    with requests.Session() as session:
        for year in years:
            if year < 2011:
                continue
            elif f'{year}.csv' in os.listdir(stats_fp):
                continue
            else:
                time.sleep(random.uniform(7, 15))
                try:
                    (
                        get_bart_stats(year, session)
                        .collect()
                        .write_csv(f'{stats_fp}/{year}.csv')
                    )
                except Exception as e:
                    print(f"Error fetching {year}:")
                    print(e)


def update_basic_stats(save_game: bool = False):
    """Try to get advanced team stats for team-years not yet saved

    Parameters
    ----------
    save_game: bool, default = False
        Passed to get_basic_stats. Saves game files while getting summary stats if True.
    """
    stats_fp = 'data/season_stats/basic_stats'
    # Create requests session to re-use in loop
    with requests.Session() as session:
        for year, team in team_years.iter_rows():
            if year < 2011:
                continue
            team_slug = get_team_slug(team)
            if f'{year}_{team_slug}.csv' in os.listdir(stats_fp):
                continue
            else:
                time.sleep(random.uniform(7, 15))
                try:
                    stats = get_basic_stats(team_slug, year, session=session, save_game=save_game)
                    (
                        pl.concat(
                            [
                                (pl.LazyFrame({'Year': year, 'Team': team_slug})
                                 .with_columns(pl.col('Year').cast(pl.UInt16))),
                                stats
                            ],
                            how='horizontal'
                        )
                        .collect()
                        .write_csv(f'{stats_fp}/{year}_{team_slug}.csv')
                    )
                except Exception as e:
                    print(f"Error fetching {team} {year}:")
                    print(e)


def update_advanced_stats(save_game: bool = False):
    """Try to get advanced team stats for team-years not yet saved

    Parameters
    ----------
    save_game: bool, default = False
        Passed to get_advanced_stats. Saves game files while getting summary stats if True.
    """
    stats_fp = 'data/season_stats/advanced_stats'
    # Create requests session to re-use in loop
    with requests.Session() as session:
        for year, team in team_years.iter_rows():
            if year < 2011:
                continue
            team_slug = get_team_slug(team)
            if f'{year}_{team_slug}.csv' in os.listdir(stats_fp):
                continue
            else:
                try:
                    time.sleep(random.uniform(3, 7))
                    stats = get_advanced_stats(team_slug, year, session=session, save_game=save_game)
                    (
                        pl.concat(
                            [
                                (pl.LazyFrame({'Year': year, 'Team': team_slug})
                                 .with_columns(pl.col('Year').cast(pl.UInt16))),
                                stats
                            ],
                            how='horizontal'
                        )
                        .collect()
                        .write_csv(f'{stats_fp}/{year}_{team_slug}.csv')
                    )
                except Exception as e:
                    print(f"Error fetching {team} {year}:")
                    print(e)


def combine(*paths: str,
            isdir: bool = True):
    """Join two sets of stats to a given team-year and output as csv

    Stats are joined using `Team` and `Year` field names as keys. Join is done as a full join, preserving all
    rows in both sets of stats. Any duplicate field names will be dropped from data sourced from dir2.

    Parameters
    ----------
    *paths: str
        Path to stats files
    isdir: bool, optional, default = True
        If True, the paths provided in `dir1` & `dir2` will be iterated through, reading in all files in the directory.
        If False, it is assumed the paths provided are the full paths to the csv of interest.

    Returns
    -------
    pl.LazyFrame
        New LazyFrame containing all basic stats joined to advanced stats by team-year

    Examples
    --------
    ```{py}
    from src.march_madness_data_score import combine

    # Combine all basic and advanced stats
    df = combine(
        bas_dir='data/season_stats/basic_stats',
        adv_dir='data/season_stats/advanced_stats'
        lazy=False
    )

    print(df)
    ```
    """
    def __readit(path: str, isdir: bool) -> pl.LazyFrame:
        if isdir:
            files = [os.path.join(path, f) for f in os.listdir(path)
                     if os.path.isfile(os.path.join(path, f))]
        else:
            files = [path]
        read_df = (
            pl.concat([pl.scan_csv(file) for file in files], how='vertical')
            .with_columns(pl.col('Year').cast(pl.UInt16))  # standardize year data type
        )
        return read_df

    # Load and concat the first dataset
    df = __readit(paths[0], isdir)

    # Iteratively join with remaining datasets
    for path in paths[1:]:
        df_next = __readit(path, isdir)
        df = df.join(df_next, on=['Team', 'Year'], how='inner').drop(pl.selectors.ends_with('_right'))

    return df


def combine_stats():
    """Combine all stats into one dataframe, prep, and save

    Files from each directory of downloaded stats are joined with `combine()`, then prepped for
    machine learning with one-hot encoding, i.e., and finally saved to the output dir for use in
    model training.
    """
    df = combine('data/season_stats/bart_stats',
                 'data/season_stats/opp_bart_stats',
                 'data/season_stats/advanced_stats',
                 'data/season_stats/basic_stats',
                 'data/season_stats/last5_stats',
                 'data/player_stats')

    df = df.with_columns(
        # Lump Pac-10 (rip) in with Pac-12 (rip)
        pl.when(pl.col('conf').eq('P10'))
        .then(pl.lit('P12'))
        # Change conferences outside of Power 5, Mountain West, Atlantic 10, West Coast, and American to Other
        .when(pl.col('conf').is_in(['Amer', 'WCC', 'A10', 'MWC', 'P12', 'ACC', 'BE', 'SEC', 'B12', 'B10']))
        .then(pl.col('conf'))
        .otherwise(pl.lit('Other'))
        .alias('conf')
    ).collect()

    # One-hot encode conferences
    df = df.to_dummies('conf', drop_first=False)
    # Drop Other
    df = df.drop('conf_Other')

    # Save combined file
    df.write_parquet('data/combined_stats.parquet')


def join_bart(team: str, year: int | str) -> pl.LazyFrame:
    """Join bart stats to sports-ref stats for game opponents"""
    # Read in team schedule from basic game stats
    df = (
        pl.scan_csv(f'data/game_stats/basic_stats/{year}_{team}.csv')
        .select('Date', 'Opp_name')
        .filter(pl.col('Opp_name').ne('') & pl.col('Opp_name').is_not_null())
        .with_columns(pl.col('Opp_name').map_elements(get_team_slug, pl.String))
    )

    df_bart = pl.scan_csv(f'data/season_stats/bart_stats/{year}.csv')

    # Validate names based on sports-ref data
    df_names = df.select('Opp_name').collect().to_numpy().ravel()
    df_names_unq = np.unique(df_names)
    df_bart_names = df_bart.select('Team').collect().to_numpy().ravel()
    name_validation = np.isin(df_names_unq, df_bart_names)
    if not all(name_validation):
        nonames = df_names_unq[~name_validation]
        # Match each DF1 slug to closest DF2 slug
        fmatches = [(rf.process.extractOne(noname, df_bart_names), noname) for noname in nonames]
        vfmatches = {match[0][0]: match[1] for match in fmatches if match[0][1] > 90}
        for match in fmatches:
            if match[0][2] <= 90:
                print(f'Skipping names even after fuzzy matching:\n{match}')
        df_bart = df_bart.with_columns(
            pl.Series([vfmatches.get(name, name) for name in df_bart_names]).alias('Team')
        )

    df_full = df.join(df_bart, left_on='Opp_name', right_on='Team', how='inner').sort('Date')
    df_full = (
        df_full
        .rename(lambda x: "opp_" + x if x not in ['Date', 'Year', 'Opp_name'] else x)
        .with_columns(pl.lit(team).alias('Team'))
        .select(['Year', 'Team', 'Date', 'Opp_name', pl.selectors.starts_with('opp_')])
    )

    return df_full


def summarize_bart(df: pl.LazyFrame) -> pl.LazyFrame:
    """Summarize (opponent) bart stats for a set of games"""
    yt_cols = ['Year', 'Team']
    year_team = df.select(yt_cols).unique()
    df = df.drop(yt_cols + ['Opp_name', 'opp_conf', 'Date'])

    return pl.concat([year_team, df.mean()], how='horizontal')


def update_opp_barts(save_game=True):
    """Join opponent bart stats to games and get season summaries"""
    stats_fp = 'data/game_stats/opp_bart_stats'
    season_stats_fp = 'data/season_stats/opp_bart_stats'
    for year, team in team_years.iter_rows():
        if year < 2011:
            continue
        team_slug = get_team_slug(team)
        if f'{year}_{team_slug}.csv' in os.listdir(stats_fp):
            continue
        else:
            df = join_bart(team_slug, year)
            if save_game:
                df.collect().write_csv(f'{stats_fp}/{year}_{team_slug}.csv')
            df_mean = summarize_bart(df)
            df_mean.collect().write_csv(f'{season_stats_fp}/{year}_{team_slug}.csv')


def get_last5(team: str, year: int | str):
    """Get basic stat summaries for last 5 games"""
    basic_stat_fp = 'data/game_stats/basic_stats'
    advanced_stat_fp = 'data/game_stats/advanced_stats'
    opp_bart_stat_fp = 'data/game_stats/opp_bart_stats'
    basic_df = pl.scan_csv(f'{basic_stat_fp}/{year}_{team}.csv')
    advanced_df = pl.scan_csv(f'{advanced_stat_fp}/{year}_{team}.csv')
    opp_bart_df = pl.scan_csv(f'{opp_bart_stat_fp}/{year}_{team}.csv')
    # Join the game data
    df_joined = (
        basic_df
        .join(advanced_df, on='Date', how='inner').drop(pl.selectors.ends_with('_right'))
        .join(opp_bart_df, on='Date', how='inner').drop(pl.selectors.ends_with('_right'))
    )
    # Get stats for the last 5 games
    df = (
        df_joined
        .top_k(5, by='Date')
        .with_columns(
            pl.col('Tm').gt(pl.col('Opp_score')).cast(pl.Int8).alias('Rslt'),
            pl.col('Tm').sub(pl.col('Opp_score')).alias('Spread'),
            pl.col('OT').str.contains('OT').cast(pl.Int8)
        )
        .drop('Gtm', 'Date', 'Location', 'Type', 'Year', 'Team', 'opp_conf', 'Opp_name')
        .rename(lambda x: f'last5_{x}')
        .mean()
        .with_columns(pl.all().fill_null(0.0))
        .with_columns(pl.lit(team).alias('Team'), pl.lit(year).alias('Year'))
    )

    return df


def update_last5():
    """Update season stats for team-years in their last 5 games"""
    stats_fp = 'data/season_stats/last5_stats'
    for year, team in team_years.iter_rows():
        if year < 2011:
            continue
        team_slug = get_team_slug(team)
        if f'{year}_{team_slug}.csv' in os.listdir(stats_fp):
            continue
        else:
            get_last5(team_slug, year).collect().write_csv(f'{stats_fp}/{year}_{team_slug}.csv')


def player_game_stats(year: int, session: requests.Session = None) -> pl.LazyFrame:
    """ Get pre-tournament game stats for NCAA players in a given year.

    These stats are hosted and maintained by Bart Torvik on https://barttorvik.com
    include NCAA tournament games. For the use case of predicting spreads during the tournament, we
    should only have stats for non-NCAA tournament games to prevent data leak during model development.

    Parameters
    ---------
    year: int
        The year of the end of the season of interest, e.g., 2023-24 is referenced by `2023`
    session: requests.Session, optional
        Session to use with requests. If not provided, requests will use one-time use get

    Returns
    -------
    pl.LazyFarme
        Player season stats for the given year for all NCAA teams available in the barttorvik data sets

    Examples
    --------
    ```{py}
    from src.march_madness_data_score import get_player_stats

    # Get 2024 men's basketball player stats:
    df = get_player_stats(2024)

    print(df.collect())
    ```
    """
    url = f'https://barttorvik.com/{year}_all_advgames.json.gz'
    # Set headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Connection': 'close'
    }
    # Get HTML content (can also be from a local file)
    if session:
        res = session.get(url, headers=headers)
    else:
        res = requests.get(url, headers=headers)
    # Raise an error if status code is not 200
    res.raise_for_status()
    # Create polars lazyframe by parsing json
    data = res.json()
    df = pl.LazyFrame(data,
                      orient='row',
                      infer_schema_length=None,
                      schema=['numdate', 'datetext', 'opstyle', 'quality', 'win1',
                              'opponent', 'muid', 'win2', 'Min_per', 'ORtg',
                              'Usage', 'eFG', 'TS_per', 'ORB_per', 'DRB_per',
                              'AST_per', 'TO_per', 'dunksmade', 'dunksatt',
                              'rimmade', 'rimatt', 'midmade', 'midatt', 'twoPM',
                              'twoPA', 'TPM', 'TPA', 'FTM', 'FTA', 'bpm_rd', 'Obpm',
                              'Dbpm', 'bpm_net', 'pts', 'ORB', 'DRB', 'AST', 'TOV',
                              'STL', 'BLK', 'stl_per', 'blk_per', 'PF',
                              'possessions', 'bpm', 'sbpm', 'loc', 'tt', 'pp',
                              'inches', 'cls', 'pid', 'year']
                      ).sort(['tt', 'pid'])
    # Get the estimated selection sunday date
    ss = get_selection_sunday(year)
    # Remove march madness games
    df = df.with_columns(pl.col('numdate').str.to_date('%Y%m%d')).filter(pl.col('numdate').lt(ss))
    df = df.drop(['datetext', 'opstyle', 'quality', 'muid', 'win2', 'loc', 'year'])
    # Convert class to numeric (1: Freshman, 2: Sophomore, 3: Junior, 4: Senior)
    df = df.with_columns(
        pl.col('cls')
        .replace({'Fr': 1, 'So': 2, 'Jr': 3, 'Sr': 4})
        .cast(pl.UInt8, strict=False)
        .fill_null(strategy='mean')
    )
    df = (
        df
        # Set year as a field
        .with_columns(pl.lit(year).cast(pl.UInt16).alias('Year'))
        # Get team name as slug
        .with_columns(pl.col('tt').map_elements(get_team_slug, pl.String).alias('Team'))
        .drop('tt')
    )
    return df


def player_game_to_season(year: int |str):
    grouping_cols = ['pp', 'pid', 'Team', 'Year']
    ss = get_selection_sunday(year)
    df = player_game_stats(year).with_columns(Min = pl.col('Min_per').mul(40/100))
    # Identify the players we care about: the top 5 in minutes played over the last 4 weeks
    keep_pid = (
        df
        .filter(pl.col('numdate').ge(ss - datetime.timedelta(weeks=4)))
        .group_by(grouping_cols)
        .sum()
        .select(pl.col('pid').top_k_by(k=5, by='Min').over(['Team', 'Year'], mapping_strategy='explode'))
    ).collect()
    df = df.filter(pl.col('pid').is_in(keep_pid['pid']))
    df = df.drop('opponent', 'numdate')
    df_gp = (
        df
        .with_columns(gp = pl.lit(1))
        .rename({'Min': 'Min_total'})
        .select(grouping_cols+['gp', 'Min_total']).group_by(grouping_cols).sum()
    )
    df_mean = df.group_by(grouping_cols).mean()
    # Join the games played sum
    df_joined = df_mean.join(df_gp, on=grouping_cols, how='inner')
    # create a new feat for total mins played (assuming 40 min games which is not always true)
    df_joined = (
        df_joined
        # Get shot type makes / attempt rates
        .with_columns([
            pl.col(shot_m).truediv(pl.col(shot_a) + 1e-10).alias(f'{shot_m}_pct')
            for shot_m, shot_a in [('rimmade', 'rimatt'), ('midmade', 'midatt'),
                                   ('dunksmade', 'dunksatt'), ('twoPM', 'twoPA'),
                                   ('TPM', 'TPA'), ('FTM', 'FTA'),
                                   ('AST', 'TOV')]  # add in assist/turnover ratio as well
        ])
        # Rename assist to turnover ratio
        .rename({'AST_pct': 'AST_TOV_ratio'})
        .with_columns([
            pl.col(col).truediv(pl.col('Min_total')).mul(40).alias(f'{col}_per40min')
            for col in df_joined.collect_schema().names()
            if col not in grouping_cols + ['cls', 'gp', 'inches', 'win1', 'Min_per', 'Min', 'Min_total']
        ])
    )
    # Keep stats for only top 5 players (by season-long playing time)
    df_selected = df_joined.select(
        pl.all().top_k_by(k=5, by='Min_total').over(['Year', 'Team'], mapping_strategy='explode')
    )
    # Get team averages
    df_final = (
        df_selected
        .drop('pp', 'pid')
        .group_by('Team', 'Year')
        .mean()
    )

    return df_final


def update_player_stats() -> None:
    """ Save pre-tournament player season stats for NCAA teams.

    These stats are hosted and maintained by Bart Torvik on https://barttorvik.com
    include NCAA tournament games. For the use case of predicting spreads during the tournament, we
    should only have stats for non-NCAA tournament games to prevent data leak during model development.

    Returns
    -------
    None
    """
    stats_fp = 'data/player_stats'
    for year in years:
        if year < 2011:
            continue
        elif f'{year}.csv' in os.listdir(stats_fp):
            continue
        else:
            # Get player data and convert it to team data
            df = player_game_to_season(year)
            # Rename cols
            df = df.rename(lambda x: f'player_{x}' if x not in {'Team', 'Year'} else x)
            # Save player data to destination
            df.collect().write_csv(f'{stats_fp}/{year}.csv')
    return None
