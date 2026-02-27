import requests
import polars as pl
from datetime import date, datetime, timedelta
from src.march_madness_data import MarchMadnessData, get_team_slug, round_regex
from src.config import ROUND_NAMES
import joblib


# Get the pipeline
saved_model = joblib.load('model/model.joblib')
PIPELINE = saved_model['pipeline']

ESPN_URL = 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard'
DEFAULT_LOGO = 'https://upload.wikimedia.org/wikipedia/commons/a/a3/Image-not-found.png'
TODAY = date.today()

SCORE_SCHEMA = {
    'Year': pl.Utf8,
    'Round': pl.Utf8,
    'W_Region': pl.Utf8,
    'W_Seed': pl.Utf8,
    'W_Team': pl.Utf8,
    'W_Score': pl.Utf8,
    'L_Region': pl.Utf8,
    'L_Seed': pl.Utf8,
    'L_Team': pl.Utf8,
    'L_Score': pl.Utf8,
    'OT': pl.Utf8,
    'GameID': pl.Utf8,
    'Date': pl.Utf8,
    'EventName': pl.Utf8,
    'Status': pl.Utf8,
    'A_Team_Logo': pl.Utf8,
    'B_Team_Logo': pl.Utf8,
}


def __query(url: str) -> dict:
    """Generic requests query which GETs url and returns data as json (dict)"""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def __find_games(data: dict) -> list[dict]:
    """Format March Madness game data from ESPN API data."""
    games = []
    for event in data.get('events', {}):
        comp = event.get('competitions', [{}])[0]
        # season type of 3 is post-season; tournamentID of 22 is the NCAA Tournament
        if event.get('season', {}).get('type', 0) == 3 and comp.get('tournamentId', 0) == 22:
            competitors = comp.get('competitors', [{}, {}])
            team_a = competitors[0]
            team_b = competitors[1]

            event_date = event.get('date')
            if event_date is not None:
                event_utc = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
                event_cst = event_utc - timedelta(hours=6)  # Convert UTC to Central
                event_date = event_cst.strftime('%Y-%m-%d')

            games.append({
                'Year': event.get('season', {}).get('year'),
                'Round': comp.get('notes', [{}])[0].get('headline', ''),
                'W_Seed': team_a.get('seed'),
                'W_Team': team_a.get('team', {}).get('shortDisplayName'),
                'W_Score': team_a.get('score'),
                'L_Seed': team_b.get('seed'),
                'L_Team': team_b.get('team', {}).get('shortDisplayName'),
                'L_Score': team_b.get('score'),
                'OT': '0',
                'GameID': event.get('id'),
                'Date': event_date,
                'EventName': event.get(
                    'shortName',
                    f"{team_a.get('team', {}).get('abbreviation')} vs {team_b.get('team', {}).get('abbreviation')}"
                ),
                'Status': comp.get('status', {}).get('type', {}).get('description'),
                'A_Team_Logo': team_a.get('team', {}).get('logo', DEFAULT_LOGO),
                'B_Team_Logo': team_b.get('team', {}).get('logo', DEFAULT_LOGO),
            })

    return games


def find_espn_games(url: str) -> list[dict]:
    """Fetch data from a given ESPN API URL and return list[dict] of events"""
    data = __query(url)
    return __find_games(data)


def find_future_games(start: date, days: int):
    """Fetch a future day's NCAA men's basketball games from ESPN

    Params:
    ------
    start (datetime.date): start day to reference
    days (int): number of days in future from start date to query ESPN for game data
    """
    next_date = (start + timedelta(days=days))
    return find_espn_games(f'{ESPN_URL}?dates={next_date.strftime('%Y%m%d')}')


def get_next_games() -> pl.DataFrame:
    """Fetch today's NCAA men's basketball games from ESPN."""
    games = find_espn_games(ESPN_URL)
    days = 0
    while not games and days < 6:
        days += 1
        games = find_future_games(TODAY, days)
    # Get the next day's games as well
    days += 1
    games = games + find_future_games(TODAY, days)

    df = pl.DataFrame(games, schema=SCORE_SCHEMA).with_columns(round_regex('Round').alias('Round'))

    return df


def predict_bracket(year):
    year = str(year) if isinstance(year, int) else year  # ensure year is a string
    
    po_dfs = []
    for i in range(4):
        rnd_df = (
            pl.LazyFrame({
                'Round': ROUND_NAMES[i],
                'Seed': list(range(17)),
            })
        )

        if i == 0:
            all_seed_expr = pl.concat_list([pl.col('Seed'), 17 - pl.col('Seed')])
        elif i == 1:
            min_seed = ((17 - (2 * pl.col('Seed') - 17).abs()) / 2).cast(pl.Int64)
            all_seed_expr = pl.concat_list([min_seed, 17 - min_seed, 9 - min_seed, 8 + min_seed])
        elif i == 2:
            seed_half_1 = [1, 4, 5, 8, 9, 12, 13, 16]
            seed_half_2 = [2, 3, 6, 7, 10, 11, 14, 15]
            all_seed_expr = (
                pl.when(pl.col('Seed').is_in(seed_half_1))
                .then(pl.lit(seed_half_1))
                .otherwise(pl.lit(seed_half_2))
            )
        else:
            all_seed_expr = pl.lit(list(range(17)))

        rnd_df = rnd_df.with_columns(
            all_seed_expr
            .list.sort()
            .alias('All_seeds')
        )

        po_dfs += [rnd_df]

    po_df = pl.concat(po_dfs)

    mm = MarchMadnessData().data.filter(pl.col('Year').eq(year))

    df = (
        mm
        .with_columns(round_regex('Round').alias('Round'))
        .with_columns(
            pl.cum_count('Year').cast(pl.String).alias('GameID'),
            pl.lit(None).alias('Date'),
            pl.lit(None).alias('EventName'),
            pl.lit(None).alias('Status'),
            pl.lit(None).alias('A_Team_Logo'),
            pl.lit(None).alias('B_Team_Logo')
        )
        .select(SCORE_SCHEMA.keys())
        .cast(SCORE_SCHEMA)
    )

    first4 = df.filter(pl.col('Round').eq('First Four'))
    df = df.filter(pl.col('Round').eq('Round of 64'))

    if df.select(pl.len()).collect().item() == 0:
        return None

    dfs = []

    if (
            first4.select(pl.len()).collect().item() > 0  # there are first 4 games in the df
            and df.select('W_Score').drop_nulls().select(pl.len()).collect().item() < 32  # R64 is not completed yet
    ):
        pred = predict_next_games(first4)  # predict scores
        if pred is not None:
            pred = pred.with_columns(
                pl.when(pl.col('A_Pred_Score').gt(pl.col('B_Pred_Score')))
                .then(True)
                .otherwise(False)
                .alias('A_Team_Win')
            )

        round_po = po_df.filter(pl.col('Round').eq('Round of 64')).drop('Round')

        pred = (
            pred
            .with_columns(
                pl.when(pl.col('A_Team_Win'))
                .then(
                    pl.struct([
                        pl.col('A_Region').alias('f4_Region'),
                        pl.col('A_Seed').alias('f4_Seed'),
                        pl.col('A_Team').alias('f4_Team')
                    ])
                )
                .otherwise(
                    pl.struct([
                        pl.col('B_Region').alias('f4_Region'),
                        pl.col('B_Seed').alias('f4_Seed'),
                        pl.col('B_Team').alias('f4_Team')
                    ])
                )
                .alias('f4')
            )
            .join(round_po, left_on='A_Seed', right_on='Seed')
            .explode('All_seeds')
            .filter(pl.col('A_Seed').ne(pl.col('All_seeds')))
            # A_Region == B_Region and A_Seed == B_Seed, always
            .with_columns(pl.concat_str(pl.col('A_Region'), pl.lit('_'), pl.col('All_seeds')).alias('shared_key'))
            .select('shared_key', 'f4')
            .unnest('f4')
        )

        # Join predicted first 4 winners to the bracket
        df = (
            df
            .with_columns(
                pl.concat_str(pl.col('W_Region'), pl.lit('_'), pl.col('W_Seed')).alias('key_L'),
                pl.concat_str(pl.col('L_Region'), pl.lit('_'), pl.col('L_Seed')).alias('key_W')
            )
            .join(pred, left_on='key_L', right_on='shared_key', how='left')
            .join(pred, left_on='key_W', right_on='shared_key', suffix='_W', how='left')
            .with_columns(
                pl.coalesce('f4_Region', 'L_Region').alias('L_Region'),
                pl.coalesce('f4_Seed', 'L_Seed').alias('L_Seed'),
                pl.coalesce('f4_Team', 'L_Team').alias('L_Team'),
                pl.coalesce('f4_Region_W', 'W_Region').alias('W_Region'),
                pl.coalesce('f4_Seed_W', 'W_Seed').alias('W_Seed'),
                pl.coalesce('f4_Team_W', 'W_Team').alias('W_Team')
            )
            .drop('^f4_.+$', '^key_.$')
        )
    # END if first4 statement

    all_cols = df.collect_schema().keys()
    generic_cols = [col.replace('W_', '') for col in all_cols if 'W_' in col]
    w_cols = ['A_' + col for col in generic_cols]
    l_cols = ['B_' + col for col in generic_cols]

    bracket_order_dict = {
        '1.4.5.8.9.12.13.16': '1',
        '2.3.6.7.10.11.14.15': '2',
        '1.8.9.16': '1',
        '4.5.12.13': '2',
        '3.6.11.14': '3',
        '2.7.10.15': '4',
        '1.16': '1',
        '8.9': '2',
        '5.12': '3',
        '4.13': '4',
        '6.11': '5',
        '3.14': '6',
        '7.10': '7',
        '2.15': '8',
    }

    def __key_expr(prefix: str = '', round_col: str = 'Round'):
        return (
            pl.when(pl.col(round_col) == ROUND_NAMES[-1])
            .then(pl.lit('5'))
            .when(pl.col(round_col) == ROUND_NAMES[-2])
            .then(pl.when(pl.col(f'{prefix}Region').lt(3)).then(pl.lit('412')).otherwise(pl.lit('434')))
            .when(pl.col(round_col) == ROUND_NAMES[-3])
            .then(pl.concat_str(pl.lit('3'), pl.col(f'{prefix}Region')))
            .when(pl.col(round_col).is_in(ROUND_NAMES[-6:-3]))
            .then(
                pl.concat_str([
                    pl.col(round_col).replace({ROUND_NAMES[-6]: '0', ROUND_NAMES[-5]: '1', ROUND_NAMES[-4]: '2'}),
                    pl.col(f'{prefix}Region'),
                    pl.lit('_'),
                    (
                        pl.col('All_seeds')
                        .list.eval(pl.element().cast(pl.String))
                        .list.join('.')
                        .replace_strict(bracket_order_dict, default='1')
                    )
                ])
            )
        )

    df_next = df
    for i in range(len(ROUND_NAMES)):
        next_round = ROUND_NAMES[min(i+1, 5)]
        # Predict this round's scores
        pred = (
                   predict_next_games(df_next.collect())
                   .with_columns(pl.lit(next_round).alias('Next_Round'))
                   .join(
                       po_df.select('Round', 'Seed', 'All_seeds'),
                       left_on=['Next_Round', 'A_Seed'], right_on=['Round', 'Seed'], how='left'
                   )
                   .with_columns(
                       pl.when(pl.col('A_Pred_Score').gt(pl.col('B_Pred_Score')))
                       .then(True)
                       .otherwise(False)
                       .alias('A_Team_Win'),
                       __key_expr('A_', 'Next_Round').alias('shared_key'),
                       # __key_expr('B_', 'Next_Round').alias('B_key'),
                   )
                   .drop('Next_Round', 'All_seeds')
        )

        dfs += [pred]

        if i < len(ROUND_NAMES)-1:
            next = (
                pred.lazy()
                .with_columns(
                    pl.when(pl.col('A_Team_Win'))
                    .then(pl.struct(w_cols).struct.rename_fields(generic_cols).alias('Winner'))
                    .otherwise(pl.struct(l_cols).struct.rename_fields(generic_cols).alias('Winner'))
                )
                .drop(w_cols + l_cols + ['A_Pred_Score', 'B_Pred_Score'])
                .unnest('Winner')
                .with_columns(pl.lit(ROUND_NAMES[i+1]).alias('Round'))
            )

            df_next = (
                next
                .join(next.select(generic_cols + ['shared_key']), on='shared_key', suffix='_L')
                .filter(pl.col('Team').ne(pl.col('Team_L')))
                .group_by('shared_key').agg(pl.all().sort_by(pl.col('Seed')).first())
                .rename({
                    'Region': 'W_Region',
                    'Seed': 'W_Seed',
                    'Team': 'W_Team',
                    'Score': 'W_Score',
                    'Region_L': 'L_Region',
                    'Seed_L': 'L_Seed',
                    'Team_L': 'L_Team',
                    'Score_L': 'L_Score',
                })
                .select(list(all_cols))
                .with_columns(pl.all().cast(pl.String))
            )

    combined_df = pl.concat(dfs)

    # Get all scores from completed games
    wl_scores = (
        mm
        .with_columns(round_regex('Round').alias('Round'))
        .filter(pl.col('W_Score').is_not_null() | pl.col('L_Score').is_not_null())
        .select('Round', '^[WL]_Region$', '^[WL]_Seed$', '^[WL]_Team$', '^[WL]_Score$')
        .with_columns(
            pl.when(pl.col('W_Score').str.to_integer() > pl.col('L_Score').str.to_integer())
            .then(pl.col('W_Team'))
            .otherwise(pl.col('L_Team'))
            .alias('winner'),
            pl.when(pl.col('W_Score').str.to_integer() < pl.col('L_Score').str.to_integer())
            .then(pl.col('W_Team'))
            .otherwise(pl.col('L_Team'))
            .alias('loser')
        )
    )
    all_scores = pl.concat([
        wl_scores
        .select('Round', f'{pre}Team', f'{pre}Score')
        .rename({f'{pre}Team': 'Team', f'{pre}Score': 'Score'})
        for pre in ['W_', 'L_']
    ])

    # Join scores to predicted outcomes
    schema = combined_df.collect_schema().keys()
    combined_df = (
        combined_df
        .drop('A_Score', 'B_Score')
        .join(all_scores.rename({'Score': 'A_Score', 'Team': 'A_Team'}), on=['Round', 'A_Team'], how='left')
        .join(all_scores.rename({'Score': 'B_Score', 'Team': 'B_Team'}), on=['Round', 'B_Team'], how='left')
        .select(schema)
    )

    # Get the winning team in each round's matchups
    w_teams = (
        wl_scores
        .select('Round', 'winner')
        .with_columns(pl.lit(True).alias('Prediction_Correct'))
    )

    # Get predicted winner vs actual winner fields
    combined_df = (
        combined_df
        .with_columns(
            pl.when(pl.col('A_Team_Win'))
            .then(pl.col('A_Team'))
            .otherwise(pl.col('B_Team'))
            .alias('Pred_Winner'),
            pl.when(pl.col('A_Team_Win'))
            .then(pl.col('B_Team'))
            .otherwise(pl.col('A_Team'))
            .alias('Pred_Loser'),

        )
        .join(w_teams, left_on=['Round', 'Pred_Winner'], right_on=['Round', 'winner'], how='left')
        .with_columns(pl.col('Prediction_Correct').fill_null(False))
    )

    # Determine game keys
    scores_exist = (
        wl_scores
        .with_columns(pl.col('W_Region').str.to_integer(), pl.lit(True).alias('game_played'))
        .join(
            po_df.with_columns(pl.col('Seed').cast(pl.String)).select('Round', 'Seed', 'All_seeds'),
            left_on=['Round', 'W_Seed'], right_on=['Round', 'Seed'], how='left'
        )
        .with_columns(__key_expr('W_').alias('GameID'))
        .select('GameID', 'winner', 'loser', 'game_played')
    )

    # Join the winner/loser/game played fields and sort the df by GameID
    combined_df_sorted = (
        combined_df
        .join(po_df.select('Round', 'Seed', 'All_seeds'),
              left_on=['Round', 'A_Seed'], right_on=['Round', 'Seed'], how='left')
        .with_columns(__key_expr('A_').alias('GameID')).drop('All_seeds')
        .join(scores_exist, on='GameID', how='left')
        .with_columns(pl.col('game_played').fill_null(False))
        .sort('GameID')
    )

    return combined_df_sorted.collect()


def predict_next_games(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | None:
    """
    Get upcoming game data from ESPN and run data through the score predictor model.

    Params
    ------
    df (pl.DataFrame):
        Polars DataFrame with a schema matching the SCORE_SCHEMA-specified schema.
        Should be an output of `get_next_games()`.

    Returns
    -------
    pl.DataFrame
    """
    if isinstance(df, pl.DataFrame):
        if df.height == 0:
            return None
        df = df.lazy()
    elif df.select(pl.len()).collect().item() == 0:
        return None

    X = MarchMadnessData()
    X.data = df
    X.load()
    X.transform()
    data = X.collect().drop('Target_Score')
    predictions = PIPELINE.predict(data)
    pred_df = (
        pl.LazyFrame({'ID': data['GameID'], 'Pred_Score': predictions})
        .with_columns(pl.col('Pred_Score').round(1).cast(pl.String))
    )
    return (
        df
        .with_columns(
            pl.concat_str([
                pl.col('GameID'),
                pl.lit('_'),
                pl.col(f'{col}_Team').map_elements(get_team_slug, return_dtype=pl.Utf8)
                ]).alias(f'{col}_ID')
            for col in ['W', 'L']
        )
        .with_columns(pl.col('Date').str.to_date('%Y-%m-%d'))
        .join(pred_df.rename(lambda x: 'W_' + x), on='W_ID')
        .join(pred_df.rename(lambda x: 'L_' + x), on='L_ID')
        .drop('W_ID', 'L_ID')
        .rename(lambda x: x.replace('W_', 'A_').replace('L_', 'B_'))
        .sort('Date')
        .cast({
            'Year': pl.UInt16,
            'A_Seed': pl.UInt8,
            'A_Region': pl.UInt8,
            'A_Score': pl.UInt8,
            'B_Seed': pl.UInt8,
            'B_Region': pl.UInt8,
            'B_Score': pl.UInt8,
        })
    )
