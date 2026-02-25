import requests
import polars as pl
from datetime import date, datetime, timedelta
from src.march_madness_data import MarchMadnessData, get_team_slug, round_dict, round_regex
import joblib


# Get the pipeline
saved_model = joblib.load('model/model.joblib')
PIPELINE = saved_model['pipeline']

ESPN_URL = 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard'
DEFAULT_LOGO = 'https://upload.wikimedia.org/wikipedia/commons/a/a3/Image-not-found.png'
TODAY = date.today() - timedelta(days=340)

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

# Possible opponents by round:
def possible_opponents(round_name: str, as_string: bool = False) -> dict:
    """Return possible opponent seeds in a 16-team region"""

    if round_name == 'Round of 64':
        if as_string:
            return {str(i): [str(17 - i)] for i in range(1, 17)}
        return {i: [17 - i] for i in range(1, 17)}

    bracket_order = [
        1, 16, 8, 9, 5, 12, 4, 13,
        6, 11, 3, 14, 7, 10, 2, 15
    ]

    rounds = {
        'Round of 32': 4,
        'Sweet 16': 8,
        'Elite 8': 16,
    }

    group_size = rounds[round_name]

    # map seed → position in bracket
    pos = {seed: i for i, seed in enumerate(bracket_order)}

    result = {}

    for seed in range(1, 9):
        p = pos[seed]

        group_start = (p // group_size) * group_size
        group = bracket_order[group_start:group_start + group_size]

        half = group_size // 2

        if p < group_start + half:
            opponents = group[half:]
        else:
            opponents = group[:half]

        if as_string:
            opponents = [str(opp) for opp in opponents]
            result[str(seed)] = opponents
            result[str(17 - seed)] = opponents
        else:
            result[seed] = opponents
            result[17-seed] = opponents

    return dict(sorted(result.items()))


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


def __get_seed_key(seed, po: dict):
    possible_seeds = {str(seed)}
    prev_len = 0
    while prev_len < len(possible_seeds):
        prev_len = len(possible_seeds)
        for s in list(possible_seeds):
            for ps in po[s]:
                possible_seeds.add(ps)
    return '.'.join(sorted(possible_seeds))


def predict_bracket(year):

    year = str(year) if isinstance(year, int) else year  # ensure year is a string

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

    first4 = df.filter(pl.col('Round').eq('First Four')).collect()
    df = df.filter(pl.col('Round').eq('Round of 64')).collect()

    if df.height == 0:
        return None

    dfs = []

    if first4.height > 0:

        pred = (
            predict_next_games(first4)  # predict scores
            .with_columns(
                pl.when(pl.col('A_Pred_Score').gt(pl.col('B_Pred_Score')))
                .then(True)
                .otherwise(False)
                .alias('A_Team_Win')
            )
        )

        dfs += [pred.lazy()]

        for row in pred.iter_rows(named=True):
            po = possible_opponents('Round of 64', True)

            winner = row['A_Team'] if row['A_Team_Win'] else row['B_Team']
            df = df.with_columns(
                pl.when(
                    pl.col('L_Region').eq(str(row['A_Region'])) &
                    pl.col('W_Seed').is_in(po.get(row['A_Seed']))
                )
                .then(pl.lit(winner))
                .otherwise(pl.col('L_Team'))
                .alias('L_Team'),
                pl.when(
                    pl.col('W_Region').eq(str(row['A_Region'])) &
                    pl.col('L_Seed').is_in(po.get(row['A_Seed']))
                )
                .then(pl.lit(winner))
                .otherwise(pl.col('W_Team'))
                .alias('W_Team')
            )

    rounds = ['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final 4', 'National Championship']

    all_cols = df.columns
    w_cols = [col for col in all_cols if 'W_' in col]
    l_cols = [col for col in all_cols if 'L_' in col]
    generic_cols = [x.replace('W_', '') for x in w_cols]
    df_next = df
    for i in range(len(rounds)):
        pred = (
            predict_next_games(df_next)  # predict scores
            .with_columns(
                pl.when(pl.col('A_Pred_Score').gt(pl.col('B_Pred_Score')))
                .then(True)
                .otherwise(False)
                .alias('A_Team_Win')
            )
        )

        dfs += [pred.lazy()]

        if i < len(rounds)-1:
            # Get possible opponents for the next round


            if i < 3:  # Ro64 - S16
                po = possible_opponents(rounds[i + 1], True)
                key_expr = pl.concat_str(
                    pl.col('Seed').map_elements(lambda x: __get_seed_key(x, po), return_dtype=pl.String),
                    pl.lit('-'),
                    pl.col('Region')
                ).alias('key')
            elif i == 3:  # E8
                key_expr = pl.when(pl.col('Region').lt(3)).then(pl.lit('1.2')).otherwise(pl.lit('3.4')).alias('key')
            else:  # F4
                df_saved = df_next
                key_expr = pl.lit('NATTY').alias('key')

            next = (
                pred
                .drop('A_Pred_Score', 'B_Pred_Score')
                .rename(lambda x: x.replace('A_', 'W_') if x not in ['A_Team_Logo', 'A_Team_Win'] else x)
                .rename(lambda x: x.replace('B_', 'L_') if x != 'B_Team_Logo' else x)
                .with_columns(
                    pl.when(pl.col('A_Team_Win'))
                    .then(pl.struct(w_cols).struct.rename_fields(generic_cols).alias('Winner'))
                    .otherwise(pl.struct(l_cols).struct.rename_fields(generic_cols).alias('Winner'))
                )
                .drop('^[WL]_.+$')
                .unnest('Winner')
                .with_columns(pl.lit(rounds[i+1]).alias('Round'))
                .with_columns(key_expr)
            )

            df_next = (
                next
                .join(next.select(generic_cols + ['key']), on='key', suffix='_right')
                .filter(pl.col('Team').ne(pl.col('Team_right')))
                .group_by('key').agg(pl.all().sort_by(pl.col('Seed')).first())
                .rename(lambda x: 'L_' + x.replace('_right', '') if '_right' in x else x)
                .rename(lambda x: 'W_' + x if x in generic_cols else x)
                .select(all_cols)
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

    # Set logic to determine game keys
    key_expr_2 = (
        pl.when(pl.col('Round').is_in(['Round of 64', 'Round of 32', 'Sweet 16']))
        .then(
            pl.concat_str([
                pl.col('Round').replace({'Round of 64': '1', 'Round of 32': '1', 'Sweet 16': '2'}),
                pl.col('W_Region'),
                pl.lit('_'),
                (
                    pl.struct('W_Seed', 'Round')
                    .map_elements(
                        lambda x:  __get_seed_key(x['W_Seed'], possible_opponents(x['Round'], True))
                        if x['Round'] in ['Round of 64', 'Round of 32', 'Sweet 16'] else None,
                        return_dtype=pl.String
                    )
                )
            ])
        )
        .when(pl.col('Round') == 'Elite 8')
        .then(pl.concat_str(pl.lit('3'), pl.col('W_Region')))
        .when(pl.col('Round') == 'Final 4')
        .then(pl.when(pl.col('W_Region').lt(3)).then(pl.lit('412')).otherwise(pl.lit('434')))
        .when(pl.col('Round') == 'National Championship')
        .then(pl.lit('5'))
        .alias('key')
    )

    scores_exist = (
        wl_scores
        .with_columns(pl.col('W_Region').str.to_integer(), pl.lit(True).alias('game_played'))
        .with_columns(key_expr_2)
        .select('key', 'winner', 'loser', 'game_played')
    )
    combined_df = (
        combined_df
        .rename(lambda x: x.replace('A_', 'W_'))
        .with_columns(key_expr_2)
        .rename(lambda x: x.replace('W_', 'A_'))
        .join(scores_exist, on='key', how='left')
        #.drop('key')
        .with_columns(pl.col('game_played').fill_null(False))
    )

    BRACKET_SLOTS = {
        'Round of 64': {
            '1.16': '1',
            '8.9': '2',
            '12.5': '3',
            '13.4': '4',
            '11.6': '5',
            '14.3': '6',
            '10.7': '7',
            '15.2': '8',
        },
        'Round of 32': {
            '1.16.8.9': '1',
            '12.13.4.5': '2',
            '11.14.3.6': '3',
            '10.15.2.7': '4',
        },
        'Sweet 16': {
            '1.12.13.16.4.5.8.9': '1',
            '10.11.14.15.2.3.6.7': '2',
        }
    }

    combined_df_sorted = (
        combined_df
        .with_columns(pl.col('key').str.replace(r'^.+_', '').alias('opp_suff'))
        .with_columns(
            pl.struct('opp_suff', 'Round').map_elements(
                lambda x: BRACKET_SLOTS.get(x['Round'], {}).get(x['opp_suff'], '')
                if x['Round'] in ['Round of 64', 'Round of 32', 'Sweet 16'] else '1',
                pl.String
            ).alias('opp_suff')
        )
        .with_columns(pl.col('key').str.replace(r'_.+', pl.col('opp_suff')))
        .drop('opp_suff')
        .sort('key')
    )

    return combined_df_sorted.collect()


def predict_next_games(df: pl.DataFrame) -> pl.DataFrame | None:
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
    if df.height == 0:
        return None
    else:
        X = MarchMadnessData()
        X.data = df.lazy()
        X.load()
        X.transform()
        data = X.collect().drop('Target_Score')
        predictions = PIPELINE.predict(data)
        pred_df = (
            pl.DataFrame({'ID': data['GameID'], 'Pred_Score': predictions})
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
