import requests
import polars as pl
from datetime import date, datetime, timedelta
from src.march_madness_data import MarchMadnessData, get_team_slug, round_dict
import joblib
from re import sub

# Get the pipeline
saved_model = joblib.load('model/model.joblib')
PIPELINE = saved_model['pipeline']

ESPN_URL = 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard'
DEFAULT_LOGO = 'https://upload.wikimedia.org/wikipedia/commons/a/a3/Image-not-found.png'
TODAY = date.today()

ROUND_RAW_DICT = {
    'First Four': 'First Four',
    '1st Round': 'Round of 64',
    '2nd Round': 'Round of 32',
    'Sweet 16': 'Sweet 16',
    'Elite 8': 'Elite 8',
    'Final Four': 'Final 4',
    'National Championship': 'National Championship',
}

SCORE_SCHEMA = {
    'Year': pl.Utf8,
    'Round': pl.Utf8,
    'W_Seed': pl.Utf8,
    'W_Team': pl.Utf8,
    'W_Score': pl.Utf8,
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

            round_raw = comp.get('notes', [{}])[0].get('headline', '')
            round_cleaned = next(
                (val for key, val in ROUND_RAW_DICT.items() if key in round_raw),
                sub(r'^.+- ', '', round_raw)
            )

            event_date = event.get('date')
            if event_date is not None:
                event_utc = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
                event_cst = event_utc - timedelta(hours=6)  # Convert UTC to Central
                event_date = event_cst.strftime('%Y-%m-%d')

            games.append({
                'Year': event.get('season', {}).get('year'),
                'Round': round_dict.get(round_cleaned, 0),
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

    return pl.DataFrame(games, schema=SCORE_SCHEMA)


def predict_next_games() -> pl.DataFrame:
    """
    Get upcoming game data from ESPN and run data through the score predictor model.

    Returns
    -------
    pl.DataFrame
    """
    scores = get_next_games()
    if scores.height == 0:
        return None
    else:
        X = MarchMadnessData()
        X.data = scores.lazy()
        X.load()
        X.transform()
        data = X.collect().drop('Target_Score')
        predictions = PIPELINE.predict(data)
        pred_df = (
            pl.DataFrame({'ID': data['GameID'], 'Pred_Score': predictions})
            .with_columns(pl.col('Pred_Score').round().cast(pl.Int16))
        )
        return (
            scores
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
        )
