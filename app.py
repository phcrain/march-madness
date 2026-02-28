from shiny import App, ui, render, reactive
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import floor
from io import BytesIO
from matplotlib.ticker import PercentFormatter
from src.march_madness_data import march_madness_data, round_dict
from src.todays_scores import predict_next_games, get_next_games, predict_bracket
from src.config import ROUND_NAMES
import time

df = (
    march_madness_data()
    .filter(pl.col('Self_Score').gt(pl.col('Opp_Score')))
    .with_columns(pl.col(f'{pre}_Score').mod(10).alias(f'{pre}_Last_Digit') for pre in ['Self', 'Opp'])
    .with_columns(pl.col('Round').replace_strict({v: k for k, v in round_dict.items()}, default=''))
    .collect()
)
n = df.shape[0]
year = df.select(pl.max('Year')).item()

def team_row(row, team_prefix: str):
    """Render team-row UI for bracket display"""
    eliminated = False

    team = row[f'{team_prefix}_Team']
    actual_team = row[f'{team_prefix}_Actual_Team']
    seed = row[f'{team_prefix}_Seed']
    actual_seed = row[f'{team_prefix}_Actual_Seed']
    score = row[f'{team_prefix}_Score']
    actual_score = row[f'{team_prefix}_Actual_Score']

    pred_score = row[f'{team_prefix}_Pred_Score']

    pred_winner = row['Pred_Winner']
    pred_loser = row['Pred_Loser']
    actual_winner = row['winner']
    actual_loser = row['loser']
    game_played = row['game_played']
    pred_correct = row['Prediction_Correct']

    classes = []
    elim_incorrect = ''
    elim_classes = []

    # determine "winner"
    if game_played and actual_winner:
        true_winner = actual_winner
    else:
        true_winner = pred_winner

    if team == true_winner or actual_team == true_winner:
        classes.append('winner-bold')

    if game_played and actual_loser:
        true_loser = actual_loser
    else:
        true_loser = pred_loser

    # add classes for shading prediction accuracy
    if game_played and team == pred_winner:
        if pred_correct is True:
            classes.append('correct')
        elif pred_correct is False:
            classes.append('incorrect')
            elim_classes.append('incorrect')
            elim_incorrect = 'team incorrect'

    # future elimination detection
    if (
            actual_winner is not None
            and team != true_winner
            and team != true_loser
    ):
        eliminated = True
        elim_classes.append('strike')
        elim_classes.append('eliminated')
        classes.append('eliminated')

    class_str = ' '.join(classes)

    if eliminated:
        elim_class_str = ' '.join(elim_classes)
        return ui.div(
            {'class': elim_incorrect},
            ui.div(
                {'class': f'team {elim_class_str}'},
                ui.div(f'{seed} {team}', class_='team-name'),
                ui.div(f'Pred: {pred_score}' if pred_score is not None else '', class_='team-score'),
            ),
            ui.div(
                {'class': f'team {class_str}'},
                ui.div(f'{actual_seed} {actual_team}', class_='team-name'),
                ui.div(f'Score: {actual_score}' if actual_score is not None else '', class_='team-score'),
            ),
        )

    return ui.div(
        {'class': f'team {class_str}'},
        ui.div(f'{seed} {team}', class_='team-name'),
        ui.div(
            [
                f'Score: {score}  | ' if score is not None else '',
                f'Pred: {pred_score}' if pred_score is not None else ''
            ],
            class_='team-score',
        ),
    )


def format_rounds(selected_rounds):
    """Format selected rounds by collapsing consecutive rounds into a range."""

    if selected_rounds is None:
        return ''

    # Sort selected rounds
    selected_rounds = sorted(selected_rounds, key=lambda x: ROUND_NAMES.index(x))

    if selected_rounds == ROUND_NAMES:
        return ''

    formatted_rounds = []
    temp_range = [selected_rounds[0]]

    for i in range(1, len(selected_rounds)):
        prev_round = selected_rounds[i - 1]
        current_round = selected_rounds[i]

        if ROUND_NAMES.index(current_round) == ROUND_NAMES.index(prev_round) + 1:
            temp_range.append(current_round)
        else:
            if len(temp_range) >= 3:
                formatted_rounds.append(f'{temp_range[0]} - {temp_range[-1]}')
            else:
                formatted_rounds.extend(temp_range)
            temp_range = [current_round]

    # Handle the last range
    if len(temp_range) >= 3:
        formatted_rounds.append(f'{temp_range[0]} - {temp_range[-1]}')
    else:
        formatted_rounds.extend(temp_range)

    if formatted_rounds:
        return f"\n({', '.join(formatted_rounds)})"

    return ''


def heatmap_df(df: pl.DataFrame):
    freq_table = df.group_by(['Self_Last_Digit', 'Opp_Last_Digit']).agg(
        pl.len().alias('Count')
    ).sort(by=['Self_Last_Digit', 'Opp_Last_Digit'])

    freq_table = freq_table.with_columns(
        (pl.col('Count') / df.shape[0]).alias('Probability')
    )

    all_digits = pl.DataFrame({
        'Self_Last_Digit': np.tile(np.arange(10), 10),
        'Opp_Last_Digit': np.repeat(np.arange(10), 10),
    })
    freq_table = all_digits.join(freq_table, on=['Self_Last_Digit', 'Opp_Last_Digit'], how='left').fill_null(0)

    output_df = freq_table.pivot(
        values='Probability',
        index='Opp_Last_Digit',
        on='Self_Last_Digit'
    )

    return output_df.to_pandas().set_index('Opp_Last_Digit')


def heatmap(df: pl.DataFrame, round_filter: list = None, fontsize: None | int = None, **kwargs):

    fig, ax = plt.subplots(figsize=(24, 24), dpi=100)  # Init fig and ax
    # Create heatmap
    ax = sns.heatmap(
        df, cmap='coolwarm',
        fmt='', linewidths=0.5, center=0.01,
        cbar_kws={'location': 'bottom',
                  'format': PercentFormatter(1, 2),
                  'ticks': [np.min(df), 0.01, np.max(df)]},
        **kwargs
    )
    ax.tick_params(axis='both', labelsize=fontsize)  # Adjust tick font size
    ax.set_xlabel('Winning Team Last Digit')  # Add axes labels
    ax.set_ylabel('Losing Team Last Digit')  # Add axes labels

    # Add title
    title = 'Squares Probability Heatmap'  # Init title
    if round_filter:
        title += format_rounds(round_filter)
    ax.set_title(title)

    return fig, ax


def winner(score_x, score_y):
    if score_x > score_y:
        return 'winner'
    return ''


# Add page title and sidebar
app_ui = ui.page_navbar(

    # Heatmap
    ui.nav_panel(
        'Squares Heatmap',
        ui.page_sidebar(
            ui.sidebar(
                # your existing controls unchanged
                ui.input_switch('annot', 'Display Frequencies', value=False),
                ui.panel_conditional(
                    'input.annot',
                    ui.input_slider(
                        'annot_digits',
                        'Precision:',
                        min=1,
                        max=4,
                        value=1
                    )
                ),
                ui.input_switch('cbar', 'Display Colorbar', value=True),
                ui.input_switch('enable_clicks', 'Click to Highlight', value=False),

                ui.accordion(
                    ui.accordion_panel(
                        ui.HTML('Filters'),
                        ui.input_checkbox_group(
                            id='rounds',
                            label='Rounds:',
                            choices=ROUND_NAMES,
                            selected=ROUND_NAMES
                        ),
                        ui.output_ui('years_ui'),
                        value='filters'
                    ),
                    open=False
                ),
                open='desktop'
            ),

            # your existing main content unchanged
            ui.include_css('www/styles.css'),
            ui.div(
                ui.div(
                    {'class': 'square-plot-container'},
                    ui.output_plot('heatmap_plot', height='95%', width='120%', click=True),
                ),
                ui.div(
                    ui.div({'style': 'float: left'},
                           ui.download_button('download_plot', 'Download',
                                              width='100px', style='padding: 12px 0')),
                    ui.div({'style': 'float: right'}, ui.output_text('sample_size'))
                )
            ),
            fillable=True,
            fillable_mobile=True
        )
    ),

    # Upcoming games
    ui.nav_panel(
        'Upcoming Games',
        ui.h3('Live Predictions'),
        ui.output_ui('today_games_ui')
    ),

    # Bracket with predictions
    ui.nav_panel(
        'Predicted Bracket',
        ui.output_ui('bracket_ui'),
        ui.output_ui('bracket_position')
    )
)


def server(input, output, session):
    # Create reactive list for holding highlighted cell tuples
    highlight_cells = reactive.value([])

    df_react = reactive.value(df)
    n_react = reactive.value(n)
    notification = reactive.value(0)

    @render.ui
    def years_ui():
        data = df
        min_val = data['Year'].min()
        max_val = data['Year'].max()
        # Render the slider with dynamic min and max values
        return ui.input_slider('years', 'Years:', min=min_val, max=max_val, value=[min_val, max_val], sep='')

    @render.text
    def sample_size():
        return f'N: {n_react.get()}'

    @reactive.effect
    def filter_df():
        new_df = df
        if input.rounds():
            new_df = new_df.filter(pl.col('Round').is_in(input.rounds()))
        miny, maxy = input.years()
        new_df = new_df.filter(pl.col('Year').ge(miny) & pl.col('Year').le(maxy))
        df_react.set(new_df)

        new_n = new_df.shape[0]
        n_react.set(new_n)

        # Determine the new category
        if new_n < 500:
            new_notification = 2
        elif new_n < 1000:
            new_notification = 1
        else:
            new_notification = 0

        # Only show a notification if the category has changed
        if new_notification != notification.get():
            notification.set(new_notification)

            if new_notification == 2:
                ui.notification_show('Sample size < 500. Display shows random variation.', type='error', duration=3)
            elif new_notification == 1:
                ui.notification_show('Sample size < 1000. Display shows random variation.', type='warning', duration=3)

    def generate_heatmap(fontsize=None, **kwargs):
        df_heatmap = heatmap_df(df_react.get())
        if input.annot():
            digits = input.annot_digits()
            annot = np.vectorize(lambda x: f'{x * 100:.{digits}f}')(df_heatmap)
        else:
            annot = False
        annot_kws = {'size': fontsize} if fontsize else {}  # Adjust font size
        fig, ax = heatmap(df_heatmap, input.rounds(), annot=annot, cbar=input.cbar(), annot_kws=annot_kws, **kwargs)
        to_highlight = highlight_cells.get()
        if to_highlight:
            for (i, j) in to_highlight:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=3,
                                           clip_on=False))  # allows for highlighting outside axes dimensions
        return fig


    @render.plot
    def heatmap_plot():
        fig = generate_heatmap()
        return fig


    @reactive.effect
    @reactive.event(input.heatmap_plot_click)
    def update_highlighted_cells():
        if input.enable_clicks() == False:
            return None  # Exit if enable_clicks is turned off
        # Get click data from plot
        click_data = input.heatmap_plot_click()

        if click_data:
            if click_data['domain']['right'] == 1:
                return None  # Exit if key/gradient was clicked instead of heatmap cells
            # Extract raw click coordinates
            raw_x, raw_y = click_data['x'], click_data['y']

            # Ensure valid numeric values
            if raw_x is None or raw_y is None:
                return None  # Exit if x or y values are invalid

            # Convert click positions to nearest integer heatmap indices
            clicked_w = floor(raw_x)  # X-axis == Winning Team Last Digit
            clicked_l = floor(raw_y)  # Y-axis == Losing Team Last Digit

            # Ensure indices are within the 0-9 range
            if 0 <= clicked_w <= 9 and 0 <= clicked_l <= 9:
                clicked_cell = (clicked_l, clicked_w)

                current_cells = highlight_cells.get()
                if clicked_cell in current_cells:
                    new_cells = [cell for cell in current_cells if cell != clicked_cell]
                else:
                    new_cells = current_cells + [clicked_cell]

                highlight_cells.set(new_cells)  # Update highlight list

    @render.download(filename='heatmap.png')
    def download_plot():
        # Get default font size
        fs = plt.rcParams['font.size']
        # Override default font size
        plt.rcParams['font.size'] = 24
        # Generate figure
        fig = generate_heatmap(fontsize=24)
        # Reset font size back to default
        plt.rcParams['font.size'] = fs
        # Save the plot as a PNG image in memory
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
        plt.close(fig)  # Close the figure to free memory

        yield img_buffer.getvalue()

    @reactive.poll(lambda: int(time.time() // 300), 300)
    def today_games():
        return get_next_games()

    @output
    @render.ui
    def today_games_ui():
        games = predict_next_games(today_games())

        if games is None:
            return ui.p('No games this week.')


        days = games.collect().partition_by('Date', as_dict=True)

        ui_blocks = []

        for game_date, day_df in days.items():
            ui_blocks.append(
                ui.div(
                    game_date[0].strftime('%A, %B %d'),
                    class_='games-date-header'
                )
            )

            for g in day_df.iter_rows(named=True):
                ui_blocks.append(
                    ui.div(
                        {'class': 'game-card'},
                        ui.div(g['EventName'], class_='game-title'),
                        ui.div(
                            {'class': 'teams-row'},

                            # Team A
                            ui.div(
                                {
                                    'class': f"team-block {winner(g['A_Score'], g['B_Score'])}"
                                },
                                ui.img(src=g['A_Team_Logo'], class_='team-logo'),
                                ui.div(g['A_Team'], class_='team-name'),
                                ui.div(str(g['A_Score']), class_='team-score'),
                                ui.div(f"Pred: {g['A_Pred_Score']}", class_='team-pred'),
                            ),

                            # Team B
                            ui.div(
                                {
                                    'class': f"team-block {winner(g['B_Score'], g['A_Score'])}"
                                },
                                ui.img(src=g['B_Team_Logo'], class_='team-logo'),
                                ui.div(g['B_Team'], class_='team-name'),
                                ui.div(str(g['B_Score']), class_='team-score'),
                                ui.div(f"Pred: {g['B_Pred_Score']}", class_='team-pred'),
                            ),
                        ),
                    )
                )

        return ui.TagList(ui_blocks)

    # Initial bracket round
    current_bracket_round = reactive.Value(0)

    @render.ui
    def bracket_ui():
        df_bracket = predict_bracket(year)

        if df_bracket is None:
            return ui.div('No bracket data')

        # Build rounds divs
        round_pages = []
        for rnd in ROUND_NAMES:
            games = (
                df_bracket
                .filter(pl.col('Round') == rnd)
                .sort('GameID')
            )

            # Build games divs
            game_cards = []
            for row in games.iter_rows(named=True):
                if rnd in ROUND_NAMES[0:4]:
                    region = (row['A_Region'] + 1) % 2 + 1
                else:
                    region = 1
                game_cards.append(
                    ui.div(
                        {'class': f'game-card region-{region}'},
                        team_row(row, 'A'),
                        team_row(row, 'B'),
                    )
                )

            round_pages.append(
                ui.div(
                    {'class': 'round-page'},
                    *game_cards
                )
            )

        return ui.div(
            {'class': 'bracket-wrapper'},
            ui.div({"class": "year-header"}, f"{year}"),
            ui.div(
                {'class': 'nav-buttons'},
                ui.input_action_button(id='prev_round', label='←', class_='nav-left'),
                ui.div(
                    {'class': 'round-header'},
                    ui.output_text('round_header'),
                ),
                ui.input_action_button(id='next_round', label='→', class_='nav-right'),
            ),
            ui.div({'class': 'bracket-carousel'}, *round_pages),
        )


    @reactive.Effect
    @reactive.event(input.prev_round)
    def _():
        i = max(0, current_bracket_round.get() - 1)
        current_bracket_round.set(i)

    @reactive.Effect
    @reactive.event(input.next_round)
    def _():
        max_rounds = len(ROUND_NAMES)
        i = min(max_rounds - 1, current_bracket_round.get() + 1)
        current_bracket_round.set(i)

    @output
    @render.ui
    def bracket_position():
        idx = current_bracket_round.get()
        return ui.tags.style(
            f"""
            .bracket-carousel {{
                transform: translateX(-{idx * 100}%);
                transition: transform 0.3s ease;
            }}
            """
        )

    @output
    @render.text
    def round_header():
        return ROUND_NAMES[current_bracket_round.get()]


app = App(app_ui, server)
