from shiny import App, ui, render, reactive
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import floor


def format_rounds(selected_rounds):
    """Format selected rounds by collapsing consecutive rounds into a range."""

    if selected_rounds is None:
        return ""

    all_rounds = ["Round of 64",
                  "Round of 32",
                  "Sweet 16",
                  "Elite 8",
                  "Final 4",
                  "National Championship"]

    # Sort selected rounds
    selected_rounds = sorted(selected_rounds, key=lambda x: all_rounds.index(x))

    if selected_rounds == all_rounds:
        return ""

    formatted_rounds = []
    temp_range = [selected_rounds[0]]

    for i in range(1, len(selected_rounds)):
        prev_round = selected_rounds[i - 1]
        current_round = selected_rounds[i]

        if all_rounds.index(current_round) == all_rounds.index(prev_round) + 1:
            temp_range.append(current_round)
        else:
            if len(temp_range) >= 3:
                formatted_rounds.append(f"{temp_range[0]} - {temp_range[-1]}")
            else:
                formatted_rounds.extend(temp_range)
            temp_range = [current_round]

    # Handle the last range
    if len(temp_range) >= 3:
        formatted_rounds.append(f"{temp_range[0]} - {temp_range[-1]}")
    else:
        formatted_rounds.extend(temp_range)

    if formatted_rounds:
        title = f"({', '.join(formatted_rounds)})"
        if len(title) > 37:
            title = '\n' + title
        else:
            title = ' ' + title
        return title

    return ""


def heatmap(df: pl.DataFrame, round_filter: list = None, annot: bool = True):
    if round_filter:
        df = df.filter(pl.col("Round").is_in(round_filter))

    freq_table = df.group_by(["W_Last_Digit", "L_Last_Digit"]).agg(
        pl.len().alias("Count")
    ).sort(by=['W_Last_Digit', 'L_Last_Digit'])

    freq_table = freq_table.with_columns(
        (pl.col("Count") / df.shape[0]).alias("Probability")
    )

    all_digits = pl.DataFrame({
        "W_Last_Digit": np.tile(np.arange(10), 10),
        "L_Last_Digit": np.repeat(np.arange(10), 10),
    })
    freq_table = all_digits.join(freq_table, on=["W_Last_Digit", "L_Last_Digit"], how="left").fill_null(0)

    heatmap_df = freq_table.pivot(
        values="Probability",
        index="L_Last_Digit",
        on="W_Last_Digit"
    )

    heatmap_pd = heatmap_df.to_pandas().set_index("L_Last_Digit")

    fig, ax = plt.subplots(figsize=(24, 24), dpi=100)  # Init fig and ax
    ax = sns.heatmap(heatmap_pd, annot=annot, cmap="coolwarm", fmt=".2%", linewidths=0.5)  # Create heatmap
    ax.set_xlabel("Winning Team Last Digit")  # Add axes labels
    ax.set_ylabel("Losing Team Last Digit")  # Add axes labels
    # Add sample size
    cbar = ax.collections[0].colorbar.ax
    cbar.text(0.5, -0.01, f'N: {df.shape[0]}', transform=cbar.transAxes, ha='center', va='top')
    # Add title
    title = "Squares Probability Heatmap"  # Init title
    if round_filter:
        title += format_rounds(round_filter)
    ax.set_title(title)

    return fig, ax


def process_data():
    # Load dataset
    df = pl.read_csv(
        'data/March_Madness_Dataset.csv',
        schema={'Year': pl.String,
                'Round': pl.String,
                'W_Seed': pl.String,
                'W_Team': pl.String,
                'W_Score': pl.String,
                'L_Seed': pl.String,
                'L_Team': pl.String,
                'L_Score': pl.String,
                'OT': pl.String}
    )
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

    # Extract last digits
    df = df.with_columns(
        pl.col('W_Score').mod(10).alias('W_Last_Digit'),
        pl.col('L_Score').mod(10).alias('L_Last_Digit')
    )

    # Standardize round names
    round_col = pl.col('Round')
    df = df.with_columns(
        pl.when(round_col.str.contains('64')).then(pl.lit('Round of 64'))
        .when(round_col.str.contains('32')).then(pl.lit('Round of 32'))
        .when(round_col.str.contains(r'(?i)sixteen|16')).then(pl.lit('Sweet 16'))
        .when(round_col.str.contains(r'(?i)eight|8')).then(pl.lit('Elite 8'))
        .when(round_col.str.contains(r'(?i)four|4')).then(pl.lit('Final 4'))
        .when(round_col.str.contains(r'(?i)champion')).then(pl.lit('National Championship'))
        .otherwise(round_col)
        .alias('Round')
    )

    return df


# Add page title and sidebar
app_ui = ui.page_sidebar(
    ui.sidebar(
        # Option to show or hide annotations in figure
        ui.input_switch('annot', 'Display Frequencies', value=True),
        # Option to highlight cells on click
        ui.input_switch("enable_clicks", "Click to Highlight", value=False),
        # Accordion layout to collapse filters
        ui.accordion(
            # Option to filter figure's underlying df
            ui.accordion_panel(
                ui.HTML("Filters"),  # Collapsible Filters section
                ui.input_checkbox_group(
                    id="rounds",
                    label="Rounds:",
                    choices=['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final 4', 'National Championship'],
                    selected=['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final 4', 'National Championship']
                ),
                value='filters'
            ),
            open=False  # default the filter accordion to closed
        ),
        open='desktop'
    ),
    ui.output_plot("heatmap_plot", height='100%', click=True),
    # ui.output_ui("highlighted_cells_ui"),  # DEV: unused ui for displaying highlighted cells to unhighlight
    fillable=True,
    fillable_mobile=True
)


def server(input, output, session):
    # Create reactive list for holding highlighted cell tuples
    highlight_cells = reactive.value([])

    @render.plot
    def heatmap_plot():
        df = process_data()
        fig, ax = heatmap(df, input.rounds(), annot=input.annot())
        to_highlight = highlight_cells.get()
        if to_highlight:
            for (i, j) in to_highlight:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=3,
                             clip_on=False))  # allows for highlighting outside of axes dimensions
        return fig

    @reactive.effect
    @reactive.event(input.heatmap_plot_click)
    def update_highlighted_cells():
        if input.enable_clicks() == False:
            return None  # Exit if enable_clicks is turned off
        # Get click data from plot
        click_data = input.heatmap_plot_click()

        # Print debugging info
        print(f"Clicked raw: {click_data}")

        if click_data:
            if click_data['domain']['right'] == 1:
                return None  # Exit if key/gradient was clicked instead of heatmap cells
            # Extract raw click coordinates
            raw_x, raw_y = click_data["x"], click_data["y"]

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

    # ### DEV: unused block for highlighting and de-highlighting cells based on action button
    # @render.ui
    # def highlighted_cells_ui():
    #     """Dynamically generate UI elements for highlighted cells."""
    #     cells = highlight_cells.get()
    #     if not cells:
    #         return ui.div("No highlighted cells yet.")
    #
    #     return ui.div(
    #         *[ui.div(f"W: {w} L: {l}",
    #                  ui.input_action_button(f"remove_{l}_{w}", "❌")) for (l, w) in cells]
    #     )
    #
    # @reactive.effect
    # @reactive.event(input.highlight_cell)
    # def _():
    #     cell_prev = highlight_cells.get()
    #     new_w = input.w_digit()
    #     new_l = input.l_digit()
    #     cell_new = (new_l, new_w)
    #     if cell_new not in cell_prev:
    #         highlight_cells.set(cell_prev + [cell_new])



app = App(app_ui, server)
