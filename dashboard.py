import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from statsbombpy import sb
from utils import (
    load_match,
    calculate_stats,
    visualize_tactical_formation,
    plot_shots,
    plot_xg_chart,
    plot_passes,
    plot_pass_network,
)

st.set_page_config(
    page_title="Football Dashboard",
    page_icon="assets/ball.png",
    layout="wide",
)
all_competitions = sb.competitions()
all_competitions = all_competitions.loc[all_competitions["competition_gender"] == "male"]

competition = st.sidebar.selectbox(
    "Select competition",
    all_competitions.competition_name.unique(),
    format_func=lambda x: x.title(),
)

season = st.sidebar.selectbox(
    "Select season",
    all_competitions.loc[all_competitions["competition_name"] == competition, "season_name"].unique(),
)

competition_id = all_competitions.loc[all_competitions["competition_name"] == competition, "competition_id"].values[0]
season_id = all_competitions.loc[all_competitions["season_name"] == season, "season_id"].values[0]


plt.style.use("ggplot")
matches = sb.matches(competition_id=competition_id, season_id=season_id)
morocco_matches = matches[["home_team", "away_team", "match_date", "match_id"]]
morocco_matches = morocco_matches.sort_values("match_date")

match_mapping = morocco_matches.apply(
    lambda x: {f'{x["home_team"]} vs {x["away_team"]} - {x["match_date"]}': x["match_id"]},
    axis=1,
)
match_mapping = {k: v for d in match_mapping for k, v in d.items()}



match = st.selectbox("Select match", list(match_mapping.keys()))
match_id = match_mapping[match]
df, team1, team2, tactics = load_match(match_id)
st.title(f"Analysis of {match}")


team = st.selectbox("Select team", df["team_name"].unique())
max_match_time = df.loc[df["period"] < 3, "minute"].max()
time = st.slider(
    "Select minute",
    0,
    int(max_match_time),
    (0, int(max_match_time)),
    1,
)
tactical_changes = df.loc[
    (df["type_name"].isin(["Tactical Shift", "Starting XI"]))
    & (df["team_name"] == team)
    & (df["minute"] >= time[0])
    & (df["minute"] < time[1]),
    ["id", "minute", "tactics_formation"],
]
st.subheader("Team stats")
possession, pass_completion, n_shots, shot_on_target, goals = calculate_stats(
    df, team, time
)

cols = st.columns(5)
cols[0].metric("Possession", possession)
cols[1].metric("Pass completion", pass_completion)
cols[2].metric("Shots", n_shots)
cols[3].metric("Shots on target", shot_on_target)
cols[4].metric("Goals", goals)

st.subheader("Lineup and Tactics")
change_id = st.radio(
    "Select tactical change",
    tactical_changes["id"].unique(),
    format_func=lambda x: f"{tactical_changes.loc[tactical_changes['id'] == x, 'minute'].values[0]} min",
    horizontal=True,
)

# TODO: handle if the range slider is set to a time where no tactical changes have been made

starting_lineup = tactics.loc[
    tactics["id"] == tactical_changes["id"].unique()[0],
    ["jersey_number", "player_name", "position_name"],
]
starting_lineup = starting_lineup.rename(
    columns={
        "jersey_number": "Jersey number",
        "player_name": "Player",
        "position_name": "Position",
    }
)


def color_substitutions(s):
    if "(Subbed out)" in s["Position"]:
        return ["background-color: crimson"] * len(s)
    elif "(Subbed in)" in s["Position"]:
        return ["background-color: green"] * len(s)
    else:
        return [""] * len(s)


if change_id != tactical_changes["id"].unique()[0]:
    lineup = tactics.loc[
        tactics["id"] == change_id, ["jersey_number", "player_name", "position_name"]
    ]
    lineup = lineup.rename(
        columns={
            "jersey_number": "Jersey number",
            "player_name": "Player",
            "position_name": "Position",
        }
    )
    starting_lineup.loc[
        ~starting_lineup["Jersey number"].isin(lineup["Jersey number"].unique()),
        "Position",
    ] += f" (Subbed out)"
    lineup.loc[
        ~lineup["Jersey number"].isin(starting_lineup["Jersey number"].unique()),
        "Position",
    ] += f" (Subbed in)"
    starting_lineup = pd.concat(
        [
            starting_lineup,
            lineup.loc[
                ~lineup["Jersey number"].isin(starting_lineup["Jersey number"].unique())
            ],
        ]
    )
lineup = starting_lineup.copy()

hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
fig0 = visualize_tactical_formation(tactics, change_id)
col1, col2 = st.columns(2)
st.markdown(hide_table_row_index, unsafe_allow_html=True)
col1.table(lineup.style.apply(color_substitutions, axis=1))
col2.pyplot(fig0)

st.subheader("Shots and Passes")
fig1 = plot_shots(df, team, time)
fig2 = plot_xg_chart(df, team, time)
col1, col2 = st.columns(2)
col1.pyplot(fig1)
col2.pyplot(fig2)

outcome = st.selectbox(
    "Select outcome of pass", df[df.type_name == "Pass"].outcome_name.unique()
)
show_passes = st.checkbox("Show passing map")
show_heatmap = st.checkbox("Show heatmap", value=True)
fig3 = plot_passes(
    df, team, outcome, time, show_passes=show_passes, show_heatmap=show_heatmap
)
fig4 = plot_pass_network(df, team, time)
col3, col4 = st.columns(2)
col3.pyplot(fig3)
col4.pyplot(fig4)
