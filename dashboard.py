import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from mplsoccer import Pitch, Sbopen, VerticalPitch
import streamlit as st
from scipy.ndimage import gaussian_filter
import pandas as pd
from statsbombpy import sb
import os

plt.style.use("ggplot")


matches = sb.matches(competition_id=43, season_id=106)
mask = (matches.home_team == "Morocco") | (matches.away_team == "Morocco")
morocco_matches = matches.loc[
    mask, ["home_team", "away_team", "match_date", "match_id"]
]
morocco_matches = morocco_matches.sort_values("match_date")
morocco_matches.loc[
    morocco_matches.away_team == "Croatia", "away_team"
] = "Croatia Group Stage"
morocco_matches.loc[
    morocco_matches.home_team == "Croatia", "home_team"
] = "Croatia 3rd Place Play-off"


position_coordinates = {
    "Goalkeeper": (2, 40),
    "Right Back": (25, 75),
    "Right Center Back": (20, 50),
    "Left Center Back": (20, 30),
    "Left Back": (25, 5),
    "Left Center Midfield": (50, 25),
    "Right Center Midfield": (50, 55),
    "Left Midfield": (55, 5),
    "Right Midfield": (55, 75),
    "Right Center Forward": (80, 45),
    "Left Center Forward": (80, 35),
    "Right Wing": (75, 75),
    "Left Wing": (75, 10),
    "Right Defensive Midfield": (40, 50),
    "Left Defensive Midfield": (40, 30),
    "Right Wing Back": (35, 70),
    "Left Wing Back": (35, 10),
    "Center Defensive Midfield": (35, 40),
    "Center Forward": (80, 40),
    "Center Attacking Midfield": (65, 40),
    "Center Back": (20, 40),
    
    
}

match_mapping = morocco_matches.apply(
    lambda x: {x["home_team"]: x["match_id"]}
    if x["away_team"] == "Morocco"
    else {x["away_team"]: x["match_id"]},
    axis=1,
)
match_mapping = {k: v for d in match_mapping for k, v in d.items()}

@st.cache_data
def load_match(match_id):
    parser = Sbopen()
    df, related, freeze, tactics = parser.event(match_id)
    team1, team2 = df.team_name.unique()
    df["outcome_name"] = df["outcome_name"].fillna("Complete")

    players_nums = tactics[["player_name", "jersey_number"]].drop_duplicates()

    df = df.merge(players_nums, on="player_name", how="left")
    df = df.merge(
        players_nums,
        left_on="pass_recipient_name",
        right_on="player_name",
        how="left",
        suffixes=("", "_recipient"),
    )

    if not os.path.exists("data"):
        os.mkdir("data")
    df.to_csv(f"data/{match_id}.csv", index=False)
    return df, team1, team2, tactics

@st.cache_data
def calculate_stats(df, team, timeframe):

    if timeframe == (0, 0):
        df = df.copy()
    else:
        df = df[(df["minute"] >= timeframe[0]) & (df["minute"] <= timeframe[1])].copy()

    pass_mask = (df.type_name == "Pass") & (df.team_name == team)
    pass_completion = (
        df.loc[pass_mask, "outcome_name"].value_counts(normalize=True).loc["Complete"]
    )
    possession = df["possession_team_name"].value_counts().loc[team] / len(df)

    pass_completion = round(pass_completion, 2)
    possession = round(possession, 2)

    shot_mask = (df.type_name == "Shot") & (df.team_name == team) & (df.period != 5)
    n_shots = df.loc[shot_mask, "outcome_name"].shape[0]

    shot_types = df.loc[shot_mask, "outcome_name"].value_counts()
    if "Goal" not in shot_types.index:
        shot_types.loc["Goal"] = 0
    if "Saved" not in shot_types.index:
        shot_types.loc["Saved"] = 0
    try:
        goals = shot_types.loc["Goal"]
        shot_on_target = shot_types.loc["Goal"] + shot_types.loc["Saved"]
    except KeyError:
        shot_on_target = shot_types.loc["Saved"]
        goals = 0

    return possession, pass_completion, n_shots, shot_on_target, goals


def plot_shots(df, team, timeframe):

    color_map = {
                "Saved": "chocolate",
                "Saved Off Target": "chocolate",
                "Saved to Post": "chocolate",
                "Off T": "black",
                "Blocked": "brown",
                "Post": "gray",
                "Wayward": "burlywood",
            }
    

    if timeframe == (0, 0):
        df = df.copy()
    else:
        df = df[(df["minute"] >= timeframe[0]) & (df["minute"] <= timeframe[1])].copy()

    df = df[df["period"] != 5]
    mask = (df.type_name == "Shot") & (df.team_name == team)
    df = df.loc[mask, ["x", "y", "outcome_name", "player_name", "shot_statsbomb_xg"]]
    df["player_name"] = df["player_name"].str.split().str[-1]

    pitch = VerticalPitch(line_color="black", half=True, goal_alpha=0.8)
    fig, ax = pitch.grid(
        grid_height=0.9,
        title_height=0.06,
        axis=False,
        endnote_height=0.04,
        title_space=0,
        endnote_space=0,
    )

    for i, row in df.iterrows():
        if row["outcome_name"] == "Goal":
            pitch.scatter(
                row.x,
                row.y,
                alpha=min(0.5 + row["shot_statsbomb_xg"], 1),
                s=250 + row["shot_statsbomb_xg"] * 500,
                edgecolors="crimson",
                marker="football",
                ax=ax["pitch"],
            )
            pitch.annotate(
                row["player_name"], (row.x + 1, row.y - 2), ax=ax["pitch"], fontsize=12
            )
        else:
            pitch.scatter(
                row.x,
                row.y,
                s=250 + row["shot_statsbomb_xg"] * 500,
                color=color_map[row["outcome_name"]],
                ax=ax["pitch"],
            )

    legend_elements = [
        Patch(facecolor=color_map["Saved"], label="Saved"),
        Patch(facecolor=color_map["Off T"], label="Off Target"),
        Patch(facecolor=color_map["Blocked"], label="Blocked"),
        Patch(facecolor=color_map["Post"], label="Post"),
        Patch(facecolor=color_map["Wayward"], label="Wayward"),
    ]
    ax["pitch"].legend(
        handles=legend_elements,
        loc="lower left",
        bbox_to_anchor=(0.05, 0.05),
        ncol=2,
        fontsize=12,
    )
    fig.suptitle(f"{team} shots", fontsize=24)
    return fig


def plot_xg_chart(df, team, timeframe):

    if timeframe == (0, 0):
        df = df.copy()
    else:
        df = df[(df["minute"] >= timeframe[0]) & (df["minute"] <= timeframe[1])].copy()

    df = df[df["period"] != 5]
    df_shots = df.loc[
        (df.type_name == "Shot"),
        [
            "x",
            "y",
            "end_x",
            "end_y",
            "player_name",
            "outcome_name",
            "team_name",
            "shot_statsbomb_xg",
            "minute",
            "second",
        ],
    ]

    df_shots["time"] = df_shots["minute"] + df_shots["second"] / 60

    shots = df_shots[df_shots["shot_statsbomb_xg"] > 0].copy()

    shots["marker_size"] = (
        shots["shot_statsbomb_xg"] / shots["shot_statsbomb_xg"].max() * 1500
    )

    fig, ax = plt.subplots(figsize=(9.5, 8))
    ax.set_xlabel("Minute")
    ax.set_ylabel("xG")
    ax.set_title("xG flow chart")
    ax.grid(False)
    plt.xticks(np.arange(timeframe[0], timeframe[1], 5))

    team_df = df_shots.loc[df_shots["team_name"] == team].copy()
    team_df = team_df.sort_values("time")
    team_df["xG_cum"] = team_df["shot_statsbomb_xg"].cumsum()
    ax.step(team_df["time"], team_df["xG_cum"], label=team)

    for i, row in team_df.iterrows():
        if row["outcome_name"] == "Goal":
            ax.scatter(row["time"], row["xG_cum"], color="crimson", s=100)
            ax.text(
                row["time"] + 0.5,
                row["xG_cum"],
                f"Goal by {row['player_name'].split()[-1]}",
                fontsize=14,
            )

    ax.legend()
    return fig


def plot_pass_network(df, team, timeframe):

    if timeframe == (0, 0):
        df = df.copy()
    else:
        df = df[(df["minute"] >= timeframe[0]) & (df["minute"] <= timeframe[1])].copy()

    mask = (
        (df.type_name == "Pass")
        & (df.team_name == team)
        & (df.minute < timeframe[1])
        & (df.minute > timeframe[0])
        & (df.outcome_name == "Complete")
        & (df.sub_type_name != "Throw-in")
    )
    df_pass = df.loc[
        mask, ["x", "y", "end_x", "end_y", "jersey_number", "jersey_number_recipient"]
    ]

    scatter_df = pd.DataFrame()
    for i, name in enumerate(df_pass["jersey_number"].unique()):
        passx = df_pass.loc[df_pass["jersey_number"] == name]["x"].to_numpy()
        recx = df_pass.loc[df_pass["jersey_number_recipient"] == name][
            "end_x"
        ].to_numpy()
        passy = df_pass.loc[df_pass["jersey_number"] == name]["y"].to_numpy()
        recy = df_pass.loc[df_pass["jersey_number_recipient"] == name][
            "end_y"
        ].to_numpy()
        scatter_df.at[i, "jersey_number"] = name
        scatter_df.at[i, "x"] = np.mean(np.concatenate([passx, recx]))
        scatter_df.at[i, "y"] = np.mean(np.concatenate([passy, recy]))
        scatter_df.at[i, "no"] = (
            df_pass.loc[df_pass["jersey_number"] == name].count().iloc[0]
        )

    scatter_df["marker_size"] = scatter_df["no"] / scatter_df["no"].max() * 1500
    df_pass["pair_key"] = df_pass.apply(
        lambda x: "_".join(
            sorted(
                [
                    str(x["jersey_number"].astype(int)),
                    str(x["jersey_number_recipient"].astype(int)),
                ]
            )
        ),
        axis=1,
    )
    lines_df = df_pass.groupby(["pair_key"]).x.count().reset_index()
    lines_df.rename({"x": "pass_count"}, axis="columns", inplace=True)
    lines_df = lines_df[lines_df["pass_count"] > 2]

    pitch = Pitch(line_color="black", line_zorder=2)
    fig, ax = pitch.grid(
        endnote_height=0.03,
        endnote_space=0,
        grid_width=0.88,
        left=0.025,
        title_height=0.06,
        title_space=0,
        axis=False,
        grid_height=0.86,
    )
    pitch.scatter(
        scatter_df.x,
        scatter_df.y,
        s=scatter_df.marker_size,
        color="crimson",
        edgecolors="grey",
        linewidth=1,
        alpha=1,
        ax=ax["pitch"],
        zorder=3,
    )
    for i, row in scatter_df.iterrows():
        pitch.annotate(
            row.jersey_number.astype(int),
            xy=(row.x, row.y),
            c="black",
            va="center",
            ha="center",
            weight="bold",
            size=14,
            ax=ax["pitch"],
            zorder=4,
        )

    for i, row in lines_df.iterrows():
        player1 = int(row["pair_key"].split("_")[0])
        player2 = int(row["pair_key"].split("_")[1])
        player1_x = scatter_df.loc[scatter_df["jersey_number"] == player1]["x"].iloc[0]
        player1_y = scatter_df.loc[scatter_df["jersey_number"] == player1]["y"].iloc[0]
        player2_x = scatter_df.loc[scatter_df["jersey_number"] == player2]["x"].iloc[0]
        player2_y = scatter_df.loc[scatter_df["jersey_number"] == player2]["y"].iloc[0]
        num_passes = row["pass_count"]
        line_width = num_passes / lines_df["pass_count"].max() * 5
        pitch.lines(
            player1_x,
            player1_y,
            player2_x,
            player2_y,
            alpha=1,
            lw=line_width,
            zorder=2,
            color="crimson",
            ax=ax["pitch"],
        )

    fig.suptitle(f"{team} Passing Network", fontsize=24)
    return fig


def plot_passes(df, team, outcome, timeframe, **kwargs):

    if timeframe == (0, 0):
        df = df.copy()
    else:
        df = df[(df["minute"] >= timeframe[0]) & (df["minute"] <= timeframe[1])].copy()

    mask = (df.type_name == "Pass") & (df.team_name == team)
    passes = df.loc[mask, ["x", "y", "end_x", "end_y", "outcome_name"]]
    passes = passes.loc[passes["outcome_name"] == outcome]

    show_passes = kwargs.get("show_passes", False)
    show_heatmap = kwargs.get("show_heatmap", True)

    pitch = Pitch(pitch_type="statsbomb", line_zorder=2)
    fig, ax = pitch.grid(
        endnote_height=0.03,
        endnote_space=0,
        grid_width=0.88,
        left=0.025,
        title_height=0.06,
        title_space=0,
        axis=False,
        grid_height=0.86,
    )

    if show_heatmap:

        passmap = pitch.bin_statistic(
            passes.end_x, passes.end_y, statistic="count", bins=(30, 30), normalize=True
        )
        passmap["statistic"] = gaussian_filter(passmap["statistic"], 1)

        pitch.heatmap(passmap, ax=ax["pitch"], cmap="plasma", edgecolors="#22312b")

    if show_passes:
        pitch.arrows(
            passes.x,
            passes.y,
            passes.end_x,
            passes.end_y,
            width=1,
            headwidth=4,
            headlength=4,
            color="crimson",
            ax=ax["pitch"],
            zorder=3,
        )

    fig.suptitle(f"{team} passes", fontsize=24)
    return fig


def visualize_tactical_formation(df, tactics, team, shift_id=None):

    pos = tactics.loc[tactics["id"] == shift_id].copy()

    pitch = Pitch(line_color="black")
    fig, ax = pitch.draw(figsize=(16, 11))

    for i, row in pos.iterrows():
        x, y = position_coordinates[row["position_name"]]
        pitch.scatter(
            [x], [y], ax=ax, s=700, color="black", edgecolor="white", zorder=3
        )
        ax.text(
            x,
            y,
            row["jersey_number"],
            ha="center",
            va="center",
            color="white",
            fontsize=20,
            zorder=4,
        )

    return fig


st.set_page_config(
    page_title="Analysis of WC 2022 - Morocco",
    page_icon="assets/ball.png",
    layout="wide",
)

opponent = st.selectbox("Select match", list(match_mapping.keys()))
match_id = match_mapping[opponent]
df, team1, team2, tactics = load_match(match_id)
st.title(f"Analysis of WC 2022 - Morocco vs {opponent}")


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
fig0 = visualize_tactical_formation(df, tactics, team, change_id)
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
