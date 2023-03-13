# About

Dashboards for visualizing Morocco's 2022 World Cup Run using event data from each of their matches, though this can be extended to any match in any competition.

Deployed [here](https://morocco-worldcup2022.streamlit.app/)

# TODO

## Visualization style

- [x] Add slider to select the time interval to visualize
- [x] Keep only events from game normal time, not penalties (penalties will be treated differently)
- [ ] Add a button to toggle between the two modes (normal time vs. penalties, only available for matches with penalties)
- [ ] Mark the different types of shots (on target, off target, blocked, saved, etc.) in the shot map

## Analytical Aspects
- [ ] Visualizing changes of tactics
- [ ] Quantifying the impact of substitutions
- [ ] Calculating expected threat (xT) of passes
- [ ] Visualizing progressive passes
- [ ] Add comparative analysis between opposing teams

## Documentation
- [ ] Add documentation for each visualization and analysis
- [ ] Write a blog post about the project explaining findings

# Try it out

## Locally
We deployed a Docker image of the app on Docker Hub. To run it locally, you need to have Docker installed on your machine. Then, run the following command:

```bash
docker pull pillarxyz/football-analytics-morocco
docker run -p 8501:8501 pillarxyz/football-analytics-morocco
```

## On Streamlit Sharing
You can also run the app on Streamlit Sharing. Just click on the button below:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://morocco-worldcup2022.streamlit.app/)