import os
import time
import json
import requests
import pandas as pd
from pathlib import Path

# =========================
# --- Constants ---
# =========================
FPL_BASE_URL = "https://fantasy.premierleague.com/api/"
DATA_DIR = Path("data/fpl")
TTL_HOURS = 12  # refresh every 12 hours

DIFFICULTY_MULTIPLIER = {
    1: 1.20,
    2: 1.10,
    3: 1.00,
    4: 0.85,
    5: 0.70
}

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

pd.options.mode.chained_assignment = None


# =========================
# --- Utilities ---
# =========================
def is_stale(path: Path, hours: int = TTL_HOURS) -> bool:
    """Check if file is older than TTL"""
    if not path.exists():
        return True
    return (time.time() - path.stat().st_mtime) > hours * 3600


# =========================
# --- Download ---
# =========================
def download_fpl_data(force=False):
    """
    Downloads FPL bootstrap data and saves locally.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DATA_DIR / "bootstrap_static.json"

    if not force and not is_stale(json_path):
        return

    print("⬇️ Downloading FPL data...")
    url = f"{FPL_BASE_URL}bootstrap-static/"
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    data = response.json()

    # Save CSVs for fast loading
    pd.DataFrame(data["elements"]).to_csv(DATA_DIR / "players.csv", index=False)
    pd.DataFrame(data["teams"]).to_csv(DATA_DIR / "teams.csv", index=False)
    pd.DataFrame(data["element_types"]).to_csv(DATA_DIR / "positions.csv", index=False)

    print("✅ FPL data updated")


# =========================
# --- Load & Feature Engineer ---
# =========================
def load_fpl_data():
    """
    Loads FPL data from disk (downloads if missing/stale)
    """
    download_fpl_data()

    players_df = pd.read_csv(DATA_DIR / "players.csv")
    teams_df = pd.read_csv(DATA_DIR / "teams.csv")
    positions_df = pd.read_csv(DATA_DIR / "positions.csv")

    # ---- Team mapping ----
    team_map = teams_df.set_index("id")["name"].to_dict()
    players_df["team"] = players_df["team"].map(team_map)

    # ---- Position mapping ----
    players_df["position_name"] = players_df["element_type"].map(POSITION_MAP)

    # ---- Name ----
    players_df["name"] = (
        players_df["first_name"].fillna("") + " " +
        players_df["second_name"].fillna("")
    ).str.strip()

    # ---- Age ----
    players_df['birth_date'] = pd.to_datetime(
        players_df['birth_date'], errors='coerce'
    )
    
    players_df['age'] = (
        (pd.Timestamp.now() - players_df['birth_date'])
        .dt.days // 365
    )

    return players_df, teams_df, positions_df


# =========================
# --- Example Usage ---
# =========================
if __name__ == "__main__":
    players_df, teams_df, positions_df = load_fpl_data()

    print("Players:", players_df.shape)
    print("Teams:", teams_df.shape)
    print("Positions:", positions_df.shape)

    print(players_df[[
        "name",
        "team",
        "position_name",
        "now_cost",
        "total_points",
        "age"
    ]].head())
