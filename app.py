"""

╔══════════════════════════════════════════════════════════════╗
║   RAINFALL DATA ANALYZER - Flask Backend                     ║
║   REST API with Pandas data management                       ║
╚══════════════════════════════════════════════════════════════╝
"""

from flask import Flask, jsonify, request, render_template, send_file
import pandas as pd
import os
import io
import json
from datetime import datetime

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB max upload

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "rainfall_data.csv")

MONTHS_ORDER = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

# ══════════════════════════════════════════════════════════════
#  DATA HELPERS
# ══════════════════════════════════════════════════════════════

def _load() -> pd.DataFrame:
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            df["Rainfall_mm"] = pd.to_numeric(df["Rainfall_mm"], errors="coerce")
            df.dropna(inplace=True)
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["City","Month","Rainfall_mm"])


def _save(df: pd.DataFrame):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    df.to_csv(DATA_FILE, index=False)


def _month_index(month: str) -> int:
    try:
        return MONTHS_ORDER.index(month)
    except ValueError:
        return 99


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# ── GET /api/data ─────────────────────────────────────────────
@app.route("/api/data", methods=["GET"])
def get_data():
    df = _load()
    records = df.to_dict(orient="records")
    return jsonify({"success": True, "data": records, "count": len(records)})


# ── POST /api/add ─────────────────────────────────────────────
@app.route("/api/add", methods=["POST"])
def add_data():
    body = request.get_json(force=True)
    city = str(body.get("city","")).strip()
    month = str(body.get("month","")).strip()
    rainfall = body.get("rainfall")

    # Validation
    if not city:
        return jsonify({"success": False, "error": "City name required"}), 400
    if month not in MONTHS_ORDER:
        return jsonify({"success": False, "error": "Invalid month"}), 400
    try:
        rainfall = float(rainfall)
        if rainfall < 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Rainfall must be a positive number"}), 400

    df = _load()
    mask = (df["City"].str.lower() == city.lower()) & (df["Month"] == month)
    if mask.any():
        df.loc[mask, "Rainfall_mm"] = rainfall
        action = "updated"
    else:
        new_row = pd.DataFrame({"City":[city],"Month":[month],"Rainfall_mm":[rainfall]})
        df = pd.concat([df, new_row], ignore_index=True)
        action = "added"

    _save(df)
    return jsonify({"success": True, "action": action,
                    "record": {"City":city,"Month":month,"Rainfall_mm":rainfall}})


# ── DELETE /api/delete ────────────────────────────────────────
@app.route("/api/delete", methods=["DELETE"])
def delete_data():
    body = request.get_json(force=True)
    city = str(body.get("city","")).strip()
    month = str(body.get("month","")).strip()

    df = _load()
    mask = (df["City"].str.lower() == city.lower()) & (df["Month"] == month)
    if not mask.any():
        return jsonify({"success": False, "error": "Record not found"}), 404

    df = df[~mask].reset_index(drop=True)
    _save(df)
    return jsonify({"success": True, "message": f"Deleted {city} - {month}"})


# ── GET /api/stats ────────────────────────────────────────────
@app.route("/api/stats", methods=["GET"])
def get_stats():
    df = _load()
    if df.empty:
        return jsonify({"success": True, "stats": None})

    max_row = df.loc[df["Rainfall_mm"].idxmax()]
    min_row = df.loc[df["Rainfall_mm"].idxmin()]

    # Per city totals
    city_stats = (
        df.groupby("City")["Rainfall_mm"]
          .agg(total="sum", average="mean", maximum="max", count="count")
          .reset_index()
          .round(2)
          .to_dict(orient="records")
    )

    # Monthly aggregate (all cities combined)
    monthly = (
        df.groupby("Month")["Rainfall_mm"]
          .sum()
          .reset_index()
    )
    monthly["month_idx"] = monthly["Month"].apply(_month_index)
    monthly = monthly.sort_values("month_idx")[["Month","Rainfall_mm"]].to_dict(orient="records")

    # Per-city per-month (for trend chart)
    trend = {}
    for city in df["City"].unique():
        city_df = df[df["City"] == city].copy()
        city_df["month_idx"] = city_df["Month"].apply(_month_index)
        city_df = city_df.sort_values("month_idx")
        trend[city] = city_df[["Month","Rainfall_mm"]].to_dict(orient="records")

    stats = {
        "total":     round(df["Rainfall_mm"].sum(), 2),
        "average":   round(df["Rainfall_mm"].mean(), 2),
        "maximum":   round(df["Rainfall_mm"].max(), 2),
        "minimum":   round(df["Rainfall_mm"].min(), 2),
        "max_city":  max_row["City"],
        "max_month": max_row["Month"],
        "min_city":  min_row["City"],
        "min_month": min_row["Month"],
        "total_records": len(df),
        "cities":    sorted(df["City"].unique().tolist()),
        "city_stats": city_stats,
        "monthly":   monthly,
        "trend":     trend,
    }
    return jsonify({"success": True, "stats": stats})


# ── POST /api/import ──────────────────────────────────────────
@app.route("/api/import", methods=["POST"])
def import_csv():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"success": False, "error": "Only CSV files allowed"}), 400

    try:
        imported = pd.read_csv(file)
        # Normalize columns
        col_map = {}
        for col in imported.columns:
            cl = col.strip().lower()
            if "city" in cl:
                col_map[col] = "City"
            elif "month" in cl:
                col_map[col] = "Month"
            elif "rain" in cl:
                col_map[col] = "Rainfall_mm"
        imported.rename(columns=col_map, inplace=True)

        required = {"City","Month","Rainfall_mm"}
        if not required.issubset(set(imported.columns)):
            return jsonify({"success": False,
                            "error": "CSV must have: City, Month, Rainfall_mm columns"}), 400

        imported = imported[["City","Month","Rainfall_mm"]]
        imported["Rainfall_mm"] = pd.to_numeric(imported["Rainfall_mm"], errors="coerce")
        imported.dropna(inplace=True)

        # Filter only valid months
        imported = imported[imported["Month"].isin(MONTHS_ORDER)]

        df = _load()
        df = pd.concat([df, imported], ignore_index=True)
        df = df.drop_duplicates(subset=["City","Month"], keep="last")
        _save(df)

        return jsonify({"success": True,
                        "message": f"Imported {len(imported)} records successfully!",
                        "count": len(imported)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── GET /api/export ───────────────────────────────────────────
@app.route("/api/export", methods=["GET"])
def export_csv():
    df = _load()
    if df.empty:
        return jsonify({"success": False, "error": "No data to export"}), 400
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"rainfall_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )


# ── GET /api/cities ───────────────────────────────────────────
@app.route("/api/cities", methods=["GET"])
def get_cities():
    df = _load()
    cities = sorted(df["City"].unique().tolist()) if not df.empty else []
    return jsonify({"success": True, "cities": cities})


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
    print("\n  Rainfall Data Analyzer")
    print("  --------------------------------")
    print("  Server: http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop\n")
    app.run(debug=True, host="127.0.0.1", port=5000)
