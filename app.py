"""
Pluvia - Rainfall Data Analyzer
================================

A cross-platform Flet application for exploring rainfall trends across major
world cities. Provides a pandas-backed data engine, a pure analytics engine
(KPIs + dynamic filtering), and a dark-mode glassmorphism dashboard with
interactive record management, paginated data, summary reports, and CSV export.

-------------------------------------------------------------------------------
INSTALLATION
-------------------------------------------------------------------------------
    pip install flet pandas matplotlib

-------------------------------------------------------------------------------
RUN
-------------------------------------------------------------------------------
    flet run app.py

NOTE: charts are rendered with Matplotlib (headless "Agg" backend) to
transparent PNGs and embedded via ft.Image; flet 1.0.4 ships no native chart
widgets. Matplotlib is therefore imported at startup to power the
visualization engine.
"""

from __future__ import annotations

import asyncio
import base64
import calendar
import http.client
import io
import json
import math
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import flet as ft
import pandas as pd

# ============================================================================ #
#  1. Constants & Metadata
# ============================================================================ #

# Canonical column that stores rainfall amounts in millimetres.
RAINFALL_COL = "Rainfall (mm)"

# Ordered month labels (reused across dataset, analytics and future charts).
MONTHS: list[str] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# Indian cities represented in the bundled historical dataset.
CITIES: list[str] = ["Mumbai", "Delhi", "Chennai", "Kolkata", "Bengaluru", "Nagpur"]

# Mean monthly precipitation (mm) for 1991-2020 from Open-Meteo ERA5
# reanalysis daily precipitation sums, aggregated by month and averaged
# across the 30 calendar years. Values represent the nearest ERA5 grid cell.
SAMPLE_RAINFALL_MM: dict[str, dict[str, float]] = {
    "Mumbai": {
        "Jan": 0.8, "Feb": 0.5, "Mar": 0.8, "Apr": 0.3, "May": 15.3, "Jun": 420.2,
        "Jul": 636.2, "Aug": 476.7, "Sep": 294.7, "Oct": 61.0, "Nov": 7.9, "Dec": 5.5,
    },
    "Delhi": {
        "Jan": 24.7, "Feb": 31.0, "Mar": 21.9, "Apr": 14.0, "May": 19.5, "Jun": 69.2,
        "Jul": 194.7, "Aug": 178.1, "Sep": 93.1, "Oct": 12.9, "Nov": 5.2, "Dec": 6.7,
    },
    "Chennai": {
        "Jan": 13.8, "Feb": 6.4, "Mar": 8.2, "Apr": 14.6, "May": 38.0, "Jun": 66.5,
        "Jul": 70.3, "Aug": 93.6, "Sep": 102.5, "Oct": 208.0, "Nov": 227.4, "Dec": 117.0,
    },
    "Kolkata": {
        "Jan": 13.0, "Feb": 23.7, "Mar": 29.5, "Apr": 67.7, "May": 121.8, "Jun": 265.0,
        "Jul": 348.3, "Aug": 307.7, "Sep": 250.2, "Oct": 131.1, "Nov": 28.5, "Dec": 10.2,
    },
    "Bengaluru": {
        "Jan": 3.4, "Feb": 6.0, "Mar": 14.5, "Apr": 43.9, "May": 123.4, "Jun": 119.8,
        "Jul": 128.5, "Aug": 122.0, "Sep": 121.1, "Oct": 138.4, "Nov": 61.4, "Dec": 22.8,
    },
    # NOTE: Nagpur uses APPROXIMATE long-term climatological normals (IMD-style,
    # ~1,110 mm/yr), not the ERA5 aggregation used for the cities above. Replace
    # with ERA5-derived values if exact consistency matters.
    "Nagpur": {
        "Jan": 14.2, "Feb": 17.8, "Mar": 17.1, "Apr": 11.0, "May": 20.8, "Jun": 190.4,
        "Jul": 308.0, "Aug": 262.6, "Sep": 188.9, "Oct": 56.0, "Nov": 15.8, "Dec": 11.5,
    },
}

# ============================================================================ #
#  2. Sample Data Engine
#
#  The dataset lives in two interchangeable shapes:
#   * LONG format  -> one row per (City, Month) observation. This is the
#                     canonical frame used by every analytics function.
#   * WIDE format  -> a City x Month matrix, useful for displaying tables
#                     in the UI and as a natural fit for charting later.
# ============================================================================ #


def generate_sample_data() -> pd.DataFrame:
    """Build the initial rainfall dataset in LONG format.

    Returns a DataFrame with the columns ``City``, ``Month`` and
    ``Rainfall (mm)`` containing curated, realistic monthly rainfall norms
    for the Indian cities defined in :data:`SAMPLE_RAINFALL_MM`.
    """
    records = [
        {"City": city, "Month": month, RAINFALL_COL: value}
        for city, monthly in SAMPLE_RAINFALL_MM.items()
        for month, value in monthly.items()
    ]
    return pd.DataFrame(records, columns=["City", "Month", RAINFALL_COL])


def parse_rainfall_csv(contents: bytes) -> pd.DataFrame:
    """Validate a CSV spreadsheet and return canonical rainfall records.

    Required columns are City, Month, and Rainfall (mm). Month names may use
    abbreviations or full English names. Validation is all-or-nothing so an
    invalid spreadsheet never partially changes the active dataset.
    """
    try:
        uploaded = pd.read_csv(io.BytesIO(contents), encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc

    column_aliases = {
        "city": "City",
        "month": "Month",
        "rainfall": RAINFALL_COL,
        "rainfallmm": RAINFALL_COL,
        "rainfallinmm": RAINFALL_COL,
    }
    canonical_columns: dict[str, str] = {}
    for column in uploaded.columns:
        normalized = "".join(char for char in str(column).casefold() if char.isalnum())
        canonical = column_aliases.get(normalized)
        if canonical:
            if canonical in canonical_columns.values():
                raise ValueError(f"CSV contains duplicate {canonical!r} columns.")
            canonical_columns[str(column)] = canonical

    required_columns = {"City", "Month", RAINFALL_COL}
    if set(canonical_columns.values()) != required_columns:
        raise ValueError(
            "CSV must include City, Month, and Rainfall (mm) columns. "
            "Example header: City,Month,Rainfall (mm)"
        )

    uploaded = uploaded.rename(columns=canonical_columns)[
        ["City", "Month", RAINFALL_COL]
    ]
    month_names = {
        name.casefold(): MONTHS[index - 1]
        for index, name in enumerate(calendar.month_abbr)
        if name
    }
    month_names.update({
        name.casefold(): MONTHS[index - 1]
        for index, name in enumerate(calendar.month_name)
        if name
    })

    records: list[dict[str, str | float]] = []
    errors: list[str] = []
    for row_number, row in enumerate(uploaded.itertuples(index=False, name=None), start=2):
        city_value, month_value, rainfall_value = row
        city = str(city_value).strip() if pd.notna(city_value) else ""
        month_key = str(month_value).strip().casefold() if pd.notna(month_value) else ""
        month = month_names.get(month_key)
        try:
            rainfall = float(rainfall_value)
        except (TypeError, ValueError):
            rainfall = math.nan

        row_errors = []
        if not city:
            row_errors.append("City is required")
        if month is None:
            row_errors.append("Month must be a valid month name")
        if not math.isfinite(rainfall) or rainfall < 0:
            row_errors.append("Rainfall must be a finite, non-negative number")
        if row_errors:
            errors.append(f"Row {row_number}: {', '.join(row_errors)}")
        else:
            records.append({"City": city, "Month": month, RAINFALL_COL: rainfall})

    if errors:
        details = "; ".join(errors[:5])
        remaining = len(errors) - 5
        if remaining > 0:
            details += f"; and {remaining} more invalid row(s)"
        raise ValueError(details)
    if not records:
        raise ValueError("CSV contains no rainfall records.")
    return pd.DataFrame(records, columns=["City", "Month", RAINFALL_COL])


def to_wide_format(data: pd.DataFrame) -> pd.DataFrame:
    """Reshape a LONG-format frame into a City x Month matrix (WIDE format)."""
    present_cities = data["City"].dropna().astype(str).unique().tolist()
    city_order = [city for city in CITIES if city in present_cities]
    city_order.extend(city for city in present_cities if city not in CITIES)
    wide = (
        data.pivot_table(index="City", columns="Month", values=RAINFALL_COL)
        .reindex(index=city_order, columns=MONTHS)
        .reset_index()
    )
    wide.columns.name = None  # drop the pivot axis label for clean display
    return wide
# ============================================================================ #
#  3. Core Analytics Engine
#
#  All functions accept a LONG-format DataFrame (optionally pre-filtered),
#  which makes composition trivial, e.g.:
#      total = compute_total_rainfall(filter_by_month(data, "Jun"))
# ============================================================================ #


@dataclass(frozen=True)
class RainfallExtreme:
    """A single extreme rainfall observation (peak or low)."""

    value_mm: float
    city: str
    month: str

    def __str__(self) -> str:
        return f"{self.value_mm:.1f} mm ({self.city}, {self.month})"


def compute_total_rainfall(data: pd.DataFrame) -> float:
    """Total rainfall in mm across every observation in ``data``."""
    return float(data[RAINFALL_COL].sum())


def compute_average_monthly_rainfall(data: pd.DataFrame) -> float:
    """Average rainfall in mm across all monthly observations in ``data``."""
    return float(data[RAINFALL_COL].mean()) if not data.empty else 0.0


def find_max_rainfall(data: pd.DataFrame) -> RainfallExtreme:
    """Return the wettest observation: its value, city and month."""
    if data.empty:
        raise ValueError("Cannot evaluate extremes on an empty dataset.")
    row = data.loc[data[RAINFALL_COL].idxmax()]
    return RainfallExtreme(float(row[RAINFALL_COL]), str(row["City"]), str(row["Month"]))


def find_min_rainfall(data: pd.DataFrame) -> RainfallExtreme:
    """Return the driest observation: its value, city and month."""
    if data.empty:
        raise ValueError("Cannot evaluate extremes on an empty dataset.")
    row = data.loc[data[RAINFALL_COL].idxmin()]
    return RainfallExtreme(float(row[RAINFALL_COL]), str(row["City"]), str(row["Month"]))


def average_by_month(data: pd.DataFrame) -> pd.DataFrame:
    """Average rainfall per calendar month, ordered Jan -> Dec."""
    by_month = data.groupby("Month", as_index=False)[RAINFALL_COL].mean()
    # GroupBy sorts alphabetically by default; restore the canonical Jan->Dec order.
    return by_month.set_index("Month").reindex(MONTHS).fillna(0).reset_index()


# -- Dynamic filtering -------------------------------------------------------- #

def _normalise_city(value: str) -> str:
    """Normalise a city label (trim, case-fold) for forgiving lookups."""
    return value.strip().casefold()


def filter_by_city(data: pd.DataFrame, city: str) -> pd.DataFrame:
    """Dynamically slice ``data`` to a single city (case-insensitive).

    Raises :class:`ValueError` if the city is not present in the data.
    """
    wanted = _normalise_city(city)
    mask = data["City"].map(_normalise_city) == wanted
    filtered = data[mask]
    if filtered.empty:
        raise ValueError(f"Unknown city: {city!r} (expected one of {CITIES}).")
    return filtered.copy()


def filter_by_month(data: pd.DataFrame, month: str) -> pd.DataFrame:
    """Dynamically slice ``data`` to a single month (case-insensitive).

    Accepts both 'Jun' and 'jun' / 'JUN'; raises :class:`ValueError` for
    anything not matching a calendar month.
    """
    wanted = month.strip().title()
    if wanted not in MONTHS:
        raise ValueError(f"Unknown month: {month!r} (expected one of {MONTHS}).")
    return data[data["Month"] == wanted].copy()


# Weight given to the short-term trend vs. the seasonal norm of the target month.
FORECAST_TREND_WEIGHT = 0.4
FORECAST_TREND_WINDOW = 4  # trailing months used for the local linear fit


def _linear_projection(points: list[tuple[int, float]], at: float) -> float:
    """Ordinary least-squares line through ``points``, evaluated at ``at``."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = (
        sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
        if denominator
        else 0.0
    )
    return y_mean + slope * (at - x_mean)


def compute_forecast(
    data: pd.DataFrame,
    city: str = "ALL",
    today: date | None = None,
) -> dict[str, str | float]:
    """Project next month's rainfall (mm) and classify it with an alert badge.

    Rainfall is strongly seasonal, so one straight line across Jan-Dec is
    misleading (it would predict monsoon rain for a dry month). Instead this
    fits a *local* linear regression to the trailing months ending at the
    current month, extrapolates one step ahead, and blends that trend with the
    target month's own seasonal norm. Gaps in sparse data are tolerated.
    """
    if city != "ALL":
        data = filter_by_city(data, city)
    monthly = data.groupby("Month")[RAINFALL_COL].mean().reindex(MONTHS).dropna()
    if monthly.empty:
        return {
            "month": "—",
            "value": 0.0,
            "alert": "No forecast available",
            "level": "none",
            "trend": "Add monthly observations to generate a forecast",
        }

    target_date = today or date.today()
    current_index = target_date.month - 1
    next_month = MONTHS[(current_index + 1) % len(MONTHS)]

    # Trailing window ending at the current month; x = months relative to it.
    points = [
        (offset, float(monthly[MONTHS[(current_index + offset) % len(MONTHS)]]))
        for offset in range(-(FORECAST_TREND_WINDOW - 1), 1)
        if MONTHS[(current_index + offset) % len(MONTHS)] in monthly.index
    ]
    trend_value = (
        max(0.0, _linear_projection(points, at=1)) if len(points) >= 2 else None
    )
    seasonal_value = float(monthly[next_month]) if next_month in monthly.index else None

    if trend_value is not None and seasonal_value is not None:
        projected = (
            FORECAST_TREND_WEIGHT * trend_value
            + (1 - FORECAST_TREND_WEIGHT) * seasonal_value
        )
        basis = f"Linear trend over {len(points)} trailing month(s) + seasonal norm"
    elif trend_value is not None:
        projected = trend_value
        basis = f"Linear trend over {len(points)} trailing month(s)"
    elif seasonal_value is not None:
        projected = seasonal_value
        basis = "Seasonal norm (not enough recent months for a trend)"
    else:
        projected = float(monthly.mean())
        basis = f"Mean of {len(monthly)} month(s) (sparse data)"

    if trend_value is None and seasonal_value is None:
        # Nothing relevant to the coming month: report the number, but don't
        # dress a guess up as a seasonal alert.
        alert, level = "Limited Data", "none"
    elif projected >= 200:
        alert, level = "Heavy Monsoon Expected", "heavy"
    elif projected >= 100:
        alert, level = "Elevated Rainfall", "elevated"
    elif projected < 40:
        alert, level = "Dry Season Phase", "dry"
    else:
        alert, level = "Moderate Rainfall", "moderate"

    return {
        "month": next_month,
        "value": projected,
        "alert": alert,
        "level": level,
        "trend": f"{basis} · illustrative estimate, not a weather forecast",
    }


def geocode_city(city: str) -> tuple[float, float]:
    """Resolve a city name to (latitude, longitude) via Open-Meteo geocoding.

    Prefers an Indian match (the bundled dataset's region) when several exist.
    """
    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search?" + (
        urllib.parse.urlencode(
            {"name": city, "count": 10, "language": "en", "format": "json"}
        )
    )
    request = urllib.request.Request(
        geocoding_url, headers={"User-Agent": "Pluvia-Analytics/1.0"}
    )
    with urllib.request.urlopen(request, timeout=12) as response:
        geocoding_payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(geocoding_payload, dict):
        raise ValueError("Open-Meteo returned an invalid geocoding response.")
    locations = geocoding_payload.get("results", [])
    if not locations:
        raise ValueError(f"Open-Meteo could not find a location for {city!r}.")

    indian_locations = [
        location for location in locations
        if location.get("country_code") == "IN"
    ]
    location = (indian_locations or locations)[0]
    latitude = float(location["latitude"])
    longitude = float(location["longitude"])
    return latitude, longitude


# Coordinates for the bundled cities: saves a geocoding round-trip per request.
CITY_COORDS: dict[str, tuple[float, float]] = {
    "mumbai": (19.0760, 72.8777),
    "delhi": (28.6139, 77.2090),
    "chennai": (13.0827, 80.2707),
    "kolkata": (22.5726, 88.3639),
    "bengaluru": (12.9716, 77.5946),
    "nagpur": (21.1458, 79.0882),
}
_geocode_cache: dict[str, tuple[float, float]] = {}


def resolve_city(city: str) -> tuple[float, float]:
    """(latitude, longitude) for a city: bundled table, then cache, then geocoding."""
    key = city.strip().casefold()
    if key in CITY_COORDS:
        return CITY_COORDS[key]
    if key not in _geocode_cache:
        _geocode_cache[key] = geocode_city(city)
    return _geocode_cache[key]


def fetch_open_meteo_monthly_rainfall(city: str) -> tuple[str, int, float]:
    """Fetch the latest complete observed monthly total for a city.

    Open-Meteo's historical archive has a short publication delay, so only a
    fully observed calendar month is returned; partial-month totals are never
    presented as complete rainfall measurements.
    """
    latitude, longitude = resolve_city(city)

    end_date = date.today() - timedelta(days=5)
    start_date = end_date - timedelta(days=100)
    archive_url = "https://archive-api.open-meteo.com/v1/archive?" + (
        urllib.parse.urlencode(
            {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "daily": "precipitation_sum",
                "timezone": "auto",
            }
        )
    )
    request = urllib.request.Request(
        archive_url, headers={"User-Agent": "Pluvia-Analytics/1.0"}
    )
    with urllib.request.urlopen(request, timeout=12) as response:
        archive_payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(archive_payload, dict):
        raise ValueError("Open-Meteo returned an invalid archive response.")
    daily = archive_payload.get("daily")
    if not isinstance(daily, dict):
        raise ValueError("Open-Meteo did not return daily precipitation data.")

    dates = daily.get("time", [])
    precipitation = daily.get("precipitation_sum", [])
    if not dates or len(dates) != len(precipitation):
        raise ValueError("Open-Meteo returned incomplete daily precipitation data.")

    grouped: dict[tuple[int, int], dict[date, float]] = {}
    for day_value, rainfall_value in zip(dates, precipitation):
        observation_date = date.fromisoformat(day_value)
        if rainfall_value is None:
            continue
        rainfall = float(rainfall_value)
        if not math.isfinite(rainfall) or rainfall < 0:
            raise ValueError("Open-Meteo returned an invalid rainfall value.")
        grouped.setdefault(
            (observation_date.year, observation_date.month), {}
        )[observation_date] = rainfall

    complete_months = []
    for (year, month), observations in grouped.items():
        expected_days = calendar.monthrange(year, month)[1]
        if len(observations) == expected_days:
            total = sum(observations.values())
            complete_months.append((year, month, total))
    if not complete_months:
        raise ValueError("Open-Meteo has no complete recent month available yet.")

    year, month, total = max(complete_months)
    return MONTHS[month - 1], year, total


# Errors a network call to Open-Meteo can raise; UI handlers fall back gracefully.
NETWORK_ERRORS = (
    OSError,
    TimeoutError,
    urllib.error.URLError,
    http.client.HTTPException,
    json.JSONDecodeError,
    KeyError,
    TypeError,
    ValueError,
)


def describe_weather_code(code: int | None) -> tuple[str, str, str]:
    """Map a WMO weather code to (label, Material icon, icon colour)."""
    sunny = "#FFD166"
    if code is None:
        return "Unknown", ft.Icons.WB_CLOUDY, COLOR_TEXT_MUTED
    if code == 0:
        return "Clear sky", ft.Icons.WB_SUNNY, sunny
    if code in (1, 2):
        return "Partly cloudy", ft.Icons.WB_CLOUDY, COLOR_ACCENT_SOFT
    if code == 3:
        return "Overcast", ft.Icons.CLOUD, COLOR_TEXT_MUTED
    if code in (45, 48):
        return "Fog", ft.Icons.CLOUD, COLOR_TEXT_MUTED
    if 51 <= code <= 57:
        return "Drizzle", ft.Icons.WATER_DROP, COLOR_ACCENT_SOFT
    if code in (65, 67, 82):
        return "Heavy rain", ft.Icons.UMBRELLA, COLOR_ACCENT
    if 61 <= code <= 67:
        return "Rain", ft.Icons.UMBRELLA, COLOR_ACCENT
    if code in (80, 81):
        return "Rain showers", ft.Icons.UMBRELLA, COLOR_ACCENT
    if 71 <= code <= 77 or code in (85, 86):
        return "Snow", ft.Icons.AC_UNIT, COLOR_ACCENT_SOFT
    if code >= 95:
        return "Thunderstorm", ft.Icons.THUNDERSTORM, COLOR_ACCENT
    return "Mixed", ft.Icons.WB_CLOUDY, COLOR_TEXT_MUTED


def _optional_float(value) -> float | None:
    """Coerce an API number to float, treating null/NaN/garbage as missing."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_daily_block(daily: dict) -> list[dict]:
    """Turn an Open-Meteo ``daily`` object into one dict per day (None = missing)."""
    times = daily.get("time") or []

    def column(name: str) -> list:
        values = list(daily.get(name) or [])
        return values + [None] * (len(times) - len(values))

    codes = column("weather_code")
    highs = column("temperature_2m_max")
    lows = column("temperature_2m_min")
    rain = column("precipitation_sum")
    probs = column("precipitation_probability_max")
    days = []
    for index, day_text in enumerate(times):
        code = _optional_float(codes[index])
        days.append(
            {
                "date": date.fromisoformat(day_text),
                "code": int(code) if code is not None else None,
                "tmax": _optional_float(highs[index]),
                "tmin": _optional_float(lows[index]),
                "rain": _optional_float(rain[index]),
                "prob": _optional_float(probs[index]),
            }
        )
    return days


def fetch_open_meteo_daily_forecast(city: str, days: int = 7) -> list[dict]:
    """Fetch a day-by-day forecast for ``city`` from the Open-Meteo forecast API.

    Returns one dict per day: ``date``, ``code`` (WMO), ``tmax``/``tmin`` (deg C),
    ``rain`` (mm) and ``prob`` (max precipitation probability, %). Missing
    values are ``None``. Blocking: call via ``asyncio.to_thread``.
    """
    latitude, longitude = resolve_city(city)
    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "daily": ",".join(
                [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "precipitation_probability_max",
                ]
            ),
            "forecast_days": days,
            "timezone": "auto",
        }
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Pluvia-Analytics/1.0"})
    with urllib.request.urlopen(request, timeout=12) as response:
        payload = json.loads(response.read().decode("utf-8"))
    daily = payload.get("daily") if isinstance(payload, dict) else None
    if not isinstance(daily, dict) or not daily.get("time"):
        raise ValueError("Open-Meteo did not return a daily forecast.")

    return _parse_daily_block(daily)


def fetch_open_meteo_past_week(cities: list[str], days: int = 7) -> dict[str, list[dict]]:
    """Observed daily weather for the last ``days`` completed days, per city.

    Uses Open-Meteo's ``past_days`` (best-estimate analysis data, not a raw
    weather-station reading) and fetches every city in a single request. Today
    is excluded because the day is not over. Cities that cannot be located are
    skipped; network errors propagate to the caller. Blocking: call via
    ``asyncio.to_thread``.
    """
    located: dict[str, tuple[float, float]] = {}
    for city in cities:
        try:
            located[city] = resolve_city(city)
        except ValueError:
            continue  # unknown place name: leave it out of the comparison
    if not located:
        raise ValueError("None of the cities could be located for the weather lookup.")

    names = list(located)
    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(
        {
            "latitude": ",".join(f"{located[n][0]:.4f}" for n in names),
            "longitude": ",".join(f"{located[n][1]:.4f}" for n in names),
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum",
            "past_days": days,
            "forecast_days": 1,
            "timezone": "auto",
        }
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Pluvia-Analytics/1.0"})
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))

    # One location -> an object; several -> a list in request order.
    blocks = payload if isinstance(payload, list) else [payload]
    if len(blocks) != len(names):
        raise ValueError("Open-Meteo returned an unexpected number of locations.")

    today = date.today()
    result: dict[str, list[dict]] = {}
    for name, block in zip(names, blocks):
        daily = block.get("daily") if isinstance(block, dict) else None
        if not isinstance(daily, dict):
            continue
        completed = [d for d in _parse_daily_block(daily) if d["date"] < today]
        if completed:
            result[name] = completed[-days:]
    if not result:
        raise ValueError("Open-Meteo returned no recent daily data.")
    return result


def _pdf_text(value: str) -> bytes:
    """Encode text for a PDF string using WinAnsi (cp1252), escaping delimiters.

    cp1252 keeps characters such as the em dash and middle dot intact instead
    of degrading them to '?'; anything outside it is replaced safely.
    """
    raw = value.encode("cp1252", "replace")
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def build_pdf_report(report: str) -> bytes:
    """Build a compact, dependency-free, single-page PDF summary.

    The first line is rendered as a bold title with a rule beneath it; lines
    shaped like ``Label: value`` get a bold label so KPIs scan easily.
    """
    lines = [
        wrapped
        for line in report.splitlines()
        for wrapped in (textwrap.wrap(line, width=88) or [""])
    ][:44]
    stamp = f"Generated {datetime.now():%Y-%m-%d %H:%M} by Pluvia Analytics"

    commands: list[bytes] = [b"BT", b"54 750 Td", b"20 TL"]
    for index, line in enumerate(lines):
        if index == 0:
            commands.append(b"/F2 17 Tf")
            commands.append(b"(" + _pdf_text(line) + b") Tj")
            commands.append(b"/F1 11 Tf")
            commands.append(b"28 TL")
            commands.append(b"T*")
            commands.append(b"15 TL")
            continue
        label, sep, rest = line.partition(": ")
        if sep and len(label) <= 32:
            commands.append(b"/F2 11 Tf")
            commands.append(b"(" + _pdf_text(label + ":") + b") Tj")
            commands.append(b"/F1 11 Tf")
            commands.append(b"(" + _pdf_text(" " + rest) + b") Tj")
        else:
            commands.append(b"/F1 11 Tf")
            commands.append(b"(" + _pdf_text(line) + b") Tj")
        commands.append(b"T*")
    commands.append(b"ET")
    # Footer + title rule, drawn outside the text block.
    commands.extend([
        b"0.28 0.79 0.89 RG 1.5 w 54 740 m 558 740 l S",
        b"0.45 0.45 0.45 rg BT /F1 8 Tf 54 36 Td (" + _pdf_text(stamp) + b") Tj ET",
    ])
    stream = b"\n".join(commands)

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold "
        b"/Encoding /WinAnsiEncoding >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
        + stream + b"\nendstream",
    ]
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(body)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)


# ============================================================================ #
#  4. Flet UI - Dark Mode + Glassmorphism
#
#  Visual language:
#    - Canvas  : deep midnight navy gradient
#    - Glass   : semi-transparent slate panels with backdrop blur + soft
#                borders and a diffuse shadow (frosted-glass idiom)
#    - Accents : vibrant cyan primary, soft sky-blue secondary
#    - Type    : crisp white for values, muted ice for labels/context
#
#  Key performance indicators, charts, and the record table refresh whenever
#  data changes or the City/Month filters change. The KPI row reflows via
#  ResponsiveRow breakpoints so it stacks (xs) -> two-up (sm) -> four-across (lg).
# ============================================================================ #

# -- Design tokens ------------------------------------------------------------ #

COLOR_BG = "#0B132B"           # deep midnight navy   (page canvas)
COLOR_BG_DEEP = "#0E1730"      # navy lift offset for the background wash
COLOR_GLASS = "#1C2541"        # slate glass          (panel fill)
COLOR_ACCENT = "#48CAE4"       # vibrant cyan         (primary accent)
COLOR_ACCENT_SOFT = "#90E0EF"  # soft sky blue        (secondary accent)
COLOR_TEXT = "#FFFFFF"         # crisp white
COLOR_TEXT_MUTED = "#A0C4FF"   # muted ice
COLOR_ON_ACCENT = "#052033"    # deep navy for glyphs sitting on accent fills

GLASS_OPACITY = 0.46
BORDER_OPACITY = 0.22
BLUR_SIGMA = 26

KPI_COL = {"xs": 12, "sm": 6, "lg": 3}   # responsive breakpoints for cards
FILTER_COL = {"xs": 12, "md": 6, "lg": 4}  # responsive breakpoints for filters

# Forecast alert badge fill per level (stays inside the cyan/ice palette; the
# only warm tone is reserved for the "dry" signal so it reads at a glance).
FORECAST_BADGE_COLORS = {
    "heavy": COLOR_ACCENT,
    "elevated": COLOR_ACCENT_SOFT,
    "moderate": COLOR_TEXT_MUTED,
    "dry": "#FFD166",
    "none": COLOR_TEXT_MUTED,
}


def _accent_gradient(accent: str = COLOR_ACCENT) -> ft.LinearGradient:
    """Diagonal accent gradient used for icon bubbles and glows."""
    return ft.LinearGradient(
        begin=ft.Alignment(-1, -1),
        end=ft.Alignment(1, 1),
        colors=[accent, COLOR_ACCENT_SOFT],
    )


def _glass_shadow() -> ft.BoxShadow:
    """Soft diffuse shadow that lifts glass panels off the canvas."""
    return ft.BoxShadow(
        blur_radius=30,
        spread_radius=-8,
        color=ft.Colors.with_opacity(0.35, "#000000"),
        offset=ft.Offset(0, 10),
    )


def glass_container(
    content,
    *,
    padding: int = 20,
    radius: int = 20,
    bg_opacity: float = GLASS_OPACITY,
    border_opacity: float = BORDER_OPACITY,
    blur_sigma: int = BLUR_SIGMA,
    shadow: bool = True,
    hoverable: bool = False,
    col=None,
) -> ft.Container:
    """Build a reusable frosted-glass panel.

    The semi-transparent slate fill plus a backdrop blur lets the navy
    gradient behind the panel show through - the core glassmorphism effect.
    """
    panel = ft.Container(
        content=content,
        padding=ft.Padding.all(padding),
        bgcolor=ft.Colors.with_opacity(bg_opacity, COLOR_GLASS),
        blur=ft.Blur(0, blur_sigma),
        border=ft.Border.all(
            1, ft.Colors.with_opacity(border_opacity, COLOR_ACCENT_SOFT)
        ),
        border_radius=ft.BorderRadius.all(radius),
        shadow=_glass_shadow() if shadow else None,
        col=col,
    )
    if hoverable:
        transition = ft.Animation(180, ft.AnimationCurve.EASE_OUT)
        panel.animate = transition
        panel.animate_scale = transition

        def _set_hover(event) -> None:
            hovered = event.data == "true"
            panel.scale = 1.012 if hovered else 1.0
            panel.bgcolor = ft.Colors.with_opacity(
                min(bg_opacity + (0.10 if hovered else 0), 1.0), COLOR_GLASS
            )
            panel.border = ft.Border.all(
                1,
                ft.Colors.with_opacity(
                    min(border_opacity + (0.18 if hovered else 0), 1.0),
                    COLOR_ACCENT_SOFT,
                ),
            )
            panel.update()

        panel.on_hover = _set_hover
    return panel


def icon_bubble(icon_name: str, accent: str = COLOR_ACCENT, size: float = 46) -> ft.Container:
    """Circular accent-gradient medallion that hosts a KPI icon."""
    return ft.Container(
        width=size,
        height=size,
        alignment=ft.Alignment.CENTER,
        gradient=_accent_gradient(accent),
        border_radius=ft.BorderRadius.all(size / 2),
        shadow=ft.BoxShadow(
            blur_radius=18,
            spread_radius=-4,
            color=ft.Colors.with_opacity(0.55, accent),
        ),
        content=ft.Icon(icon=icon_name, size=size * 0.52, color=COLOR_ON_ACCENT),
    )


def _intensity_tier(avg_mm: float) -> tuple[str, str]:
    """Map average monthly rainfall to a (dynamic icon, accent) pair."""
    if avg_mm <= 40:
        return ft.Icons.WATER_DROP, COLOR_ACCENT_SOFT
    if avg_mm <= 120:
        return ft.Icons.WATER_DROP, COLOR_ACCENT
    if avg_mm <= 250:
        return ft.Icons.WATER, COLOR_ACCENT
    return ft.Icons.THUNDERSTORM, COLOR_ACCENT_SOFT


def compute_kpis(data: pd.DataFrame, city: str = "ALL", month: str = "ALL") -> dict:
    """Pure analytics bundle for the current filter scope (unit-testable).

    Applies the requested City/Month slices to ``data`` and aggregates the
    four headline metrics plus a human-readable scope label.
    """
    filtered = data
    if city != "ALL":
        filtered = filter_by_city(filtered, city)
    if month != "ALL":
        filtered = filter_by_month(filtered, month)

    place = "All Cities" if city == "ALL" else city
    period = "All Months" if month == "ALL" else month

    return {
        "total": compute_total_rainfall(filtered),
        "avg": compute_average_monthly_rainfall(filtered),
        "peak": find_max_rainfall(filtered) if not filtered.empty else None,
        "trough": find_min_rainfall(filtered) if not filtered.empty else None,
        "rows": len(filtered),
        "context": f"{place}  \u2022  {period}",
    }
# -- Widget builders ------------------------------------------------------------ #

def _field_label(text: str) -> ft.Text:
    """Tiny uppercase label used above form controls."""
    return ft.Text(text, size=11, weight=ft.FontWeight.W_600, color=COLOR_TEXT_MUTED)


def _glass_dropdown(
    value: str | None,
    options: list[ft.DropdownOption],
    on_select,
) -> ft.Dropdown:
    """Frosted dropdown styled from the glass design tokens."""
    return ft.Dropdown(
        value=value,
        options=options,
        on_select=on_select,
        expand=True,
        dense=False,
        filled=True,
        fill_color=ft.Colors.with_opacity(0.10, COLOR_TEXT),
        bgcolor=ft.Colors.with_opacity(0.12, COLOR_GLASS),
        border=ft.OutlineInputBorder(
            side=ft.BorderSide(
                width=1, color=ft.Colors.with_opacity(0.35, COLOR_ACCENT_SOFT)
            ),
            border_radius=ft.BorderRadius.all(12),
        ),
        color=COLOR_TEXT,
    )


def _kpi_card(
    title: str,
    icon_name: str,
    accent: str,
    subtitle_hint: str,
) -> tuple[ft.Container, dict]:
    """Build one glassmorphism KPI card.

    Returns ``(card, refs)`` where ``refs`` exposes the mutable value
    text, subtitle text, icon and icon-bubble gradient so ``refresh()`` can
    update the card in place without rebuilding the control tree.
    """
    value_text = ft.Text("—", size=25, weight=ft.FontWeight.BOLD, color=COLOR_TEXT)
    sub_text = ft.Text(subtitle_hint, size=11, color=COLOR_TEXT_MUTED, max_lines=2)
    bubble = icon_bubble(icon_name, accent, size=44)

    card = glass_container(
        ft.Column(
            [
                ft.Row(
                    [bubble, _field_label(title)],
                    spacing=10,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                value_text,
                sub_text,
            ],
            spacing=8,
        ),
        padding=14,
        radius=20,
        hoverable=True,
        col=KPI_COL,
    )
    refs = {
        "value": value_text,
        "sub": sub_text,
        "bubble": bubble,
        "icon_name": icon_name,
        "accent": accent,
    }
    return card, refs
def build_kpi_row() -> tuple[ft.ResponsiveRow, dict[str, dict]]:
    """Build the four-card KPI row; returns the row and its mutable refs."""
    cards, refs = [], {}
    for key, title, icon, accent in (
        ("total", "TOTAL RAINFALL", ft.Icons.WATER_DROP, COLOR_ACCENT),
        ("avg", "AVG MONTHLY", ft.Icons.WATER, COLOR_ACCENT_SOFT),
        ("peak", "PEAK RAINFALL", ft.Icons.THUNDERSTORM, COLOR_ACCENT),
        ("min", "MINIMUM", ft.Icons.WB_SUNNY, COLOR_ACCENT_SOFT),
    ):
        card, card_refs = _kpi_card(title, icon, accent, "—")
        cards.append(card)
        refs[key] = card_refs

    row = ft.ResponsiveRow(cards, spacing=14, run_spacing=14)
    return row, refs


def build_app_header(export_button: ft.Control) -> ft.Container:
    """Top navigation bar: app identity and a report-export action."""
    logo = ft.Container(
        width=52,
        height=52,
        alignment=ft.Alignment.CENTER,
        gradient=_accent_gradient(),
        border_radius=ft.BorderRadius.all(16),
        shadow=ft.BoxShadow(
            blur_radius=22,
            spread_radius=-4,
            color=ft.Colors.with_opacity(0.6, COLOR_ACCENT),
        ),
        content=ft.Icon(icon=ft.Icons.WATER_DROP, size=28, color=COLOR_ON_ACCENT),
    )
    return glass_container(
        ft.Row(
            [
                logo,
                ft.Column(
                    [
                        ft.Text(
                            "Pluvia Analytics",
                            size=20,
                            weight=ft.FontWeight.BOLD,
                            color=COLOR_TEXT,
                        ),
                        ft.Text(
                            "Indian city rainfall · 1991–2020 monthly averages",
                            size=12.5,
                            color=COLOR_TEXT_MUTED,
                        ),
                    ],
                    spacing=2,
                    expand=True,
                ),
                export_button,
            ],
            spacing=14,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=14,
        radius=22,
        bg_opacity=0.38,
        blur_sigma=20,
    )


def build_filter_bar(
    city_drop: ft.Dropdown,
    month_drop: ft.Dropdown,
    reset_button: ft.IconButton,
    sync_button: ft.IconButton,
    context_text: ft.Text,
) -> ft.Container:
    """Responsive filter bar: City + Month dropdowns and a live scope pill."""
    return glass_container(
        ft.ResponsiveRow(
            [
                ft.Column(
                    [_field_label("CITY"), city_drop],
                    spacing=6,
                    col=FILTER_COL,
                ),
                ft.Column(
                    [_field_label("MONTH"), month_drop],
                    spacing=6,
                    col=FILTER_COL,
                ),
                ft.Column(
                    [
                        _field_label("SCOPE"),
                        ft.Row(
                            [context_text, sync_button, reset_button],
                            spacing=8,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ],
                    spacing=6,
                    col=FILTER_COL,
                ),
            ],
            spacing=16,
            run_spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=14,
        radius=20,
        bg_opacity=0.40,
        blur_sigma=20,
    )


def build_weather_day_card(day: dict, is_today: bool = False) -> ft.Container:
    """One glass tile in the daily-weather strip."""
    label, icon_name, icon_color = describe_weather_code(day["code"])
    day_date: date = day["date"]
    high = f"{day['tmax']:.0f}°" if day["tmax"] is not None else "—"
    low = f"{day['tmin']:.0f}°" if day["tmin"] is not None else "—"
    rain = f"{day['rain']:.1f} mm" if day["rain"] is not None else "— mm"
    chance = f"{day['prob']:.0f}%" if day["prob"] is not None else "—"
    tile = glass_container(
        ft.Column(
            [
                ft.Text(
                    "Today" if is_today else day_date.strftime("%a"),
                    size=13,
                    weight=ft.FontWeight.BOLD,
                    color=COLOR_ACCENT if is_today else COLOR_TEXT,
                ),
                ft.Text(f"{day_date.day} {day_date.strftime('%b')}", size=11, color=COLOR_TEXT_MUTED),
                ft.Icon(icon=icon_name, size=34, color=icon_color),
                ft.Text(label, size=11, color=COLOR_TEXT_MUTED, text_align=ft.TextAlign.CENTER, max_lines=1),
                ft.Row(
                    [
                        ft.Text(high, size=16, weight=ft.FontWeight.BOLD, color=COLOR_TEXT),
                        ft.Text(low, size=13, color=COLOR_TEXT_MUTED),
                    ],
                    spacing=6,
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
                ft.Row(
                    [
                        ft.Icon(icon=ft.Icons.WATER_DROP, size=12, color=COLOR_ACCENT),
                        ft.Text(
                            rain if day["prob"] is None else f"{rain} · {chance}",
                            size=11,
                            color=COLOR_ACCENT_SOFT,
                        ),
                    ],
                    spacing=3,
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
            ],
            spacing=5,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=12,
        radius=18,
        bg_opacity=0.56 if is_today else 0.34,
        border_opacity=0.60 if is_today else BORDER_OPACITY,
        blur_sigma=16,
        hoverable=True,
    )
    tile.width = 118
    return tile


def build_context_text() -> ft.Text:
    """The 'All Cities • All Months' scope label (mutable)."""
    return ft.Text(
        "All Cities  •  All Months",
        size=13,
        weight=ft.FontWeight.W_600,
        color=COLOR_ACCENT_SOFT,
        expand=True,
    )
# ============================================================================ #
#  5. Visualization Engine (Matplotlib -> base64 PNG)
#
#  flet 1.0.4 does not ship native chart widgets, so charts are rendered with
#  Matplotlib (headless "Agg" backend) to transparent-background PNGs and
#  streamed into the glass panels via ft.Image(src=base64). Re-rendering on
#  every data or filter change keeps the dashboard fully reactive.
#
#  Chart style follows the app palette: cyan/sky accents, transparent canvas,
#  white grid lines and labels.
# ============================================================================ #

import matplotlib

matplotlib.use("Agg")  # headless-safe renderer (no GUI backend needed)
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# Consistency knobs for every chart.
CHART_DPI = 115
CHART_FIGSIZE = (5.2, 2.8)
CHART_GRID = mcolors.to_rgba(COLOR_TEXT, alpha=0.16)
CHART_TICK = mcolors.to_rgba(COLOR_TEXT, alpha=0.82)
CHART_SPINE = mcolors.to_rgba(COLOR_TEXT, alpha=0.30)
CHART_BAR_DIM = mcolors.to_rgba(COLOR_ACCENT, alpha=0.28)    # non-selected bars
CHART_BAR_ACTIVE = mcolors.to_rgba(COLOR_ACCENT, alpha=1.0)  # active/selected bar


def _figure_to_base64(fig) -> str:
    """Render a Matplotlib figure to a base64-encoded PNG (transparent)."""
    buffer = io.BytesIO()
    try:
        fig.savefig(
            buffer,
            format="png",
            transparent=True,
            dpi=CHART_DPI,
            bbox_inches="tight",
            pad_inches=0.12,
        )
    finally:
        plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _style_axes(ax, *, grid_axis: str = "y") -> None:
    """Apply the Pluvia glass-chart look: transparent canvas, white ink."""
    ax.set_facecolor("none")          # transparent plot area
    for spine in ax.spines.values():
        spine.set_visible(False)      # drop the box frame
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(CHART_SPINE)
    if grid_axis == "y":
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(CHART_SPINE)
    ax.grid(
        axis=grid_axis,
        color=CHART_GRID,
        linestyle=(0, (1, 1.6)),
        linewidth=0.7,
        alpha=1.0,
        zorder=0,
    )
    ax.tick_params(axis="both", colors=CHART_TICK, labelsize=8.5, length=0)
    ax.margins(x=0.02)


def _period_desc(month: str) -> str:
    """Human label for the month portion of a chart subtitle."""
    return f"{month} only" if month != "ALL" else "all 12 months"


def render_city_bar_chart(data: pd.DataFrame, city: str = "ALL", month: str = "ALL") -> str:
    """Bar chart: total rainfall per city for the active filter scope.

    Every city keeps a bar so comparisons always work; the selected city
    (if any) is highlighted in sky blue while the rest dim to translucent
    cyan. Returns a base64 PNG string ready for ``ft.Image(src=...)``.
    """
    present_cities = data["City"].dropna().astype(str).unique().tolist()
    cities = [name for name in CITIES if name in present_cities]
    cities.extend(sorted(name for name in present_cities if name not in CITIES))
    totals = {}
    for name in cities:
        subset = data[data["City"] == name]
        if month != "ALL":
            subset = subset[subset["Month"] == month]
        totals[name] = compute_total_rainfall(subset)

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    fig.patch.set_facecolor("none")

    names = list(totals)
    if not names:
        ax.text(
            0.5, 0.5, "No rainfall data available",
            ha="center", va="center", color=CHART_TICK,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return _figure_to_base64(fig)

    values = [totals[n] for n in names]
    bars = ax.bar(names, values, zorder=3, width=0.62)
    peak_value = max(values)
    for name, bar in zip(names, bars):
        if city == "ALL":
            bar.set_color(CHART_BAR_ACTIVE if totals[name] == peak_value else CHART_BAR_DIM)
        else:
            bar.set_color(COLOR_ACCENT_SOFT if name == city else CHART_BAR_DIM)

    for name, value in zip(names, values):
        ax.annotate(
            f"{value:,.0f}",
            xy=(name, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=mcolors.to_rgba(COLOR_ACCENT_SOFT, alpha=0.95),
        )

    ax.set_ylabel("Total (mm)", fontsize=9, color=mcolors.to_rgba(COLOR_TEXT_MUTED, alpha=0.9))
    _style_axes(ax)
    ax.set_ylim(0, peak_value * 1.18 if peak_value else 1)
    return _figure_to_base64(fig)


def render_monthly_line_chart(data: pd.DataFrame, city: str = "ALL", month: str = "ALL") -> str:
    """Line chart: 12-month rainfall trend for the active scope.

    With 'All Cities' the line is the multi-city monthly average; with a
    selected city the city's own series is drawn bold cyan with the all-city
    average shown faintly for context. The selected month (if any) is marked
    with a vertical band and an emphasised point.
    """
    x_pos = list(range(len(MONTHS)))

    if city == "ALL" or city not in data["City"].values:
        primary = average_by_month(data)[RAINFALL_COL].tolist()
        secondary = None
    else:
        subset = filter_by_city(data, city)
        primary = average_by_month(subset)[RAINFALL_COL].tolist()
        secondary = average_by_month(data)[RAINFALL_COL].tolist()

    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    fig.patch.set_facecolor("none")

    ax.fill_between(
        x_pos, primary, color=CHART_BAR_ACTIVE, alpha=0.10, zorder=1, linewidth=0
    )
    ax.plot(
        x_pos, primary, color=COLOR_ACCENT, linewidth=2.4, marker="o",
        markersize=4.5, markerfacecolor=COLOR_ACCENT, markeredgewidth=0, zorder=3,
    )
    if secondary is not None:
        ax.plot(
            x_pos, secondary, color=COLOR_ACCENT_SOFT, linewidth=1.2,
            linestyle="--", alpha=0.55, zorder=2,
        )

    if month != "ALL":
        idx = MONTHS.index(month)
        ax.axvspan(idx - 0.35, idx + 0.35, color=COLOR_ACCENT_SOFT, alpha=0.13, zorder=0)
        ax.scatter(
            [idx], [primary[idx]], s=64, color=COLOR_ACCENT_SOFT,
            edgecolor=COLOR_TEXT, linewidth=1.2, zorder=4,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(MONTHS, fontsize=8.5, color=mcolors.to_rgba(COLOR_TEXT_MUTED, alpha=0.95))
    ax.set_ylabel("Rainfall (mm)", fontsize=9, color=mcolors.to_rgba(COLOR_TEXT_MUTED, alpha=0.9))
    ax.set_ylim(0, max(primary) * 1.22 if max(primary) else 1)
    _style_axes(ax)

    if secondary is not None:
        ax.plot([], [], color=COLOR_ACCENT, linewidth=2.4, label=city)
        ax.plot([], [], color=COLOR_ACCENT_SOFT, linewidth=1.2, linestyle="--", alpha=0.7,
                label="All cities avg")
        ax.legend(
            loc="upper left", frameon=False, fontsize=8,
            labelcolor=COLOR_TEXT_MUTED, ncol=2,
        )

    return _figure_to_base64(fig)
WEEK_LINE_COLORS = [
    "#48CAE4", "#FFD166", "#80ED99", "#FF8FA3", "#C77DFF", "#F4A261", "#90E0EF", "#E9ECEF",
]
WEEK_METRICS = {
    "rain": ("rain", "Rainfall (mm)", "Rainfall"),
    "temp": ("tmax", "Max temperature (°C)", "Max temperature"),
}


def render_past_week_chart(
    series: dict[str, list[dict]],
    metric: str = "rain",
    focus: str = "ALL",
    message: str = "No weather data yet",
) -> str:
    """Multi-city line chart of the last 7 observed days (one line per city).

    The focused city is drawn bold while the others dim, so a selected city
    stays readable against the rest. ``message`` is shown when there is no data.
    """
    field, y_label, _ = WEEK_METRICS.get(metric, WEEK_METRICS["rain"])
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    fig.patch.set_facecolor("none")

    if not series:
        ax.text(
            0.5, 0.5, message, ha="center", va="center",
            color=CHART_TICK, transform=ax.transAxes,
        )
        ax.set_axis_off()
        return _figure_to_base64(fig)

    reference = max(series.values(), key=len)
    x_pos = list(range(len(reference)))
    peak = 0.0
    for index, (city, days) in enumerate(series.items()):
        by_date = {day["date"]: day for day in days}
        values = [
            (by_date.get(ref["date"]) or {}).get(field) for ref in reference
        ]
        values = [math.nan if v is None else float(v) for v in values]
        finite = [v for v in values if math.isfinite(v)]
        peak = max(peak, max(finite, default=0.0))
        highlighted = focus == "ALL" or city == focus
        ax.plot(
            x_pos, values,
            color=WEEK_LINE_COLORS[index % len(WEEK_LINE_COLORS)],
            linewidth=2.8 if city == focus else 1.7,
            alpha=1.0 if highlighted else 0.38,
            marker="o", markersize=4.2 if city == focus else 3.2,
            markeredgewidth=0, label=city,
            zorder=4 if city == focus else 3,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [d["date"].strftime("%a %d") for d in reference],
        fontsize=8.2, color=mcolors.to_rgba(COLOR_TEXT_MUTED, alpha=0.95),
    )
    ax.set_ylabel(y_label, fontsize=9, color=mcolors.to_rgba(COLOR_TEXT_MUTED, alpha=0.9))
    if field == "rain":
        ax.set_ylim(0, peak * 1.2 if peak > 0 else 1)
    _style_axes(ax)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False,
        fontsize=8, labelcolor=COLOR_TEXT_MUTED, ncol=min(len(series), 6),
    )
    return _figure_to_base64(fig)


VIZ_COL = {"xs": 12, "md": 6}  # charts stack on phones, pair up on desktop


def build_viz_card(
    title: str,
    icon_name: str,
    image: ft.Image,
    subtitle: ft.Text,
) -> ft.Container:
    """Glass panel hosting one chart + a live scope subtitle."""
    return glass_container(
        ft.Column(
            [
                ft.Row(
                    [
                        icon_bubble(icon_name, accent=COLOR_ACCENT, size=38),
                        ft.Column(
                            [_field_label(title), subtitle],
                            spacing=2,
                            expand=True,
                        ),
                    ],
                    spacing=10,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Container(
                    image,
                    height=210,
                    expand=True,
                    animate=ft.Animation(180, ft.AnimationCurve.EASE_OUT),
                ),
            ],
            spacing=12,
            expand=True,
        ),
        padding=12,
        radius=20,
        hoverable=True,
        col=VIZ_COL,
    )


def build_viz_section(
    bar_image: ft.Image,
    line_image: ft.Image,
) -> tuple[ft.ResponsiveRow, dict[str, ft.Text]]:
    """Responsive row of the two glass chart panels.

    Returns ``(row, refs)`` where ``refs`` maps ``"bar"`` / ``"line"`` to the
    mutable subtitle texts the filter refresh pipeline updates.
    """
    bar_subtitle = ft.Text(
        "rendering…", size=12, color=COLOR_TEXT_MUTED, max_lines=2
    )
    line_subtitle = ft.Text(
        "rendering…", size=12, color=COLOR_TEXT_MUTED, max_lines=2
    )
    bar_card = build_viz_card(
        "CITY RAINFALL COMPARISON", ft.Icons.BAR_CHART, bar_image, bar_subtitle
    )
    line_card = build_viz_card(
        "MONTHLY RAINFALL TREND", ft.Icons.SHOW_CHART, line_image, line_subtitle
    )
    row = ft.ResponsiveRow([bar_card, line_card], spacing=14, run_spacing=14)
    return row, {"bar": bar_subtitle, "line": line_subtitle}


# -- Application ---------------------------------------------------------------- #

def main(page: ft.Page) -> None:
    """Assemble the responsive Pluvia glassmorphism dashboard."""
    page.title = "Pluvia Analytics"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = COLOR_BG
    page.padding = 0
    page.spacing = 0
    page.scroll = ft.ScrollMode.AUTO

    data = generate_sample_data()

    state = {
        "city": "ALL",
        "month": "ALL",
        "table_page": 0,
        "filter_transition": 0,
        "weather_city": None,     # city whose forecast is shown / last requested
        "weather_request": 0,     # token so a slow, stale response is discarded
        "past_cities": None,      # tuple of cities the past-week data was requested for
        "past_request": 0,        # stale-response token for the past-week fetch
        "past_metric": "rain",    # "rain" | "temp" for the week chart
        "past_view": None,        # last rendered (tile city, focus, metric, data stamp)
    }

    city_drop = _glass_dropdown(
        "ALL",
        [ft.DropdownOption(key="ALL", text="All Cities")]
        + [ft.DropdownOption(key=city, text=city) for city in CITIES],
        on_select=lambda e: page.run_task(_change_filter, e, "city"),
    )
    month_drop = _glass_dropdown(
        "ALL",
        [ft.DropdownOption(key="ALL", text="All Months")]
        + [ft.DropdownOption(key=month, text=month) for month in MONTHS],
        on_select=lambda e: page.run_task(_change_filter, e, "month"),
    )
    reset_button = ft.IconButton(
        icon=ft.Icons.REFRESH,
        icon_color=COLOR_ACCENT,
        icon_size=20,
        width=48,
        height=48,
        tooltip="Reset filters",
        bgcolor=ft.Colors.with_opacity(0.12, COLOR_GLASS),
        on_click=lambda e: _reset_filters(),
    )
    sync_button = ft.IconButton(
        icon=ft.Icons.SYNC,
        icon_color=COLOR_ACCENT,
        icon_size=20,
        width=48,
        height=48,
        tooltip="Sync Live Data from Open-Meteo",
        bgcolor=ft.Colors.with_opacity(0.12, COLOR_GLASS),
        on_click=lambda e: page.run_task(_sync_live_data, e),
    )
    export_button = ft.OutlinedButton(
        "Export Report",
        icon=ft.Icons.DOWNLOAD,
        on_click=lambda e: _open_export_dialog(e),
        height=44,
    )
    context_text = build_context_text()

    kpi_row, kpis = build_kpi_row()
    forecast_value = ft.Text(
        "—", size=24, weight=ft.FontWeight.BOLD, color=COLOR_TEXT
    )
    forecast_month_text = ft.Text(
        "Upcoming month", size=12, color=COLOR_TEXT_MUTED
    )
    forecast_alert = ft.Container(
        ft.Text("Forecast", size=12, weight=ft.FontWeight.W_600, color=COLOR_ON_ACCENT),
        bgcolor=COLOR_ACCENT_SOFT,
        padding=ft.Padding.symmetric(horizontal=12, vertical=7),
        border_radius=ft.BorderRadius.all(18),
    )
    forecast_trend = ft.Text(
        "Linear trend · illustrative estimate, not a weather forecast",
        size=11,
        color=COLOR_TEXT_MUTED,
    )
    forecast_card = glass_container(
        ft.ResponsiveRow(
            [
                ft.Column(
                    [
                        icon_bubble(ft.Icons.AUTO_AWESOME, COLOR_ACCENT, size=42),
                    ],
                    col={"xs": 2, "sm": 1},
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Column(
                    [
                        ft.Text(
                            "AI Forecast & Insight",
                            size=15,
                            weight=ft.FontWeight.W_600,
                            color=COLOR_TEXT,
                        ),
                        forecast_trend,
                    ],
                    spacing=3,
                    col={"xs": 10, "sm": 6},
                ),
                ft.Column(
                    [forecast_month_text, forecast_value],
                    spacing=2,
                    horizontal_alignment=ft.CrossAxisAlignment.START,
                    col={"xs": 6, "sm": 2},
                ),
                ft.Container(
                    forecast_alert,
                    alignment=ft.Alignment.CENTER_RIGHT,
                    col={"xs": 6, "sm": 3},
                ),
            ],
            spacing=14,
            run_spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=16,
        radius=20,
        bg_opacity=0.46,
    )

    weather_status = ft.Text(
        "Loading daily forecast…", size=12, color=COLOR_TEXT_MUTED, expand=True, max_lines=2
    )
    weather_days_row = ft.Row([], spacing=10, scroll=ft.ScrollMode.AUTO)
    weather_refresh = ft.IconButton(
        icon=ft.Icons.REFRESH,
        icon_color=COLOR_ACCENT,
        icon_size=20,
        width=48,
        height=48,
        tooltip="Refresh daily weather",
        bgcolor=ft.Colors.with_opacity(0.12, COLOR_GLASS),
        on_click=lambda e: page.run_task(_load_weather, True),
    )
    weather_panel = glass_container(
        ft.Column(
            [
                ft.Row(
                    [
                        icon_bubble(ft.Icons.WB_SUNNY, accent=COLOR_ACCENT, size=38),
                        weather_status,
                        weather_refresh,
                    ],
                    spacing=10,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                weather_days_row,
            ],
            spacing=10,
        ),
        padding=12,
        radius=20,
        bg_opacity=0.40,
        blur_sigma=20,
    )
    weather_section_label = ft.Row(
        [
            ft.Container(
                width=4, height=16, border_radius=ft.BorderRadius.all(3), bgcolor=COLOR_ACCENT
            ),
            ft.Text(
                "NEXT 7 DAYS · FORECAST",
                size=12,
                weight=ft.FontWeight.W_600,
                color=COLOR_TEXT_MUTED,
            ),
        ],
        spacing=8,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    past_status = ft.Text(
        "Loading last 7 days…", size=12, color=COLOR_TEXT_MUTED, max_lines=2
    )
    past_days_row = ft.Row([], spacing=10, scroll=ft.ScrollMode.AUTO)
    past_metric_drop = _glass_dropdown(
        "rain",
        [
            ft.DropdownOption(key="rain", text="Rainfall (mm)"),
            ft.DropdownOption(key="temp", text="Max temperature (°C)"),
        ],
        on_select=lambda e: _change_past_metric(e),
    )
    past_metric_drop.expand = False
    past_metric_drop.width = 200
    past_refresh = ft.IconButton(
        icon=ft.Icons.REFRESH,
        icon_color=COLOR_ACCENT,
        icon_size=20,
        width=48,
        height=48,
        tooltip="Refresh last 7 days",
        bgcolor=ft.Colors.with_opacity(0.12, COLOR_GLASS),
        on_click=lambda e: page.run_task(_load_past_week, True),
    )
    past_chart_img = ft.Image(
        src=render_past_week_chart({}, message="Loading last 7 days…"),
        fit=ft.BoxFit.CONTAIN,
        expand=True,
        gapless_playback=True,
    )
    past_panel = glass_container(
        ft.Column(
            [
                ft.ResponsiveRow(
                    [
                        ft.Row(
                            [
                                icon_bubble(ft.Icons.SHOW_CHART, accent=COLOR_ACCENT, size=38),
                                ft.Container(past_status, expand=True),
                            ],
                            spacing=10,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            col={"xs": 12, "md": 7},
                        ),
                        ft.Row(
                            [past_metric_drop, past_refresh],
                            spacing=8,
                            alignment=ft.MainAxisAlignment.END,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            col={"xs": 12, "md": 5},
                        ),
                    ],
                    spacing=10,
                    run_spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                past_days_row,
                ft.Container(past_chart_img, height=270),
            ],
            spacing=12,
        ),
        padding=12,
        radius=20,
        bg_opacity=0.40,
        blur_sigma=20,
    )
    past_section_label = ft.Row(
        [
            ft.Container(
                width=4, height=16, border_radius=ft.BorderRadius.all(3),
                bgcolor=COLOR_ACCENT_SOFT,
            ),
            ft.Text(
                "LAST 7 DAYS · OBSERVED",
                size=12,
                weight=ft.FontWeight.W_600,
                color=COLOR_TEXT_MUTED,
            ),
        ],
        spacing=8,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    bar_chart_img = ft.Image(
        src=render_city_bar_chart(data, "ALL", "ALL"),
        fit=ft.BoxFit.CONTAIN,
        expand=True,
        gapless_playback=True,
        animate_opacity=ft.Animation(180, ft.AnimationCurve.EASE_OUT),
    )
    line_chart_img = ft.Image(
        src=render_monthly_line_chart(data, "ALL", "ALL"),
        fit=ft.BoxFit.CONTAIN,
        expand=True,
        gapless_playback=True,
        animate_opacity=ft.Animation(180, ft.AnimationCurve.EASE_OUT),
    )
    viz_row, viz_subtitles = build_viz_section(bar_chart_img, line_chart_img)

    page_size = 10
    table_count = ft.Text(size=12, color=COLOR_TEXT_MUTED)
    table_page_text = ft.Text(size=12, color=COLOR_TEXT_MUTED)
    records_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("City")),
            ft.DataColumn(ft.Text("Month")),
            ft.DataColumn(ft.Text("Rainfall (mm)"), numeric=True),
            ft.DataColumn(ft.Text("Action")),
        ],
        rows=[],
        bgcolor=ft.Colors.with_opacity(0.08, COLOR_BG),
        heading_row_color=ft.Colors.with_opacity(0.42, COLOR_GLASS),
        data_row_color=ft.Colors.with_opacity(0.08, COLOR_GLASS),
        horizontal_lines=ft.BorderSide(
            1, ft.Colors.with_opacity(0.16, COLOR_ACCENT_SOFT)
        ),
        border_radius=12,
        column_spacing=24,
        horizontal_margin=12,
    )
    previous_page_button = ft.IconButton(
        icon=ft.Icons.CHEVRON_LEFT,
        width=48,
        height=48,
        tooltip="Previous page",
        on_click=lambda e: _change_table_page(-1),
    )
    next_page_button = ft.IconButton(
        icon=ft.Icons.CHEVRON_RIGHT,
        width=48,
        height=48,
        tooltip="Next page",
        on_click=lambda e: _change_table_page(1),
    )

    entry_city = ft.TextField(
        label="City",
        hint_text="e.g. Singapore",
        border_radius=12,
        autofocus=True,
    )
    entry_month = ft.Dropdown(
        label="Month",
        options=[ft.DropdownOption(key=month, text=month) for month in MONTHS],
        value=MONTHS[0],
        border_radius=12,
    )
    entry_rainfall = ft.TextField(
        label="Rainfall (mm)",
        hint_text="Enter a number",
        keyboard_type=ft.KeyboardType.NUMBER,
        border_radius=12,
    )
    entry_rainfall.on_submit = lambda e: _add_entry()  # Enter submits the form
    entry_error = ft.Text(color="#FF8A8A", size=12)
    add_dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text("Add rainfall record"),
        content=ft.Column(
            [entry_city, entry_month, entry_rainfall, entry_error],
            tight=True,
            spacing=12,
        ),
        actions=[
            ft.TextButton("Cancel", on_click=lambda e: page.pop_dialog()),
            ft.FilledButton("Add record", on_click=lambda e: _add_entry()),
        ],
        bgcolor=COLOR_GLASS,
    )
    export_dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text("Export report"),
        content=ft.Column(
            [
                ft.Text(
                    "Download the filtered records or a KPI summary for the current scope.",
                    size=13,
                    color=COLOR_TEXT_MUTED,
                ),
                ft.OutlinedButton(
                    "Filtered dataset · CSV",
                    icon=ft.Icons.TABLE_VIEW,
                    on_click=lambda e: page.run_task(_export_csv, e, True),
                    expand=True,
                ),
                ft.OutlinedButton(
                    "Rainfall summary · PDF",
                    icon=ft.Icons.PICTURE_AS_PDF,
                    on_click=lambda e: page.run_task(_export_summary_pdf, e),
                    expand=True,
                ),
                ft.OutlinedButton(
                    "Rainfall summary · TXT",
                    icon=ft.Icons.DESCRIPTION_OUTLINED,
                    on_click=lambda e: page.run_task(_export_summary_txt, e),
                    expand=True,
                ),
            ],
            tight=True,
            spacing=10,
        ),
        actions=[ft.TextButton("Close", on_click=lambda e: page.pop_dialog())],
        bgcolor=COLOR_GLASS,
    )
    summary_dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text("Rainfall summary"),
        content=ft.Text("", selectable=True, size=14, color=COLOR_TEXT),
        actions=[ft.TextButton("Close", on_click=lambda e: page.pop_dialog())],
        bgcolor=COLOR_GLASS,
    )

    file_picker = ft.FilePicker()
    page.services.append(file_picker)

    def _notify_success(message: str) -> None:
        page.show_dialog(
            ft.SnackBar(
                content=ft.Row(
                    [
                        ft.Icon(ft.Icons.CHECK_CIRCLE_OUTLINE, color=COLOR_ON_ACCENT),
                        ft.Text(
                            message,
                            color=COLOR_ON_ACCENT,
                            expand=True,
                            max_lines=2,
                        ),
                    ],
                    spacing=10,
                    tight=True,
                ),
                bgcolor=COLOR_ACCENT_SOFT,
                behavior=ft.SnackBarBehavior.FLOATING,
                duration=ft.Duration(seconds=3),
                show_close_icon=True,
                close_icon_color=COLOR_ON_ACCENT,
                margin=12,
                padding=ft.Padding.symmetric(horizontal=16, vertical=12),
            )
        )

    def _notify_error(message: str) -> None:
        page.show_dialog(
            ft.SnackBar(
                content=ft.Row(
                    [
                        ft.Icon(ft.Icons.ERROR_OUTLINE, color=COLOR_TEXT),
                        ft.Text(message, color=COLOR_TEXT, expand=True, max_lines=3),
                    ],
                    spacing=10,
                    tight=True,
                ),
                bgcolor="#7F1D2D",
                behavior=ft.SnackBarBehavior.FLOATING,
                duration=ft.Duration(seconds=6),
                show_close_icon=True,
                margin=12,
                padding=ft.Padding.symmetric(horizontal=16, vertical=12),
            )
        )

    def _filtered_data() -> pd.DataFrame:
        filtered = data
        if state["city"] != "ALL":
            filtered = filtered[filtered["City"] == state["city"]]
        if state["month"] != "ALL":
            filtered = filtered[filtered["Month"] == state["month"]]
        return filtered

    def _refresh_city_options() -> None:
        cities = data["City"].dropna().astype(str).unique().tolist()
        ordered_cities = [city for city in CITIES if city in cities]
        ordered_cities.extend(sorted(city for city in cities if city not in CITIES))
        city_drop.options = [ft.DropdownOption(key="ALL", text="All Cities")] + [
            ft.DropdownOption(key=city, text=city) for city in ordered_cities
        ]
        if state["city"] not in ordered_cities:
            state["city"] = "ALL"
            city_drop.value = "ALL"

    def _render_table() -> None:
        filtered = _filtered_data()
        total_rows = len(filtered)
        page_count = (total_rows + page_size - 1) // page_size
        state["table_page"] = min(state["table_page"], max(page_count - 1, 0))
        start = state["table_page"] * page_size
        visible_rows = filtered.iloc[start:start + page_size]
        records_table.rows = [
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(str(row["City"]))),
                    ft.DataCell(ft.Text(str(row["Month"]))),
                    ft.DataCell(ft.Text(f"{float(row[RAINFALL_COL]):,.1f}")),
                    ft.DataCell(
                        ft.IconButton(
                            icon=ft.Icons.DELETE_OUTLINE,
                            icon_color="#FF8A8A",
                            width=48,
                            height=48,
                            tooltip="Delete row",
                            on_click=lambda e, row_index=index: _delete_entry(row_index),
                        )
                    ),
                ]
            )
            for index, row in visible_rows.iterrows()
        ]
        table_count.value = f"{total_rows} record(s)"
        table_page_text.value = (
            f"Page {state['table_page'] + 1} of {page_count}"
            if page_count else "Page 0 of 0"
        )
        previous_page_button.disabled = state["table_page"] <= 0
        next_page_button.disabled = state["table_page"] >= page_count - 1

    def _change_table_page(offset: int) -> None:
        state["table_page"] = max(0, state["table_page"] + offset)
        _render_table()
        page.update()

    def _delete_entry(row_index: int) -> None:
        nonlocal data
        deleted = data.loc[row_index]
        data = data.drop(index=row_index).reset_index(drop=True)
        _refresh_city_options()
        _apply_filters()
        _notify_success(f"Deleted {deleted['City']} · {deleted['Month']} record")

    def _add_entry() -> None:
        nonlocal data
        city = (entry_city.value or "").strip()
        month = entry_month.value
        rainfall_value = (entry_rainfall.value or "").strip()
        if not city:
            entry_error.value = "Enter a city name."
            page.update()
            return
        if month not in MONTHS:
            entry_error.value = "Select a valid month."
            page.update()
            return
        try:
            rainfall = float(rainfall_value)
        except ValueError:
            entry_error.value = "Rainfall must be a valid number."
            page.update()
            return
        if not math.isfinite(rainfall) or rainfall < 0:
            entry_error.value = "Rainfall must be a finite, non-negative number."
            page.update()
            return

        existing_city = next(
            (
                value for value in data["City"].astype(str)
                if value.casefold() == city.casefold()
            ),
            None,
        )
        city = existing_city or city
        # One value per (City, Month): overwrite instead of double-counting totals.
        same_slot = data.index[(data["City"] == city) & (data["Month"] == month)]
        replaced = len(same_slot) > 0
        if replaced:
            data.loc[same_slot[0], RAINFALL_COL] = rainfall
            data = data.drop(index=same_slot[1:]).reset_index(drop=True)
        else:
            data = pd.concat(
                [data, pd.DataFrame(
                    [{"City": city, "Month": month, RAINFALL_COL: rainfall}],
                    columns=data.columns,
                )],
                ignore_index=True,
            )
        state["table_page"] = 0
        entry_city.value = ""
        entry_rainfall.value = ""
        entry_error.value = ""
        page.pop_dialog()
        _refresh_city_options()
        _apply_filters()
        _notify_success(
            f"{'Updated' if replaced else 'Added'} {city} · {month}: {rainfall:,.1f} mm"
        )

    def _open_add_dialog(event) -> None:
        entry_error.value = ""
        page.show_dialog(add_dialog)

    async def _save_bytes(title: str, file_name: str, ext: str, payload: bytes) -> bool:
        """Open the native save dialog and make sure the bytes reach disk.

        Web/mobile builds hand ``src_bytes`` to the platform; on desktop the
        picker only returns a path, so the file is written here (off the UI
        thread) with the right extension enforced.
        """
        saved_path = await file_picker.save_file(
            dialog_title=title,
            file_name=file_name,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=[ext],
            src_bytes=payload,
        )
        if not saved_path:
            return False
        if not getattr(page, "web", False):
            target = Path(saved_path)
            if target.suffix.lower() != f".{ext}":
                target = target.with_name(target.name + f".{ext}")
            try:
                await asyncio.to_thread(target.write_bytes, payload)
            except OSError as exc:
                _notify_error(f"Could not write {target.name}: {exc}")
                return False
        return True

    async def _export_csv(event, close_export_dialog: bool = False) -> None:
        filtered = _filtered_data()
        if filtered.empty:
            _notify_error("No records match the current filters — nothing to export.")
            return
        csv_bytes = filtered.to_csv(index=False).encode("utf-8-sig")
        if close_export_dialog:
            page.pop_dialog()
        if await _save_bytes(
            "Export filtered rainfall data", "pluvia_rainfall.csv", "csv", csv_bytes
        ):
            _notify_success(f"Exported {len(filtered)} filtered record(s) as CSV.")

    def _build_summary_text() -> str:
        filtered = _filtered_data()
        context = (
            f"{state['city'] if state['city'] != 'ALL' else 'All Cities'}"
            f" · {state['month'] if state['month'] != 'ALL' else 'All Months'}"
        )
        forecast = compute_forecast(data, state["city"])
        lines = [
            "PLUVIA ANALYTICS — RAINFALL SUMMARY",
            f"Scope: {context}",
            f"Records: {len(filtered)}",
        ]
        if filtered.empty:
            lines.append("No records match the current filters.")
        else:
            peak = find_max_rainfall(filtered)
            trough = find_min_rainfall(filtered)
            lines.extend(
                [
                    f"Total rainfall: {compute_total_rainfall(filtered):,.1f} mm",
                    f"Average rainfall: {compute_average_monthly_rainfall(filtered):,.1f} mm",
                    f"Maximum rainfall: {peak}",
                    f"Minimum rainfall: {trough}",
                ]
            )
        lines.extend(
            [
                "",
                f"Forecast ({forecast['month']}): {forecast['value']:,.1f} mm",
                f"Insight: {forecast['alert']}",
                str(forecast["trend"]),
            ]
        )
        return "\n".join(lines)

    async def _import_csv(event) -> None:
        selected_files = await file_picker.pick_files(
            dialog_title="Import rainfall spreadsheet",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["csv"],
            allow_multiple=False,
            with_data=True,
        )
        if not selected_files:
            return

        contents = selected_files[0].bytes
        if contents is None:
            _notify_error("The selected CSV could not be read. Please try again.")
            return
        try:
            imported = parse_rainfall_csv(contents)
        except ValueError as exc:
            _notify_error(str(exc))
            return

        nonlocal data
        known_cities = {
            city.casefold(): city for city in data["City"].astype(str).unique()
        }
        normalized_records = []
        for record in imported.to_dict(orient="records"):
            city = str(record["City"])
            canonical_city = known_cities.setdefault(city.casefold(), city)
            normalized_records.append(
                {**record, "City": canonical_city}
            )
        data = pd.concat(
            [data, pd.DataFrame(normalized_records, columns=data.columns)],
            ignore_index=True,
        )
        state["table_page"] = 0
        _refresh_city_options()
        _apply_filters()
        _notify_success(
            f"Imported {len(imported)} rainfall record(s) from {selected_files[0].name}"
        )

    def _generate_summary(event) -> None:
        report = _build_summary_text()
        summary_dialog.content = ft.Text(
            report, selectable=True, size=14, color=COLOR_TEXT
        )
        page.show_dialog(summary_dialog)

    def _open_export_dialog(event) -> None:
        page.show_dialog(export_dialog)

    async def _export_summary_pdf(event) -> None:
        report = _build_summary_text()
        page.pop_dialog()
        if await _save_bytes(
            "Export rainfall PDF report",
            "pluvia_rainfall_report.pdf",
            "pdf",
            build_pdf_report(report),
        ):
            _notify_success("Rainfall KPI summary exported as PDF.")

    async def _export_summary_txt(event) -> None:
        report = _build_summary_text().encode("utf-8")
        page.pop_dialog()
        if await _save_bytes(
            "Export rainfall text report", "pluvia_rainfall_report.txt", "txt", report
        ):
            _notify_success("Rainfall KPI summary exported as text.")

    async def _sync_live_data(event) -> None:
        nonlocal data
        city = state["city"]
        if city == "ALL":
            _notify_error("Select a city before syncing live weather data.")
            return
        sync_button.disabled = True
        sync_button.tooltip = f"Syncing {city} from Open-Meteo…"
        page.update()
        try:
            month, year, rainfall = await asyncio.to_thread(
                fetch_open_meteo_monthly_rainfall, city
            )
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            http.client.HTTPException,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            if isinstance(
                exc, (OSError, TimeoutError, urllib.error.URLError, http.client.HTTPException)
            ):
                reason = "Couldn't reach Open-Meteo (offline or timed out)."
            else:
                reason = str(exc)
            _notify_error(
                f"Live sync unavailable — showing bundled/manual data instead. {reason}"
            )
            return
        finally:
            sync_button.disabled = False
            sync_button.tooltip = "Sync Live Data from Open-Meteo"
            page.update()

        matching_city = next(
            (
                value for value in data["City"].astype(str)
                if value.casefold() == city.casefold()
            ),
            city,
        )
        same_month = (
            (data["City"] == matching_city) & (data["Month"] == month)
        )
        data = data.loc[~same_month].copy()
        data.loc[len(data)] = {
            "City": matching_city,
            "Month": month,
            RAINFALL_COL: rainfall,
        }
        data = data.reset_index(drop=True)
        state["table_page"] = 0
        _refresh_city_options()
        _apply_filters()
        _notify_success(
            f"Synced {matching_city} · {month} {year}: {rainfall:,.1f} mm "
            "from Open-Meteo."
        )

    weather_cache: dict[str, tuple[datetime, list[dict]]] = {}

    def _weather_city() -> str:
        """City whose forecast to show: the selected one, else the first city."""
        if state["city"] != "ALL":
            return state["city"]
        known = data["City"].dropna().astype(str).unique().tolist()
        return next((c for c in CITIES if c in known), known[0] if known else CITIES[0])

    def _show_weather(city: str, days: list[dict], fetched_at: datetime) -> None:
        weather_days_row.controls = [
            build_weather_day_card(day, is_today=index == 0)
            for index, day in enumerate(days)
        ]
        hint = " · select a city to change" if state["city"] == "ALL" else ""
        weather_status.value = (
            f"{city} · next {len(days)} days · updated {fetched_at:%H:%M}{hint}"
        )
        page.update()

    async def _load_weather(force: bool = False) -> None:
        """Fetch (or reuse a 15-minute cache of) the daily forecast, off the UI thread."""
        city = _weather_city()
        state["weather_city"] = city
        state["weather_request"] += 1
        request_id = state["weather_request"]

        cached = weather_cache.get(city)
        if cached and not force and datetime.now() - cached[0] < timedelta(minutes=15):
            _show_weather(city, cached[1], cached[0])
            return

        weather_refresh.disabled = True
        weather_status.value = f"{city} · loading daily forecast…"
        page.update()
        try:
            days = await asyncio.to_thread(fetch_open_meteo_daily_forecast, city)
        except NETWORK_ERRORS:
            if request_id == state["weather_request"]:
                weather_days_row.controls = []
                weather_status.value = (
                    f"{city} · daily weather unavailable (offline?). "
                    "Tap refresh to retry — your rainfall data is unaffected."
                )
            return
        finally:
            weather_refresh.disabled = False
            page.update()
        fetched_at = datetime.now()
        weather_cache[city] = (fetched_at, days)
        if request_id == state["weather_request"]:  # ignore stale responses
            _show_weather(city, days, fetched_at)

    past_cache: dict = {"stamp": None, "cities": (), "series": {}}

    def _known_cities() -> list[str]:
        """Dataset cities in display order (bundled first), capped for chart legibility."""
        present = data["City"].dropna().astype(str).unique().tolist()
        ordered = [c for c in CITIES if c in present]
        ordered.extend(sorted(c for c in present if c not in CITIES))
        return ordered[:8]

    def _render_past_week(force: bool = False) -> None:
        """Redraw tiles + chart from cached data (cheap; skipped if nothing changed)."""
        if past_cache["stamp"] is None:
            return
        tile_city = _weather_city()
        key = (tile_city, state["city"], state["past_metric"], past_cache["stamp"])
        if not force and state["past_view"] == key:
            return
        state["past_view"] = key
        series = past_cache["series"]
        days = series.get(tile_city, [])
        past_days_row.controls = [build_weather_day_card(day) for day in days]
        past_chart_img.src = render_past_week_chart(
            series, state["past_metric"], state["city"]
        )
        hint = " · select a city to change" if state["city"] == "ALL" else ""
        _, _, metric_name = WEEK_METRICS[state["past_metric"]]
        past_status.value = (
            f"{tile_city} · last {len(days)} days observed · chart: {metric_name}, "
            f"{len(series)} cities · updated {past_cache['stamp']:%H:%M}{hint}"
        )
        page.update()

    def _change_past_metric(event) -> None:
        state["past_metric"] = (event.data or "rain").strip() or "rain"
        _render_past_week(force=True)

    async def _load_past_week(force: bool = False) -> None:
        """Fetch the last 7 observed days for every city in one request (off-thread)."""
        cities = tuple(_known_cities())
        state["past_cities"] = cities
        state["past_request"] += 1
        request_id = state["past_request"]

        fresh = (
            past_cache["stamp"] is not None
            and past_cache["cities"] == cities
            and datetime.now() - past_cache["stamp"] < timedelta(minutes=30)
        )
        if fresh and not force:
            _render_past_week()
            return

        past_refresh.disabled = True
        past_status.value = f"Loading last 7 days for {len(cities)} cities…"
        page.update()
        try:
            series = await asyncio.to_thread(fetch_open_meteo_past_week, list(cities))
        except NETWORK_ERRORS:
            if request_id == state["past_request"]:
                past_cache.update(stamp=None, cities=(), series={})
                state["past_view"] = None
                past_days_row.controls = []
                past_chart_img.src = render_past_week_chart(
                    {}, message="Past-week data unavailable (offline?)"
                )
                past_status.value = (
                    "Last 7 days unavailable (offline?). Tap refresh to retry — "
                    "your rainfall data is unaffected."
                )
            return
        finally:
            past_refresh.disabled = False
            page.update()
        if request_id == state["past_request"]:  # ignore stale responses
            past_cache.update(stamp=datetime.now(), cities=cities, series=series)
            _render_past_week(force=True)

    def _apply_filters() -> None:
        """Recompute KPIs for the active scope and refresh the controls."""
        filtered = _filtered_data()
        scope = compute_kpis(data, state["city"], state["month"]) if not filtered.empty else None
        total = scope["total"] if scope else 0.0
        average = scope["avg"] if scope else 0.0
        dynamic_icon, dynamic_accent = _intensity_tier(average)
        kpis["total"]["value"].value = f"{total:,.1f} mm"
        kpis["total"]["sub"].value = (
            f"{len(filtered)} observation(s) · avg {average:.1f} mm/mo"
        )
        kpis["total"]["bubble"].gradient = _accent_gradient(dynamic_accent)
        kpis["total"]["bubble"].content.icon = dynamic_icon

        kpis["avg"]["value"].value = f"{average:.1f} mm"
        kpis["avg"]["sub"].value = "per month · selected scope"

        forecast = compute_forecast(data, state["city"])
        forecast_value.value = f"{forecast['value']:,.1f} mm"
        forecast_month_text.value = f"Projected · {forecast['month']}"
        forecast_trend.value = str(forecast["trend"])
        forecast_alert.content.value = str(forecast["alert"])
        forecast_alert.bgcolor = FORECAST_BADGE_COLORS.get(
            str(forecast.get("level", "none")), COLOR_ACCENT_SOFT
        )

        if scope:
            kpis["peak"]["value"].value = f"{scope['peak'].value_mm:,.1f} mm"
            kpis["peak"]["sub"].value = f"{scope['peak'].city}  •  {scope['peak'].month}"
            kpis["min"]["value"].value = f"{scope['trough'].value_mm:,.1f} mm"
            kpis["min"]["sub"].value = f"{scope['trough'].city}  •  {scope['trough'].month}"
            context_text.value = scope["context"]
        else:
            kpis["peak"]["value"].value = "—"
            kpis["peak"]["sub"].value = "No records in this scope"
            kpis["min"]["value"].value = "—"
            kpis["min"]["sub"].value = "No records in this scope"
            context_text.value = (
                f"{state['city'] if state['city'] != 'ALL' else 'All Cities'}  •  "
                f"{state['month'] if state['month'] != 'ALL' else 'All Months'}"
            )

        # -- Charts: re-render both panels for the active filter scope -- #
        bar_chart_img.src = render_city_bar_chart(data, state["city"], state["month"])
        line_chart_img.src = render_monthly_line_chart(data, state["city"], state["month"])
        bar_chart_img.opacity = 1
        line_chart_img.opacity = 1

        scope_city = state["city"] if state["city"] != "ALL" else "All Cities"
        viz_subtitles["bar"].value = (
            f"Total (mm) per city · {_period_desc(state['month'])}"
            + (f" · focus {state['city']}" if state["city"] != "ALL" else "")
        )
        viz_subtitles["line"].value = (
            f"Series: {scope_city}"
            + (f" · {state['month']} highlighted" if state["month"] != "ALL" else "")
        )
        _render_table()
        page.update()
        if state["weather_city"] != _weather_city():
            page.run_task(_load_weather)
        if state["past_cities"] != tuple(_known_cities()):
            page.run_task(_load_past_week)  # first load, or a new city was added
        else:
            _render_past_week()  # city/metric changed: redraw from cache, no network

    async def _change_filter(event, which: str) -> None:
        # e.data carries the selected option key; keep the state in sync.
        state[which] = (event.data or "ALL").strip() or "ALL"
        state["filter_transition"] += 1
        transition = state["filter_transition"]
        bar_chart_img.opacity = 0.35
        line_chart_img.opacity = 0.35
        page.update()
        await asyncio.sleep(0.08)
        if transition != state["filter_transition"]:
            return
        _apply_filters()

    def _reset_filters() -> None:
        state["city"] = "ALL"
        state["month"] = "ALL"
        city_drop.value = "ALL"
        month_drop.value = "ALL"
        _apply_filters()

    page.floating_action_button = ft.FloatingActionButton(
        icon=ft.Icons.ADD,
        tooltip="Add rainfall record",
        bgcolor=COLOR_ACCENT,
        foreground_color=COLOR_ON_ACCENT,
        on_click=_open_add_dialog,
    )

    # -- Layout assembly (responsive) ------------------------------------------- #
    section_tick = ft.Container(
        width=4, height=16, border_radius=ft.BorderRadius.all(3), bgcolor=COLOR_ACCENT
    )
    kpi_section_label = ft.Row(
        [
            section_tick,
            ft.Text(
                "KEY PERFORMANCE INDICATORS",
                size=12,
                weight=ft.FontWeight.W_600,
                color=COLOR_TEXT_MUTED,
            ),
        ],
        spacing=8,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    viz_section_label = ft.Row(
        [
            ft.Container(
                width=4, height=16, border_radius=ft.BorderRadius.all(3),
                bgcolor=COLOR_ACCENT_SOFT,
            ),
            ft.Text(
                "DATA VISUALIZATION",
                size=12,
                weight=ft.FontWeight.W_600,
                color=COLOR_TEXT_MUTED,
            ),
        ],
        spacing=8,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    footer = glass_container(
        ft.Row(
            [
                ft.Icon(icon=ft.Icons.INFO, size=16, color=COLOR_ACCENT),
                ft.Text(
                    "Bundled data: Open-Meteo ERA5 1991–2020 monthly normals · Live sync uses the latest complete observed month · Daily weather from Open-Meteo (forecast + past-week analysis) · Nagpur normals are approximate.",
                    size=12,
                    color=COLOR_TEXT_MUTED,
                    expand=True,
                ),
            ],
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=12,
        radius=16,
        bg_opacity=0.34,
        blur_sigma=16,
    )

    page.add(
        ft.Container(
            expand=True,
            gradient=ft.LinearGradient(
                begin=ft.Alignment(-0.9, -1),
                end=ft.Alignment(0.9, 1),
                colors=[COLOR_BG, COLOR_BG_DEEP, COLOR_BG],
            ),
            content=ft.Column(
                [
                    build_app_header(export_button),
                    build_filter_bar(
                        city_drop, month_drop, reset_button, sync_button, context_text
                    ),
                    kpi_section_label,
                    kpi_row,
                    forecast_card,
                    weather_section_label,
                    weather_panel,
                    past_section_label,
                    past_panel,
                    viz_section_label,
                    viz_row,
                    ft.Column(
                        [
                            ft.Text(
                                "RAINFALL RECORDS",
                                size=12,
                                weight=ft.FontWeight.W_600,
                                color=COLOR_TEXT_MUTED,
                            ),
                            ft.ResponsiveRow(
                                [
                                    ft.Container(
                                        ft.FilledButton(
                                            "Generate Summary",
                                            icon=ft.Icons.DESCRIPTION_OUTLINED,
                                            on_click=_generate_summary,
                                            height=48,
                                            expand=True,
                                        ),
                                        col={"xs": 12, "sm": 6, "md": 3},
                                    ),
                                    ft.Container(
                                        ft.OutlinedButton(
                                            "Export CSV",
                                            icon=ft.Icons.DOWNLOAD,
                                            on_click=lambda e: page.run_task(_export_csv, e),
                                            height=48,
                                            expand=True,
                                        ),
                                        col={"xs": 12, "sm": 6, "md": 3},
                                    ),
                                    ft.Container(
                                        ft.OutlinedButton(
                                            "Import CSV",
                                            icon=ft.Icons.UPLOAD_FILE,
                                            on_click=lambda e: page.run_task(_import_csv, e),
                                            height=48,
                                            expand=True,
                                        ),
                                        col={"xs": 12, "sm": 6, "md": 3},
                                    ),
                                    ft.Container(
                                        ft.FilledButton(
                                            "Add Record",
                                            icon=ft.Icons.ADD,
                                            on_click=_open_add_dialog,
                                            height=48,
                                            expand=True,
                                        ),
                                        col={"xs": 12, "sm": 6, "md": 3},
                                    ),
                                ],
                                spacing=8,
                                run_spacing=8,
                            ),
                            ft.Text(
                                "CSV format: City, Month, Rainfall (mm). "
                                "Imported rows are added to the current dataset.",
                                size=11,
                                color=COLOR_TEXT_MUTED,
                            ),
                        ],
                        spacing=10,
                    ),
                    glass_container(
                        ft.Column(
                            [
                                ft.Row(
                                    [
                                        table_count,
                                        ft.Container(expand=True),
                                        previous_page_button,
                                        table_page_text,
                                        next_page_button,
                                    ],
                                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                    spacing=8,
                                ),
                                ft.Row([records_table], scroll=ft.ScrollMode.AUTO),
                            ],
                            spacing=8,
                        ),
                        padding=12,
                        radius=20,
                        bg_opacity=0.40,
                        blur_sigma=20,
                    ),
                    footer,
                ],
                spacing=16,
            ),
            padding=ft.Padding.only(left=12, right=12, top=12, bottom=16),
        )
    )

    _refresh_city_options()
    _apply_filters()


if __name__ == "__main__":
    ft.run(main)
