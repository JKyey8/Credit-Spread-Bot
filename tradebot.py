"""
SPY Overnight Credit Spread Alert Bot v13 (Unified Next-Open Consensus Model)

Purpose
-------
Alert-only scanner for defined-risk bull put / bear call credit spreads.
It does not place orders.

v13 adds:
- one unified FINAL P(up)/P(down) next-open consensus used for direction selection
- historical analog, large-day, late-tape, VIX-regime, and bounded technical contributions
- explicit model-agreement score and heuristic probability range
- local outcome calibration can gently adjust the final consensus after enough v13 forecasts resolve
- overnight market tracking freezes the final consensus (not a competing sub-model) and live data updates it after the close
- component probabilities remain visible as diagnostics, but are no longer competing directional answers

v12.1 fixes:
- all close-to-next-open models use the same live SPY snapshot
- yesterday's completed close is explicitly separated from any forming Yahoo daily bar
- the general analog model now evaluates the live day rather than depending on Yahoo's daily-bar timing

v12 adds:
- SPY-only close-to-open model; technical alignment is secondary to overnight evidence
- specialized large-day / gap / session decomposition model
- final-30m and final-60m momentum adjustments
- explicit overnight direction probability used to choose bull-put vs bear-call ideas
- large-day reversal probability and exhaustion warnings
- preserves v11.3 live overnight recalibration, morning P/L, near-misses, and calibration

v11.3 adds:
- ALWAYS-ON overnight market watch for GLD/SPY even when no spread fully qualifies
- freezes the close-time forecast, then shows a separate live conditional next-open estimate
- explicit ORIGINAL FORECAST vs LIVE OVERNIGHT STATE so stale close-time probabilities are not mistaken for live odds
- thesis status PASS / CAUTION / INVALIDATED as extended-hours/futures prices move
- weighted-analog effective sample size and an approximate 90% uncertainty band for P(up) next open
- local forecast log that resolves prior forecasts at the next open and reports empirical calibration when enough observations exist
- clearer wording: technical alignment is a 0-100 score, not a probability

v11.2 adds:
- support/resistance can be a weighted warning instead of an automatic rejection
- strong next-morning candidates may override an S/R miss only when multiple overnight checks agree
- S/R override candidates receive a score penalty, explicit warning, and cannot retain an A grade
- configurable maximum S/R shortfall in ATR units prevents extreme overrides

v11.1 adds:
- favored-direction near-miss candidates when no normal alert qualifies
- exact failed-filter explanations and distance-to-threshold diagnostics
- near misses preserve normal liquidity/delta constraints and are never auto-tracked as positions

v11 adds:
- close-to-open strategy focus: probabilities and monitoring are centered on the next regular-session open
- next-open spread P/L simulation using historical analog returns and Black-Scholes repricing
- flat-IV and stressed-IV morning profit probabilities
- persistent tracking of alerted late-day ideas in a local JSON file
- after-hours / premarket monitoring with ETF extended-hours quotes when available
- corrected futures proxy logic: compares ES-to-ES / GC-to-GC returns instead of futures price to ETF price
- dynamic remaining-breakeven cushion and updated open-survival estimate overnight
- major SPY constituent after-hours sentiment monitor
- timed manual events inside the intended entry-to-exit holding window
- material-change Discord updates and a pre-open morning report

v9 adds:
- spread-specific OVERNIGHT PASS / CAUTION / FAIL decision
- historical next-open survival probability at the actual breakeven
- breakeven distance vs recent 90th-percentile overnight gap
- breakeven distance vs option-implied expected move
- explicit technical-vs-overnight conflict guard
- overnight eligibility is shown in console and Discord

v8 adds:
- empirical next-open / next-close probabilities from historically similar market days
- candidate-specific historical probability of finishing beyond the spread breakeven
- Black-Scholes risk-neutral probability at the actual breakeven (uses d2, not 1-delta)
- a consensus POP estimate that blends current option-implied information with historical analogs
- historical no-breach estimate for the next 1-2 sessions
- expected-move context from current option IV
- "signal strength" wording instead of presenting technical agreement as a literal probability

v7 foundation:
- automatic FOMC decision-date tracking from the Federal Reserve calendar (with a fallback calendar)
- current effective fed-funds rate and target-range context from FRED
- FOMC events are folded into the same event-risk warning system
- next-FOMC information is shown even before it becomes a near-term risk

v6 foundation:
- soft event-risk mode: scheduled events warn/downgrade instead of automatically hiding every idea
- neutral directional lean: neutral regimes can still surface a cautious directional idea
- event-day delta cap and one-alert limit for more conservative event-risk candidates
- strict direction matching (no accidental opposite-side fallback)
- multi-timeframe regime detection (daily, 30m, 5m/VWAP)
- 1d / 3d / 5d move context
- ATR-based overextension guard to avoid chasing late moves
- recent overnight-gap statistics and optional gap-buffer filter
- support/resistance clearance filter
- late reversal guard
- event-risk panel for major SPY constituent earnings (best-effort via yfinance)
- manual macro/event calendar support
- optional Alpha Vantage earnings-calendar support
- regular-hours-only scanning, liquidity/RoR filters, dedupe/cooldown
- richer Discord embeds with market context

Important
---------
Probability estimates are model estimates, not guarantees. Historical analog probabilities are backward-looking and Black-Scholes probabilities are risk-neutral, not forecasts. Earnings dates from
free data sources can be delayed or incomplete. Event awareness is a guardrail, not a substitute for checking a current calendar before placing a trade.
The v11.3 live probability is a heuristic update based on actual overnight movement; it is not a guaranteed or broker-grade calibrated probability.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import io
import json
import math
import os
import re
from html.parser import HTMLParser
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from scipy.stats import norm
from zoneinfo import ZoneInfo


EARNINGS_CACHE: Dict[str, Tuple[float, Optional[dt.date]]] = {}
ALPHA_EARNINGS_CACHE: Dict[str, object] = {"time": 0.0, "dates": {}}
FOMC_CACHE: Dict[str, object] = {"time": 0.0, "dates": []}
FED_RATE_CACHE: Dict[str, object] = {"time": 0.0, "note": ""}
VIX_CACHE: Dict[str, object] = {"time": 0.0, "daily": None}
MEGACAP_CACHE: Dict[str, object] = {"time": 0.0, "note": "", "avg": None, "moves": {}}

# Nearest liquid overnight futures proxy for each cash symbol. The v11 logic compares
# each futures contract with its OWN price at the entry/reference timestamp. It never
# divides ES by SPY or GC by GLD, since those instruments use different price scales.
FUTURES_PROXY: Dict[str, str] = {"GLD": "GC=F", "SPY": "ES=F"}


@dataclasses.dataclass
class Config:
    symbols: str = "SPY"

    # Spread filters
    target_dte_min: int = 1
    target_dte_max: int = 1
    short_delta_min: float = 0.13
    short_delta_max: float = 0.25
    max_spread_width: float = 3.0
    min_credit: float = 0.20
    min_ror: float = 0.08
    min_open_interest: int = 50
    min_volume: int = 1
    max_relative_bid_ask_width: float = 0.30
    max_alerts_per_scan: int = 3
    risk_free_rate: float = 0.045
    iv_floor: float = 0.05

    # Runtime
    scan_interval_seconds: int = 300
    late_day_only: bool = False
    regular_hours_only: bool = True
    market_timezone: str = "America/Los_Angeles"
    discord_webhook_url: Optional[str] = None

    # Market-regime data
    # v10: widened from 2y so the k-NN analog pool and gap-percentile stats aren't
    # built on ~500 rows. GLD trades back to 2004, SPY to 1993, so 10y is safe for both.
    daily_period: str = "10y"
    trend_30m_period: str = "60d"
    trend_30m_interval: str = "30m"
    intraday_period: str = "5d"
    intraday_interval: str = "5m"
    daily_ema_fast: int = 10
    daily_ema_slow: int = 20
    daily_ema_long: int = 50
    trend_ema_fast: int = 8
    trend_ema_slow: int = 21
    intraday_ema_fast: int = 8
    intraday_ema_slow: int = 21
    support_resistance_lookback: int = 20
    atr_period: int = 14
    min_market_confidence: int = 60
    allow_neutral_alerts: bool = False
    allow_neutral_directional_ideas: bool = True
    neutral_min_lean_confidence: float = 0.35
    neutral_show_both_sides: bool = False
    max_neutral_alerts_per_scan: int = 1
    min_neutral_idea_score: float = 35.0

    # Probability / overnight model
    probability_neighbors: int = 40
    probability_min_samples: int = 25
    option_pop_weight: float = 0.60
    historical_pop_weight: float = 0.40
    require_probability_filter: bool = True
    min_consensus_pop: float = 0.62
    probability_disagreement_penalty: float = 8.0
    probability_disagreement_threshold: float = 0.15

    # v10: volatility-regime conditioning for the analog pool and gap stats
    vix_regime_enabled: bool = True
    vix_period: str = "10y"
    vix_regime_lookback_days: int = 504  # ~2y of trading days for the tertile split
    vix_feature_weight: float = 1.25  # relative weight of the regime feature vs price features

    # v10: live overnight-futures signal (GC=F for GLD, ES=F for SPY). Futures trade
    # nearly 24h, so they show overnight movement that the cash ETF cannot report
    # until the next regular-session open.
    use_overnight_futures: bool = True
    futures_lookback_period: str = "5d"
    futures_lookback_interval: str = "5m"
    futures_conflict_move_pct: float = 0.0015  # futures move vs last cash close treated as "already moving"
    futures_block_buffer_fraction: float = 0.50  # if futures move already eats this much of the BE cushion, downgrade

    # Overnight eligibility engine
    overnight_eligibility_enabled: bool = True
    overnight_conflict_threshold: float = 0.58
    overnight_pass_open_survival: float = 0.70
    overnight_caution_open_survival: float = 0.58
    overnight_pass_no_breach: float = 0.72
    overnight_caution_no_breach: float = 0.60
    overnight_pass_consensus_pop: float = 0.68
    overnight_caution_consensus_pop: float = 0.62
    overnight_pass_gap_buffer_mult: float = 1.15
    overnight_caution_gap_buffer_mult: float = 0.90
    overnight_pass_expected_move_buffer_mult: float = 0.80
    overnight_caution_expected_move_buffer_mult: float = 0.55
    block_overnight_fail_alerts: bool = False

    # Reversal / overextension guard
    reversal_guard_bars: int = 6
    reversal_guard_threshold_pct: float = 0.003
    overextension_atr_threshold: float = 1.25
    block_overextended_continuation: bool = True

    # Location filters
    require_strike_beyond_sr: bool = True
    min_sr_buffer_atr: float = 0.05

    # v11.2: S/R is still respected, but for this close-to-open strategy it can be
    # treated as a weighted warning when the actual overnight statistics are exceptional.
    # Modes: strict = old behavior; weighted = conditional override; warn = never hard-block on S/R.
    sr_filter_mode: str = "weighted"
    sr_override_min_morning_profit_prob: float = 0.85
    sr_override_min_open_survival: float = 0.90
    sr_override_min_consensus_pop: float = 0.80
    sr_override_min_be_gap_mult: float = 1.00
    sr_override_max_shortfall_atr: float = 1.50
    sr_penalty_per_atr_shortfall: float = 8.0
    sr_penalty_cap: float = 18.0

    require_overnight_gap_buffer: bool = True
    overnight_gap_percentile: float = 0.90
    min_overnight_gap_buffer_mult: float = 1.00

    # Event guard
    event_guard_enabled: bool = True
    event_guard_mode: str = "warn"  # warn / block
    allow_event_risk_alerts: bool = True
    max_event_alerts_per_scan: int = 1
    event_short_delta_max: float = 0.18
    min_event_candidate_score: float = 30.0
    event_lookahead_days: int = 7
    block_same_day_events: bool = True
    block_next_day_events: bool = True
    block_same_week_events: bool = False
    major_earnings_watchlist: str = "AAPL,MSFT,NVDA,AMZN,META,GOOGL,AVGO,TSLA"
    manual_events: str = ""
    alpha_vantage_api_key: Optional[str] = None

    # Federal Reserve / rates context
    auto_fomc_events: bool = True
    show_fed_rate_context: bool = True
    fomc_display_lookahead_days: int = 45
    fed_data_cache_seconds: int = 21600

    # v11 close-to-open monitoring / morning-exit model
    overnight_monitor_enabled: bool = True
    track_alerted_ideas: bool = True
    track_only_entry_window: bool = True
    track_state_file: str = "overnight_tracked_ideas.json"
    entry_window_start: str = "12:30"
    entry_window_end: str = "12:58"
    intended_exit_time: str = "06:40"
    morning_report_start: str = "06:15"
    morning_report_end: str = "06:29"
    extended_hours_period: str = "5d"
    extended_hours_interval: str = "5m"
    extended_quote_stale_minutes: int = 45
    futures_reference_tolerance_minutes: int = 45
    require_morning_profit_filter: bool = True
    min_morning_profit_prob: float = 0.60
    morning_iv_stress_mult: float = 1.10
    morning_pass_profit_prob: float = 0.70
    morning_caution_profit_prob: float = 0.58
    monitor_probability_change: float = 0.08
    monitor_material_move_pct: float = 0.0025
    monitor_cushion_warn_fraction: float = 0.50
    monitor_cushion_fail_fraction: float = 0.80
    mega_cap_monitor_enabled: bool = True
    mega_cap_cache_seconds: int = 600
    mega_cap_move_alert_pct: float = 0.02

    # v13 unified SPY close-to-next-open consensus
    spy_overnight_model_enabled: bool = True
    large_day_model_enabled: bool = True
    large_day_min_abs_return: float = 0.010
    large_day_neighbors: int = 60
    large_day_min_samples: int = 30

    # These weights form ONE directional probability. They are re-normalized if a
    # component is unavailable. VIX is a regime prior; the analog models are also
    # VIX-conditioned, so its standalone weight stays deliberately modest.
    consensus_general_weight: float = 0.40
    consensus_large_day_weight: float = 0.30
    consensus_late_tape_weight: float = 0.15
    consensus_vix_regime_weight: float = 0.10
    consensus_technical_weight: float = 0.05
    consensus_direction_min_edge: float = 0.05
    consensus_agreement_high_max_std: float = 0.06
    consensus_agreement_moderate_max_std: float = 0.12
    consensus_use_local_calibration: bool = True
    consensus_max_calibration_weight: float = 0.35

    # Final-hour contribution is bounded. Technicals are also bounded to a small
    # directional contribution, so 100/100 alignment never means 100% probability.
    final_30m_weight: float = 0.06
    final_60m_weight: float = 0.04
    final_hour_exhaustion_threshold: float = 0.0035

    # v11.3 always-on underlying market watch / live forecast
    always_track_market_overnight: bool = True
    bootstrap_market_watch_after_close: bool = True
    live_forecast_alerts: bool = True
    live_forecast_caution_prob: float = 0.55
    live_forecast_invalidate_prob: float = 0.45
    live_forecast_material_prob_change: float = 0.10
    live_forecast_adverse_gap_caution_mult: float = 0.50
    live_forecast_adverse_gap_invalidate_mult: float = 1.00
    show_probability_uncertainty: bool = True
    forecast_log_file: str = "overnight_forecast_history.json"
    calibration_min_resolved: int = 15
    calibration_bin_width: float = 0.10

    # v11.1 near-miss diagnostics. These candidates are informational only and
    # never count as normal alerts / tracked overnight ideas. Liquidity and delta
    # rules remain unchanged; only selected strategy thresholds are relaxed.
    show_near_miss_candidates: bool = True
    max_near_miss_candidates: int = 3
    near_miss_max_failed_filters: int = 3
    near_miss_min_credit: float = 0.10
    near_miss_min_ror: float = 0.04
    near_miss_min_consensus_pop: float = 0.50
    near_miss_min_morning_profit_prob: float = 0.50

    # Alert dedupe
    alert_cooldown_seconds: int = 3600
    min_score_change_for_realert: float = 5.0
    min_credit_change_for_realert: float = 0.05

    @staticmethod
    def from_env() -> "Config":
        load_dotenv()
        g = os.getenv
        return Config(
            symbols=g("SYMBOLS", "SPY"),
            target_dte_min=int(g("TARGET_DTE_MIN", "1")),
            target_dte_max=int(g("TARGET_DTE_MAX", "1")),
            short_delta_min=float(g("SHORT_DELTA_MIN", "0.13")),
            short_delta_max=float(g("SHORT_DELTA_MAX", "0.25")),
            max_spread_width=float(g("MAX_SPREAD_WIDTH", "3")),
            min_credit=float(g("MIN_CREDIT", "0.20")),
            min_ror=float(g("MIN_ROR", "0.08")),
            min_open_interest=int(g("MIN_OPEN_INTEREST", "50")),
            min_volume=int(g("MIN_VOLUME", "1")),
            max_relative_bid_ask_width=float(g("MAX_RELATIVE_BID_ASK_WIDTH", "0.30")),
            max_alerts_per_scan=int(g("MAX_ALERTS_PER_SCAN", "3")),
            risk_free_rate=float(g("RISK_FREE_RATE", "0.045")),
            iv_floor=float(g("IV_FLOOR", "0.05")),
            scan_interval_seconds=int(g("SCAN_INTERVAL_SECONDS", "300")),
            late_day_only=g("LATE_DAY_ONLY", "false").lower() == "true",
            regular_hours_only=g("REGULAR_HOURS_ONLY", "true").lower() == "true",
            market_timezone=g("MARKET_TIMEZONE", "America/Los_Angeles"),
            discord_webhook_url=g("DISCORD_WEBHOOK_URL") or None,
            daily_period=g("DAILY_PERIOD", "10y"),
            trend_30m_period=g("TREND_30M_PERIOD", "60d"),
            trend_30m_interval=g("TREND_30M_INTERVAL", "30m"),
            intraday_period=g("INTRADAY_PERIOD", "5d"),
            intraday_interval=g("INTRADAY_INTERVAL", "5m"),
            daily_ema_fast=int(g("DAILY_EMA_FAST", "10")),
            daily_ema_slow=int(g("DAILY_EMA_SLOW", "20")),
            daily_ema_long=int(g("DAILY_EMA_LONG", "50")),
            trend_ema_fast=int(g("TREND_EMA_FAST", "8")),
            trend_ema_slow=int(g("TREND_EMA_SLOW", "21")),
            intraday_ema_fast=int(g("INTRADAY_EMA_FAST", "8")),
            intraday_ema_slow=int(g("INTRADAY_EMA_SLOW", "21")),
            support_resistance_lookback=int(g("SUPPORT_RESISTANCE_LOOKBACK", "20")),
            atr_period=int(g("ATR_PERIOD", "14")),
            min_market_confidence=int(g("MIN_MARKET_CONFIDENCE", "60")),
            allow_neutral_alerts=g("ALLOW_NEUTRAL_ALERTS", "false").lower() == "true",
            allow_neutral_directional_ideas=g("ALLOW_NEUTRAL_DIRECTIONAL_IDEAS", "true").lower() == "true",
            neutral_min_lean_confidence=float(g("NEUTRAL_MIN_LEAN_CONFIDENCE", "0.35")),
            neutral_show_both_sides=g("NEUTRAL_SHOW_BOTH_SIDES", "false").lower() == "true",
            max_neutral_alerts_per_scan=int(g("MAX_NEUTRAL_ALERTS_PER_SCAN", "1")),
            min_neutral_idea_score=float(g("MIN_NEUTRAL_IDEA_SCORE", "35.0")),
            probability_neighbors=int(g("PROBABILITY_NEIGHBORS", "40")),
            probability_min_samples=int(g("PROBABILITY_MIN_SAMPLES", "25")),
            option_pop_weight=float(g("OPTION_POP_WEIGHT", "0.60")),
            historical_pop_weight=float(g("HISTORICAL_POP_WEIGHT", "0.40")),
            require_probability_filter=g("REQUIRE_PROBABILITY_FILTER", "true").lower() == "true",
            min_consensus_pop=float(g("MIN_CONSENSUS_POP", "0.62")),
            probability_disagreement_penalty=float(g("PROBABILITY_DISAGREEMENT_PENALTY", "8.0")),
            probability_disagreement_threshold=float(g("PROBABILITY_DISAGREEMENT_THRESHOLD", "0.15")),
            vix_regime_enabled=g("VIX_REGIME_ENABLED", "true").lower() == "true",
            vix_period=g("VIX_PERIOD", "10y"),
            vix_regime_lookback_days=int(g("VIX_REGIME_LOOKBACK_DAYS", "504")),
            vix_feature_weight=float(g("VIX_FEATURE_WEIGHT", "1.25")),
            use_overnight_futures=g("USE_OVERNIGHT_FUTURES", "true").lower() == "true",
            futures_lookback_period=g("FUTURES_LOOKBACK_PERIOD", "5d"),
            futures_lookback_interval=g("FUTURES_LOOKBACK_INTERVAL", "5m"),
            futures_conflict_move_pct=float(g("FUTURES_CONFLICT_MOVE_PCT", "0.0015")),
            futures_block_buffer_fraction=float(g("FUTURES_BLOCK_BUFFER_FRACTION", "0.50")),
            overnight_eligibility_enabled=g("OVERNIGHT_ELIGIBILITY_ENABLED", "true").lower() == "true",
            overnight_conflict_threshold=float(g("OVERNIGHT_CONFLICT_THRESHOLD", "0.58")),
            overnight_pass_open_survival=float(g("OVERNIGHT_PASS_OPEN_SURVIVAL", "0.70")),
            overnight_caution_open_survival=float(g("OVERNIGHT_CAUTION_OPEN_SURVIVAL", "0.58")),
            overnight_pass_no_breach=float(g("OVERNIGHT_PASS_NO_BREACH", "0.72")),
            overnight_caution_no_breach=float(g("OVERNIGHT_CAUTION_NO_BREACH", "0.60")),
            overnight_pass_consensus_pop=float(g("OVERNIGHT_PASS_CONSENSUS_POP", "0.68")),
            overnight_caution_consensus_pop=float(g("OVERNIGHT_CAUTION_CONSENSUS_POP", "0.62")),
            overnight_pass_gap_buffer_mult=float(g("OVERNIGHT_PASS_GAP_BUFFER_MULT", "1.15")),
            overnight_caution_gap_buffer_mult=float(g("OVERNIGHT_CAUTION_GAP_BUFFER_MULT", "0.90")),
            overnight_pass_expected_move_buffer_mult=float(g("OVERNIGHT_PASS_EXPECTED_MOVE_BUFFER_MULT", "0.80")),
            overnight_caution_expected_move_buffer_mult=float(g("OVERNIGHT_CAUTION_EXPECTED_MOVE_BUFFER_MULT", "0.55")),
            block_overnight_fail_alerts=g("BLOCK_OVERNIGHT_FAIL_ALERTS", "false").lower() == "true",
            reversal_guard_bars=int(g("REVERSAL_GUARD_BARS", "6")),
            reversal_guard_threshold_pct=float(g("REVERSAL_GUARD_THRESHOLD_PCT", "0.003")),
            overextension_atr_threshold=float(g("OVEREXTENSION_ATR_THRESHOLD", "1.25")),
            block_overextended_continuation=g("BLOCK_OVEREXTENDED_CONTINUATION", "true").lower() == "true",
            require_strike_beyond_sr=g("REQUIRE_STRIKE_BEYOND_SR", "true").lower() == "true",
            min_sr_buffer_atr=float(g("MIN_SR_BUFFER_ATR", "0.05")),
            sr_filter_mode=g("SR_FILTER_MODE", "weighted").strip().lower(),
            sr_override_min_morning_profit_prob=float(g("SR_OVERRIDE_MIN_MORNING_PROFIT_PROB", "0.85")),
            sr_override_min_open_survival=float(g("SR_OVERRIDE_MIN_OPEN_SURVIVAL", "0.90")),
            sr_override_min_consensus_pop=float(g("SR_OVERRIDE_MIN_CONSENSUS_POP", "0.80")),
            sr_override_min_be_gap_mult=float(g("SR_OVERRIDE_MIN_BE_GAP_MULT", "1.00")),
            sr_override_max_shortfall_atr=float(g("SR_OVERRIDE_MAX_SHORTFALL_ATR", "1.50")),
            sr_penalty_per_atr_shortfall=float(g("SR_PENALTY_PER_ATR_SHORTFALL", "8.0")),
            sr_penalty_cap=float(g("SR_PENALTY_CAP", "18.0")),
            require_overnight_gap_buffer=g("REQUIRE_OVERNIGHT_GAP_BUFFER", "true").lower() == "true",
            overnight_gap_percentile=float(g("OVERNIGHT_GAP_PERCENTILE", "0.90")),
            min_overnight_gap_buffer_mult=float(g("MIN_OVERNIGHT_GAP_BUFFER_MULT", "1.00")),
            event_guard_enabled=g("EVENT_GUARD_ENABLED", "true").lower() == "true",
            event_guard_mode=g("EVENT_GUARD_MODE", "warn").strip().lower(),
            allow_event_risk_alerts=g("ALLOW_EVENT_RISK_ALERTS", "true").lower() == "true",
            max_event_alerts_per_scan=int(g("MAX_EVENT_ALERTS_PER_SCAN", "1")),
            event_short_delta_max=float(g("EVENT_SHORT_DELTA_MAX", "0.18")),
            min_event_candidate_score=float(g("MIN_EVENT_CANDIDATE_SCORE", "30.0")),
            event_lookahead_days=int(g("EVENT_LOOKAHEAD_DAYS", "7")),
            block_same_day_events=g("BLOCK_SAME_DAY_EVENTS", "true").lower() == "true",
            block_next_day_events=g("BLOCK_NEXT_DAY_EVENTS", "true").lower() == "true",
            block_same_week_events=g("BLOCK_SAME_WEEK_EVENTS", "false").lower() == "true",
            major_earnings_watchlist=g("MAJOR_EARNINGS_WATCHLIST", "AAPL,MSFT,NVDA,AMZN,META,GOOGL,AVGO,TSLA"),
            manual_events=g("MANUAL_EVENTS", ""),
            alpha_vantage_api_key=g("ALPHA_VANTAGE_API_KEY") or None,
            auto_fomc_events=g("AUTO_FOMC_EVENTS", "true").lower() == "true",
            show_fed_rate_context=g("SHOW_FED_RATE_CONTEXT", "true").lower() == "true",
            fomc_display_lookahead_days=int(g("FOMC_DISPLAY_LOOKAHEAD_DAYS", "45")),
            fed_data_cache_seconds=int(g("FED_DATA_CACHE_SECONDS", "21600")),
            overnight_monitor_enabled=g("OVERNIGHT_MONITOR_ENABLED", "true").lower() == "true",
            track_alerted_ideas=g("TRACK_ALERTED_IDEAS", "true").lower() == "true",
            track_only_entry_window=g("TRACK_ONLY_ENTRY_WINDOW", "true").lower() == "true",
            track_state_file=g("TRACK_STATE_FILE", "overnight_tracked_ideas.json"),
            entry_window_start=g("ENTRY_WINDOW_START", "12:30"),
            entry_window_end=g("ENTRY_WINDOW_END", "12:58"),
            intended_exit_time=g("INTENDED_EXIT_TIME", "06:40"),
            morning_report_start=g("MORNING_REPORT_START", "06:15"),
            morning_report_end=g("MORNING_REPORT_END", "06:29"),
            extended_hours_period=g("EXTENDED_HOURS_PERIOD", "5d"),
            extended_hours_interval=g("EXTENDED_HOURS_INTERVAL", "5m"),
            extended_quote_stale_minutes=int(g("EXTENDED_QUOTE_STALE_MINUTES", "45")),
            futures_reference_tolerance_minutes=int(g("FUTURES_REFERENCE_TOLERANCE_MINUTES", "45")),
            require_morning_profit_filter=g("REQUIRE_MORNING_PROFIT_FILTER", "true").lower() == "true",
            min_morning_profit_prob=float(g("MIN_MORNING_PROFIT_PROB", "0.60")),
            morning_iv_stress_mult=float(g("MORNING_IV_STRESS_MULT", "1.10")),
            morning_pass_profit_prob=float(g("MORNING_PASS_PROFIT_PROB", "0.70")),
            morning_caution_profit_prob=float(g("MORNING_CAUTION_PROFIT_PROB", "0.58")),
            monitor_probability_change=float(g("MONITOR_PROBABILITY_CHANGE", "0.08")),
            monitor_material_move_pct=float(g("MONITOR_MATERIAL_MOVE_PCT", "0.0025")),
            monitor_cushion_warn_fraction=float(g("MONITOR_CUSHION_WARN_FRACTION", "0.50")),
            monitor_cushion_fail_fraction=float(g("MONITOR_CUSHION_FAIL_FRACTION", "0.80")),
            mega_cap_monitor_enabled=g("MEGA_CAP_MONITOR_ENABLED", "true").lower() == "true",
            mega_cap_cache_seconds=int(g("MEGA_CAP_CACHE_SECONDS", "600")),
            mega_cap_move_alert_pct=float(g("MEGA_CAP_MOVE_ALERT_PCT", "0.02")),
            spy_overnight_model_enabled=g("SPY_OVERNIGHT_MODEL_ENABLED", "true").lower() == "true",
            large_day_model_enabled=g("LARGE_DAY_MODEL_ENABLED", "true").lower() == "true",
            large_day_min_abs_return=float(g("LARGE_DAY_MIN_ABS_RETURN", "0.010")),
            large_day_neighbors=int(g("LARGE_DAY_NEIGHBORS", "60")),
            large_day_min_samples=int(g("LARGE_DAY_MIN_SAMPLES", "30")),
            consensus_general_weight=float(g("CONSENSUS_GENERAL_WEIGHT", "0.40")),
            consensus_large_day_weight=float(g("CONSENSUS_LARGE_DAY_WEIGHT", "0.30")),
            consensus_late_tape_weight=float(g("CONSENSUS_LATE_TAPE_WEIGHT", "0.15")),
            consensus_vix_regime_weight=float(g("CONSENSUS_VIX_REGIME_WEIGHT", "0.10")),
            consensus_technical_weight=float(g("CONSENSUS_TECHNICAL_WEIGHT", "0.05")),
            consensus_direction_min_edge=float(g("CONSENSUS_DIRECTION_MIN_EDGE", "0.05")),
            consensus_agreement_high_max_std=float(g("CONSENSUS_AGREEMENT_HIGH_MAX_STD", "0.06")),
            consensus_agreement_moderate_max_std=float(g("CONSENSUS_AGREEMENT_MODERATE_MAX_STD", "0.12")),
            consensus_use_local_calibration=g("CONSENSUS_USE_LOCAL_CALIBRATION", "true").lower() == "true",
            consensus_max_calibration_weight=float(g("CONSENSUS_MAX_CALIBRATION_WEIGHT", "0.35")),
            final_30m_weight=float(g("FINAL_30M_WEIGHT", "0.06")),
            final_60m_weight=float(g("FINAL_60M_WEIGHT", "0.04")),
            final_hour_exhaustion_threshold=float(g("FINAL_HOUR_EXHAUSTION_THRESHOLD", "0.0035")),
            always_track_market_overnight=g("ALWAYS_TRACK_MARKET_OVERNIGHT", "true").lower() == "true",
            bootstrap_market_watch_after_close=g("BOOTSTRAP_MARKET_WATCH_AFTER_CLOSE", "true").lower() == "true",
            live_forecast_alerts=g("LIVE_FORECAST_ALERTS", "true").lower() == "true",
            live_forecast_caution_prob=float(g("LIVE_FORECAST_CAUTION_PROB", "0.55")),
            live_forecast_invalidate_prob=float(g("LIVE_FORECAST_INVALIDATE_PROB", "0.45")),
            live_forecast_material_prob_change=float(g("LIVE_FORECAST_MATERIAL_PROB_CHANGE", "0.10")),
            live_forecast_adverse_gap_caution_mult=float(g("LIVE_FORECAST_ADVERSE_GAP_CAUTION_MULT", "0.50")),
            live_forecast_adverse_gap_invalidate_mult=float(g("LIVE_FORECAST_ADVERSE_GAP_INVALIDATE_MULT", "1.00")),
            show_probability_uncertainty=g("SHOW_PROBABILITY_UNCERTAINTY", "true").lower() == "true",
            forecast_log_file=g("FORECAST_LOG_FILE", "overnight_forecast_history.json"),
            calibration_min_resolved=int(g("CALIBRATION_MIN_RESOLVED", "15")),
            calibration_bin_width=float(g("CALIBRATION_BIN_WIDTH", "0.10")),
            show_near_miss_candidates=g("SHOW_NEAR_MISS_CANDIDATES", "true").lower() == "true",
            max_near_miss_candidates=int(g("MAX_NEAR_MISS_CANDIDATES", "3")),
            near_miss_max_failed_filters=int(g("NEAR_MISS_MAX_FAILED_FILTERS", "3")),
            near_miss_min_credit=float(g("NEAR_MISS_MIN_CREDIT", "0.10")),
            near_miss_min_ror=float(g("NEAR_MISS_MIN_ROR", "0.04")),
            near_miss_min_consensus_pop=float(g("NEAR_MISS_MIN_CONSENSUS_POP", "0.50")),
            near_miss_min_morning_profit_prob=float(g("NEAR_MISS_MIN_MORNING_PROFIT_PROB", "0.50")),
            alert_cooldown_seconds=int(g("ALERT_COOLDOWN_SECONDS", "3600")),
            min_score_change_for_realert=float(g("MIN_SCORE_CHANGE_FOR_REALERT", "5.0")),
            min_credit_change_for_realert=float(g("MIN_CREDIT_CHANGE_FOR_REALERT", "0.05")),
        )


@dataclasses.dataclass
class EventRisk:
    note: str = "No major scheduled event detected"
    severity: str = "LOW"  # LOW / MEDIUM / HIGH
    block: bool = False
    next_fomc: str = ""
    fed_rate: str = ""


@dataclasses.dataclass
class HistoricalProbability:
    valid: bool = False
    samples: int = 0
    p_up_next_open: float = 0.50
    p_up_1d_close: float = 0.50
    p_up_2d_close: float = 0.50
    avg_overnight_return: float = 0.0
    avg_1d_return: float = 0.0
    avg_2d_return: float = 0.0
    weights: List[float] = dataclasses.field(default_factory=list, repr=False)
    next_open: List[float] = dataclasses.field(default_factory=list, repr=False)
    next_close_1: List[float] = dataclasses.field(default_factory=list, repr=False)
    next_close_2: List[float] = dataclasses.field(default_factory=list, repr=False)
    next_low_1: List[float] = dataclasses.field(default_factory=list, repr=False)
    next_high_1: List[float] = dataclasses.field(default_factory=list, repr=False)
    min_low_2: List[float] = dataclasses.field(default_factory=list, repr=False)
    max_high_2: List[float] = dataclasses.field(default_factory=list, repr=False)


@dataclasses.dataclass
class LargeDayOvernight:
    valid: bool = False
    samples: int = 0
    total_return: float = 0.0
    gap_return: float = 0.0
    session_return: float = 0.0
    move_atr: float = 0.0
    range_position: float = 0.5
    final_30m_return: float = 0.0
    final_60m_return: float = 0.0
    p_up_next_open: float = 0.50
    p_reversal_next_open: float = 0.50
    avg_next_open_return: float = 0.0
    note: str = ""


@dataclasses.dataclass
class ConsensusForecast:
    raw_p_up: float = 0.50
    p_up: float = 0.50
    p_down: float = 0.50
    direction: str = "neutral"
    strength: int = 0
    agreement: str = "LOW"
    weighted_std: float = 0.0
    range_low: float = 0.35
    range_high: float = 0.65
    components: Dict[str, float] = dataclasses.field(default_factory=dict)
    component_weights: Dict[str, float] = dataclasses.field(default_factory=dict)
    effective_n: float = 0.0
    calibrated_p_up: Optional[float] = None
    calibration_n: int = 0
    calibration_weight: float = 0.0
    note: str = ""


@dataclasses.dataclass
class MarketState:
    last_price: float
    bias: str
    confidence: int
    directional_lean: str
    lean_confidence: int
    reason: str
    support: float
    resistance: float
    atr14: float
    historical_iv: float
    daily_signal: int
    m30_signal: int
    m5_signal: int
    ret_1d: float
    ret_3d: float
    ret_5d: float
    day_move_atr: float
    avg_abs_gap: float
    pctl_abs_gap: float
    vwap: float
    event_risk: EventRisk
    probability: HistoricalProbability
    vix_level: Optional[float] = None
    vix_percentile: Optional[float] = None
    vix_bucket: str = "unknown"
    futures_symbol: Optional[str] = None
    futures_overnight_move_pct: Optional[float] = None
    futures_note: str = ""
    large_day: LargeDayOvernight = dataclasses.field(default_factory=LargeDayOvernight)
    overnight_p_up: float = 0.50
    overnight_direction: str = "neutral"
    overnight_direction_strength: int = 0
    overnight_direction_note: str = ""
    overnight_consensus: ConsensusForecast = dataclasses.field(default_factory=ConsensusForecast)
    block_new_alerts: bool = False


@dataclasses.dataclass
class SpreadCandidate:
    spread_type: str
    expiration: str
    dte: int
    short_strike: float
    long_strike: float
    width: float
    short_delta: float
    long_delta: float
    credit: float
    max_loss: float
    option_pop: float
    historical_pop: float
    consensus_pop: float
    no_breach_prob: float
    expected_move_pct: float
    probability_disagreement: float
    breakeven: float
    overnight_risk: str
    overnight_eligibility: str
    overnight_reasons: str
    open_survival_prob: float
    breakeven_distance_pct: float
    be_gap_buffer_mult: float
    be_expected_move_buffer_mult: float
    ror: float
    ev_heuristic: float
    score: float
    grade: str
    rationale: str
    short_iv: float = 0.0
    long_iv: float = 0.0
    morning_profit_prob: float = 0.50
    morning_profit_prob_stress: float = 0.50
    morning_median_pl: float = 0.0
    morning_p25_pl: float = 0.0
    morning_p10_pl: float = 0.0
    morning_loss25_prob: float = 0.0
    morning_loss50_prob: float = 0.0
    sr_clearance: float = 0.0
    sr_required: float = 0.0
    sr_shortfall_atr: float = 0.0
    sr_override_used: bool = False
    sr_penalty: float = 0.0

    def title(self) -> str:
        return self.spread_type.replace("_", " ").title()

    def key(self, symbol: str) -> Tuple[str, str, str, float, float]:
        return (symbol, self.spread_type, self.expiration, self.short_strike, self.long_strike)


@dataclasses.dataclass
class NearMissCandidate:
    candidate: SpreadCandidate
    failed_filters: List[str]
    miss_distance: float


# ---------------------------- Time / data helpers ----------------------------

def now_local(tz_name: str) -> dt.datetime:
    return dt.datetime.now(ZoneInfo(tz_name))


def within_market_hours(tz_name: str) -> bool:
    now = now_local(tz_name)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=6, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=13, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


def within_late_day_window(tz_name: str) -> bool:
    now = now_local(tz_name)
    if now.weekday() >= 5:
        return False
    late_start = now.replace(hour=11, minute=45, second=0, microsecond=0)
    market_close = now.replace(hour=13, minute=0, second=0, microsecond=0)
    return late_start <= now <= market_close


def get_histories(symbol: str, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ticker = yf.Ticker(symbol)
    daily = ticker.history(period=config.daily_period, interval="1d", auto_adjust=False)
    m30 = ticker.history(period=config.trend_30m_period, interval=config.trend_30m_interval, auto_adjust=False)
    m5 = ticker.history(period=config.intraday_period, interval=config.intraday_interval, auto_adjust=False)
    if daily.empty:
        raise RuntimeError(f"No daily history returned for {symbol}")
    if m30.empty:
        raise RuntimeError(f"No 30-minute history returned for {symbol}")
    if m5.empty:
        raise RuntimeError(f"No intraday history returned for {symbol}")
    return daily, m30, m5


def ema_signal(close: pd.Series, fast: int, slow: int, long_period: Optional[int] = None) -> Tuple[int, float, float, Optional[float]]:
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    last = float(close.iloc[-1])
    fast_v = float(ef.iloc[-1])
    slow_v = float(es.iloc[-1])
    long_v: Optional[float] = None
    if long_period:
        el = close.ewm(span=long_period, adjust=False).mean()
        long_v = float(el.iloc[-1])
        if last > fast_v > slow_v > long_v:
            return 1, fast_v, slow_v, long_v
        if last < fast_v < slow_v < long_v:
            return -1, fast_v, slow_v, long_v
        return 0, fast_v, slow_v, long_v
    if last > fast_v > slow_v:
        return 1, fast_v, slow_v, None
    if last < fast_v < slow_v:
        return -1, fast_v, slow_v, None
    return 0, fast_v, slow_v, None


def calc_atr(daily: pd.DataFrame, period: int) -> float:
    d = daily.copy().dropna(subset=["High", "Low", "Close"])
    prev_close = d["Close"].shift(1)
    tr = pd.concat(
        [
            d["High"] - d["Low"],
            (d["High"] - prev_close).abs(),
            (d["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean().dropna()
    if atr.empty:
        return float(tr.tail(period).mean()) if not tr.empty else 0.0
    return float(atr.iloc[-1])


def recent_returns(daily: pd.DataFrame) -> Tuple[float, float, float]:
    c = daily["Close"].dropna()
    def r(n: int) -> float:
        if len(c) <= n:
            return 0.0
        return float(c.iloc[-1] / c.iloc[-(n + 1)] - 1.0)
    return r(1), r(3), r(5)


def overnight_gap_stats(daily: pd.DataFrame, percentile: float) -> Tuple[float, float]:
    d = daily[["Open", "Close"]].dropna().copy()
    prev_close = d["Close"].shift(1)
    gaps = ((d["Open"] / prev_close) - 1.0).dropna().abs()
    if gaps.empty:
        return 0.0, 0.0
    p = min(max(percentile, 0.50), 0.99)
    return float(gaps.mean()), float(gaps.quantile(p))


def get_vix_context(config: Config) -> pd.DataFrame:
    """Cached daily VIX close history, used to bucket days by volatility regime."""
    now_ts = time.time()
    cached = VIX_CACHE.get("daily")
    if cached is not None and now_ts - float(VIX_CACHE.get("time", 0.0)) < 3600:
        return cached  # type: ignore[return-value]
    try:
        vix = yf.Ticker("^VIX").history(period=config.vix_period, interval="1d", auto_adjust=False)
        vix = vix[["Close"]].dropna().rename(columns={"Close": "vix_close"})
    except Exception as exc:
        print(f"[warn] failed to fetch VIX history: {exc}", file=sys.stderr)
        vix = pd.DataFrame(columns=["vix_close"])
    VIX_CACHE["daily"] = vix
    VIX_CACHE["time"] = now_ts
    return vix


def vix_regime_series(daily_index: pd.DatetimeIndex, vix: pd.DataFrame, lookback_days: int) -> pd.Series:
    """Causal (no-lookahead) rolling percentile rank of VIX close, aligned to daily_index.

    Each day's percentile is computed only from prior days in its own trailing
    window, so this is safe to use inside a walk-forward backtest as well as live.
    """
    if vix.empty:
        return pd.Series(np.nan, index=daily_index)
    aligned = vix["vix_close"].reindex(daily_index, method="ffill")
    pctile = aligned.rolling(lookback_days, min_periods=max(30, lookback_days // 10)).apply(
        lambda w: float((w <= w[-1]).mean()), raw=True
    )
    return pctile


def vix_bucket_label(pctile: Optional[float]) -> str:
    if pctile is None or (isinstance(pctile, float) and np.isnan(pctile)):
        return "unknown"
    if pctile < 1 / 3:
        return "low_vol"
    if pctile < 2 / 3:
        return "mid_vol"
    return "high_vol"


def overnight_gap_stats_regime(
    daily: pd.DataFrame, vix: pd.DataFrame, config: Config
) -> Tuple[float, float, str, Optional[float]]:
    """Regime-conditioned version of overnight_gap_stats: computes the p90 (etc.)
    overnight-gap threshold only from historical days in the *same* VIX-percentile
    tertile as today, instead of blending calm and volatile regimes together.
    Falls back to the unconditioned stat if VIX data or bucket sample is too thin.
    """
    d = daily[["Open", "Close"]].dropna().copy()
    if not config.vix_regime_enabled or vix.empty or len(d) < 60:
        avg_abs, pctl_abs = overnight_gap_stats(daily, config.overnight_gap_percentile)
        return avg_abs, pctl_abs, "unknown", None

    pctile = vix_regime_series(d.index, vix, config.vix_regime_lookback_days)
    bucket = pctile.apply(vix_bucket_label)
    current_bucket = bucket.iloc[-1]
    current_pctile = float(pctile.iloc[-1]) if not np.isnan(pctile.iloc[-1]) else None

    prev_close = d["Close"].shift(1)
    gaps = ((d["Open"] / prev_close) - 1.0).abs()
    same_regime = gaps[(bucket == current_bucket) & gaps.notna()]

    min_needed = max(40, config.probability_min_samples)
    if current_bucket == "unknown" or len(same_regime) < min_needed:
        avg_abs, pctl_abs = overnight_gap_stats(daily, config.overnight_gap_percentile)
        return avg_abs, pctl_abs, current_bucket, current_pctile

    p = min(max(config.overnight_gap_percentile, 0.50), 0.99)
    return float(same_regime.mean()), float(same_regime.quantile(p)), current_bucket, current_pctile


def _parse_hhmm(value: str) -> dt.time:
    """Parse HH:MM in the configured market timezone."""
    try:
        hh, mm = value.strip().split(":", 1)
        return dt.time(hour=int(hh), minute=int(mm))
    except Exception:
        return dt.time(hour=0, minute=0)


def _in_clock_window(now: dt.datetime, start_hhmm: str, end_hhmm: str) -> bool:
    start_t = _parse_hhmm(start_hhmm)
    end_t = _parse_hhmm(end_hhmm)
    t = now.timetz().replace(tzinfo=None)
    if start_t <= end_t:
        return start_t <= t <= end_t
    return t >= start_t or t <= end_t


def _previous_weekday(day: dt.date) -> dt.date:
    d = day - dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def _next_weekday(day: dt.date) -> dt.date:
    d = day + dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def next_regular_open(after: dt.datetime, tz_name: str) -> dt.datetime:
    """Best-effort next 06:30 PT regular open; weekends are skipped.

    This intentionally avoids a market-calendar dependency. Exchange holidays are not
    inferred here, so the event/monitor output should still be treated as an estimate.
    """
    tz = ZoneInfo(tz_name)
    local = after.astimezone(tz)
    today_open = dt.datetime.combine(local.date(), dt.time(6, 30), tzinfo=tz)
    if local.weekday() < 5 and local < today_open:
        return today_open
    d = _next_weekday(local.date())
    return dt.datetime.combine(d, dt.time(6, 30), tzinfo=tz)


def most_recent_regular_close(now: dt.datetime, tz_name: str) -> dt.datetime:
    tz = ZoneInfo(tz_name)
    local = now.astimezone(tz)
    today_close = dt.datetime.combine(local.date(), dt.time(13, 0), tzinfo=tz)
    if local.weekday() < 5 and local >= today_close:
        return today_close
    d = _previous_weekday(local.date())
    return dt.datetime.combine(d, dt.time(13, 0), tzinfo=tz)


def _normalize_history_index(frame: pd.DataFrame, tz_name: str) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out
    idx = pd.DatetimeIndex(out.index)
    tz = ZoneInfo(tz_name)
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    else:
        idx = idx.tz_convert(tz)
    out.index = idx
    return out


def _nearest_price_at_or_before(
    frame: pd.DataFrame,
    when: dt.datetime,
    tolerance_minutes: int,
) -> Optional[float]:
    if frame.empty:
        return None
    before = frame[frame.index <= when]
    if before.empty:
        return None
    ts = before.index[-1]
    if when - ts > dt.timedelta(minutes=max(1, tolerance_minutes)):
        return None
    value = before.iloc[-1].get("Close", np.nan)
    return None if pd.isna(value) else float(value)


def get_futures_reference_snapshot(
    symbol: str,
    reference_dt: dt.datetime,
    config: Config,
) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float], str]:
    """Return proxy, reference price, latest price, self-return, note.

    v10 incorrectly compared the absolute ES price with SPY (and GC with GLD).
    v11 compares each futures contract with its OWN price near the reference time,
    making the percentage move scale-independent and meaningful.
    """
    fut_symbol = FUTURES_PROXY.get(symbol.upper())
    if not config.use_overnight_futures or fut_symbol is None:
        return fut_symbol, None, None, None, "futures proxy disabled / unavailable"
    try:
        fut = yf.Ticker(fut_symbol).history(
            period=config.futures_lookback_period,
            interval=config.futures_lookback_interval,
            auto_adjust=False,
            prepost=True,
        )
        fut = _normalize_history_index(fut.dropna(subset=["Close"]), config.market_timezone)
        if fut.empty:
            return fut_symbol, None, None, None, f"{fut_symbol}: no recent futures data"
        ref = _nearest_price_at_or_before(fut, reference_dt, config.futures_reference_tolerance_minutes)
        latest = float(fut["Close"].iloc[-1])
        latest_ts = fut.index[-1]
        if ref is None or ref <= 0:
            return fut_symbol, None, latest, None, f"{fut_symbol}: no bar close enough to reference time"
        move = latest / ref - 1.0
        note = (
            f"{fut_symbol} {move*100:+.2f}% from its own {reference_dt.strftime('%m-%d %H:%M')} reference "
            f"to {latest_ts.strftime('%m-%d %H:%M')}"
        )
        return fut_symbol, ref, latest, move, note
    except Exception as exc:
        return fut_symbol, None, None, None, f"{fut_symbol}: futures fetch failed ({exc})"


def get_futures_overnight_move(
    symbol: str,
    last_cash_close: float,
    config: Config,
) -> Tuple[Optional[str], Optional[float], str]:
    """Compatibility wrapper used by the market snapshot.

    During regular hours there is no completed overnight session yet, so the snapshot
    uses the most recent regular close as a context reference. Tracked ideas capture
    their own exact futures baseline at alert/entry time and use that baseline overnight.
    The last_cash_close argument remains only for compatibility with the v10 call site.
    """
    now = now_local(config.market_timezone)
    ref_dt = most_recent_regular_close(now, config.market_timezone)
    fut_symbol, _, _, move, note = get_futures_reference_snapshot(symbol, ref_dt, config)
    return fut_symbol, move, note


def get_extended_hours_snapshot(
    symbol: str,
    entry_spot: float,
    config: Config,
) -> Tuple[Optional[float], Optional[float], Optional[dt.datetime], str]:
    """Best-effort latest ETF quote including pre/post-market data.

    Returns latest price, move from entry spot, quote timestamp, and a note. If the
    extended-hours quote is stale, the caller can fall back to the futures proxy.
    """
    try:
        frame = yf.Ticker(symbol).history(
            period=config.extended_hours_period,
            interval=config.extended_hours_interval,
            auto_adjust=False,
            prepost=True,
        )
        frame = _normalize_history_index(frame.dropna(subset=["Close"]), config.market_timezone)
        if frame.empty:
            return None, None, None, "no extended-hours quote"
        ts = frame.index[-1].to_pydatetime()
        price = float(frame["Close"].iloc[-1])
        move = price / entry_spot - 1.0 if entry_spot > 0 else None
        age = now_local(config.market_timezone) - ts
        stale = age > dt.timedelta(minutes=max(1, config.extended_quote_stale_minutes))
        note = f"{symbol} extended quote {price:.2f} at {ts.strftime('%m-%d %H:%M')}"
        if stale:
            note += f" (stale {age.total_seconds()/60:.0f}m)"
        return price, move, ts, note
    except Exception as exc:
        return None, None, None, f"extended-hours fetch failed ({exc})"


def get_megacap_afterhours_sentiment(config: Config) -> Tuple[Optional[float], str, Dict[str, float]]:
    """Equal-weight after-hours sentiment for the configured large SPY constituents.

    This is intentionally labeled sentiment, not an SPY fair-value estimate. The list
    is user-configurable and individual index weights change over time.
    """
    if not config.mega_cap_monitor_enabled:
        return None, "mega-cap monitor disabled", {}
    now_ts = time.time()
    if now_ts - float(MEGACAP_CACHE.get("time", 0.0)) < config.mega_cap_cache_seconds:
        return (
            MEGACAP_CACHE.get("avg") if isinstance(MEGACAP_CACHE.get("avg"), (float, int)) else None,
            str(MEGACAP_CACHE.get("note", "")),
            dict(MEGACAP_CACHE.get("moves", {})) if isinstance(MEGACAP_CACHE.get("moves"), dict) else {},
        )

    tickers = [x.strip().upper() for x in config.major_earnings_watchlist.split(",") if x.strip()]
    moves: Dict[str, float] = {}
    now = now_local(config.market_timezone)
    ref_dt = most_recent_regular_close(now, config.market_timezone)
    for ticker_name in tickers:
        try:
            frame = yf.Ticker(ticker_name).history(period="5d", interval="5m", auto_adjust=False, prepost=True)
            frame = _normalize_history_index(frame.dropna(subset=["Close"]), config.market_timezone)
            if frame.empty:
                continue
            ref = _nearest_price_at_or_before(frame, ref_dt, 30)
            latest = float(frame["Close"].iloc[-1])
            if ref and ref > 0:
                moves[ticker_name] = latest / ref - 1.0
        except Exception:
            continue

    avg = float(np.mean(list(moves.values()))) if moves else None
    if moves:
        top = sorted(moves.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
        detail = ", ".join(f"{k} {v*100:+.2f}%" for k, v in top)
        note = f"mega-cap equal-weight avg {avg*100:+.2f}% | biggest: {detail}"
    else:
        note = "no usable mega-cap extended-hours quotes"
    MEGACAP_CACHE.update({"time": now_ts, "avg": avg, "note": note, "moves": moves})
    return avg, note, moves


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0 or len(weights) == 0 or float(weights.sum()) <= 0:
        return 0.0
    return float(np.average(values, weights=weights))


def _same_calendar_date(a, b) -> bool:
    try:
        return pd.Timestamp(a).date() == pd.Timestamp(b).date()
    except Exception:
        return False


def _completed_daily_before_live_day(daily: pd.DataFrame, day: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return only completed daily bars that precede the live intraday session.

    Yahoo may or may not include today's still-forming daily candle.  The overnight
    model must never guess based on that behavior, so if the final daily row shares
    the live intraday date we explicitly remove it.
    """
    d = daily.copy().dropna(subset=["Close"])
    if d.empty or day is None or day.empty:
        return d
    live_date = day.index[-1]
    while len(d) and _same_calendar_date(d.index[-1], live_date):
        d = d.iloc[:-1]
    return d


def _daily_with_live_snapshot(daily: pd.DataFrame, day: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Append one synthetic live OHLC row to completed daily history.

    This gives every close->next-open feature the exact same timestamp/state: the
    previous completed close plus today's live open/high/low/current price.
    """
    completed = _completed_daily_before_live_day(daily, day)
    if day is None or day.empty:
        return completed
    live = pd.DataFrame(
        {
            "Open": [float(day["Open"].iloc[0])],
            "High": [float(day["High"].max())],
            "Low": [float(day["Low"].min())],
            "Close": [float(day["Close"].iloc[-1])],
        },
        index=pd.DatetimeIndex([day.index[-1]]),
    )
    return pd.concat([completed[["Open", "High", "Low", "Close"]], live])


def historical_probability_context(
    daily: pd.DataFrame, config: Config, vix: Optional[pd.DataFrame] = None,
    day: Optional[pd.DataFrame] = None,
) -> HistoricalProbability:
    """Nearest-neighbor empirical probabilities from similar prior daily regimes.

    v10: optionally folds in a causal (no-lookahead) VIX-percentile feature so the
    neighbor search prefers analog days from a similar volatility regime, not just
    a similar price/trend shape. Passing vix=None (or an empty frame) reproduces
    the v9 behavior exactly, which also keeps this function safe to reuse verbatim
    inside a walk-forward backtest.
    """
    d = _daily_with_live_snapshot(daily, day)
    if len(d) < max(config.probability_min_samples + 20, 60):
        return HistoricalProbability()

    close = d["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            d["High"] - d["Low"],
            (d["High"] - prev_close).abs(),
            (d["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(config.atr_period, min_periods=max(5, config.atr_period // 2)).mean()
    ema10 = close.ewm(span=config.daily_ema_fast, adjust=False).mean()
    ema20 = close.ewm(span=config.daily_ema_slow, adjust=False).mean()
    ema50 = close.ewm(span=config.daily_ema_long, adjust=False).mean()
    day_range = (d["High"] - d["Low"]).replace(0, np.nan)

    f = pd.DataFrame(index=d.index)
    f["ret1"] = close.pct_change(1)
    f["ret3"] = close.pct_change(3)
    f["ret5"] = close.pct_change(5)
    f["day_atr"] = (close - d["Open"]) / atr.replace(0, np.nan)
    f["range_pos"] = (close - d["Low"]) / day_range
    f["ema_fast_slow"] = (ema10 - ema20) / atr.replace(0, np.nan)
    f["price_long"] = (close - ema50) / atr.replace(0, np.nan)

    feature_cols = ["ret1", "ret3", "ret5", "day_atr", "range_pos", "ema_fast_slow", "price_long"]
    vix_weight_col: Optional[str] = None
    if config.vix_regime_enabled and vix is not None and not vix.empty:
        vix_pctile = vix_regime_series(d.index, vix, config.vix_regime_lookback_days)
        # Centered/scaled percentile rank (roughly N(0, ~0.3)) so it sits on a
        # comparable scale to the other z-scored features before weighting.
        f["vix_regime"] = (vix_pctile - 0.5) * 2.0
        feature_cols = feature_cols + ["vix_regime"]
        vix_weight_col = "vix_regime"

    out = pd.DataFrame(index=d.index)
    out["next_open"] = d["Open"].shift(-1) / close - 1.0
    out["next_close_1"] = close.shift(-1) / close - 1.0
    out["next_close_2"] = close.shift(-2) / close - 1.0
    out["next_low_1"] = d["Low"].shift(-1) / close - 1.0
    out["next_high_1"] = d["High"].shift(-1) / close - 1.0
    low2 = pd.concat([d["Low"].shift(-1), d["Low"].shift(-2)], axis=1).min(axis=1)
    high2 = pd.concat([d["High"].shift(-1), d["High"].shift(-2)], axis=1).max(axis=1)
    out["min_low_2"] = low2 / close - 1.0
    out["max_high_2"] = high2 / close - 1.0

    current = f.iloc[-1][feature_cols].replace([np.inf, -np.inf], np.nan)
    train = pd.concat([f[feature_cols], out], axis=1).iloc[:-2].replace([np.inf, -np.inf], np.nan).dropna()
    if current.isna().any() or len(train) < config.probability_min_samples:
        return HistoricalProbability()

    means = train[feature_cols].mean()
    stds = train[feature_cols].std().replace(0, 1.0).fillna(1.0)
    z_train = (train[feature_cols] - means) / stds
    z_current = (current - means) / stds
    if vix_weight_col is not None:
        # Up-weight the regime feature so neighbors are drawn preferentially from
        # a similar volatility environment, without hard-filtering the pool (which
        # would starve GLD's smaller sample).
        z_train[vix_weight_col] *= config.vix_feature_weight
        z_current[vix_weight_col] *= config.vix_feature_weight
    dist = np.sqrt(((z_train - z_current) ** 2).mean(axis=1))
    n = min(max(config.probability_min_samples, config.probability_neighbors), len(train))
    chosen_idx = dist.nsmallest(n).index
    chosen = train.loc[chosen_idx]
    chosen_dist = dist.loc[chosen_idx].to_numpy(dtype=float)
    scale = max(float(np.median(chosen_dist)) if len(chosen_dist) else 1.0, 0.25)
    weights = np.exp(-chosen_dist / scale)
    weights = weights / max(float(weights.sum()), 1e-12)

    next_open = chosen["next_open"].to_numpy(dtype=float)
    close1 = chosen["next_close_1"].to_numpy(dtype=float)
    close2 = chosen["next_close_2"].to_numpy(dtype=float)
    low1 = chosen["next_low_1"].to_numpy(dtype=float)
    high1 = chosen["next_high_1"].to_numpy(dtype=float)
    minlow2 = chosen["min_low_2"].to_numpy(dtype=float)
    maxhigh2 = chosen["max_high_2"].to_numpy(dtype=float)

    return HistoricalProbability(
        valid=True,
        samples=len(chosen),
        p_up_next_open=_weighted_mean((next_open > 0).astype(float), weights),
        p_up_1d_close=_weighted_mean((close1 > 0).astype(float), weights),
        p_up_2d_close=_weighted_mean((close2 > 0).astype(float), weights),
        avg_overnight_return=_weighted_mean(next_open, weights),
        avg_1d_return=_weighted_mean(close1, weights),
        avg_2d_return=_weighted_mean(close2, weights),
        weights=weights.tolist(),
        next_open=next_open.tolist(),
        next_close_1=close1.tolist(),
        next_close_2=close2.tolist(),
        next_low_1=low1.tolist(),
        next_high_1=high1.tolist(),
        min_low_2=minlow2.tolist(),
        max_high_2=maxhigh2.tolist(),
    )



def probability_effective_sample(prob: HistoricalProbability) -> float:
    """Effective sample size for weighted historical analogs.

    If all weights are equal, this is roughly the number of analogs. If a few analogs
    dominate, the effective sample is smaller, which is a useful warning against
    displaying an over-precise probability.
    """
    if not prob.valid or not prob.weights:
        return 0.0
    w = np.asarray(prob.weights, dtype=float)
    s = float(np.sum(w * w))
    return 0.0 if s <= 0 else 1.0 / s


def probability_uncertainty_90(prob: HistoricalProbability) -> Tuple[float, float, float]:
    """Approximate 90% uncertainty band for weighted P(up) next open.

    This is an approximation, not a formal confidence interval for a machine-learning
    forecast. It is included to stop 70% from looking more certain than it really is.
    """
    p = float(prob.p_up_next_open)
    n_eff = probability_effective_sample(prob)
    if n_eff <= 1:
        return p, p, n_eff
    se = math.sqrt(max(p * (1.0 - p), 1e-9) / n_eff)
    z = 1.6448536269514722
    return max(0.0, p - z * se), min(1.0, p + z * se), n_eff


def _forecast_direction_from_state(state: MarketState) -> str:
    """v13: there is exactly one next-open directional answer."""
    if state.overnight_direction in {"bullish", "bearish", "neutral"}:
        return state.overnight_direction
    return "neutral"


def candidate_historical_probability(
    prob: HistoricalProbability,
    spread_type: str,
    breakeven_return: float,
    horizon_sessions: int,
) -> Tuple[float, float]:
    """Return (finish-profitable probability, no-breach probability) from analog days."""
    if not prob.valid or not prob.weights:
        return 0.50, 0.50
    w = np.asarray(prob.weights, dtype=float)
    if horizon_sessions <= 1:
        closes = np.asarray(prob.next_close_1, dtype=float)
        lows = np.asarray(prob.next_low_1, dtype=float)
        highs = np.asarray(prob.next_high_1, dtype=float)
    else:
        closes = np.asarray(prob.next_close_2, dtype=float)
        lows = np.asarray(prob.min_low_2, dtype=float)
        highs = np.asarray(prob.max_high_2, dtype=float)
    if spread_type == "bull_put":
        p_finish = _weighted_mean((closes > breakeven_return).astype(float), w)
        p_no_breach = _weighted_mean((lows > breakeven_return).astype(float), w)
    else:
        p_finish = _weighted_mean((closes < breakeven_return).astype(float), w)
        p_no_breach = _weighted_mean((highs < breakeven_return).astype(float), w)
    return p_finish, p_no_breach


def historical_open_survival_probability(
    prob: HistoricalProbability, spread_type: str, breakeven_return: float
) -> float:
    """Weighted probability that the next regular-session open remains on the profitable side of breakeven."""
    if not prob.valid or not prob.weights or not prob.next_open:
        return 0.50
    w = np.asarray(prob.weights, dtype=float)
    opens = np.asarray(prob.next_open, dtype=float)
    if spread_type == "bull_put":
        return _weighted_mean((opens > breakeven_return).astype(float), w)
    return _weighted_mean((opens < breakeven_return).astype(float), w)


def evaluate_overnight_eligibility(
    state: MarketState,
    spread_type: str,
    consensus_pop: float,
    no_breach_prob: float,
    open_survival_prob: float,
    be_gap_buffer_mult: float,
    be_expected_move_buffer_mult: float,
    config: Config,
    breakeven_distance_pct: Optional[float] = None,
) -> Tuple[str, str]:
    """Return PASS / CAUTION / FAIL and concise reasons for an overnight hold."""
    if not config.overnight_eligibility_enabled:
        return "N/A", "Overnight eligibility engine disabled."

    reasons: List[str] = []
    conflict = False
    if state.probability.valid:
        p_up = state.probability.p_up_next_open
        th = config.overnight_conflict_threshold
        if spread_type == "bear_call" and p_up >= th:
            conflict = True
            reasons.append(f"analogs favor an up-gap/open ({p_up*100:.0f}% P(up))")
        elif spread_type == "bull_put" and p_up <= 1.0 - th:
            conflict = True
            reasons.append(f"analogs favor a down-gap/open ({(1.0-p_up)*100:.0f}% P(down))")

    # v10: live futures already moving against the position eats into (or wipes out)
    # the breakeven cushion in real time -- this is direct evidence, not a historical
    # estimate, so treat it as its own conflict/fail source.
    futures_conflict = False
    futures_move = state.futures_overnight_move_pct
    if futures_move is not None and abs(futures_move) >= config.futures_conflict_move_pct:
        adverse = (spread_type == "bull_put" and futures_move < 0) or (
            spread_type == "bear_call" and futures_move > 0
        )
        if adverse:
            futures_conflict = True
            reasons.append(
                f"{state.futures_symbol} already moved {futures_move*100:+.2f}% against this position"
            )
            if breakeven_distance_pct and breakeven_distance_pct > 0:
                consumed = abs(futures_move) / breakeven_distance_pct
                if consumed >= config.futures_block_buffer_fraction:
                    reasons.append(f"futures move already ~{consumed*100:.0f}% of BE cushion")

    fail_checks = [
        futures_conflict
        and breakeven_distance_pct is not None
        and breakeven_distance_pct > 0
        and (abs(futures_move) / breakeven_distance_pct) >= config.futures_block_buffer_fraction,
        open_survival_prob < config.overnight_caution_open_survival,
        no_breach_prob < config.overnight_caution_no_breach,
        consensus_pop < config.overnight_caution_consensus_pop,
        be_gap_buffer_mult < config.overnight_caution_gap_buffer_mult,
        be_expected_move_buffer_mult < config.overnight_caution_expected_move_buffer_mult,
    ]

    pass_checks = [
        open_survival_prob >= config.overnight_pass_open_survival,
        no_breach_prob >= config.overnight_pass_no_breach,
        consensus_pop >= config.overnight_pass_consensus_pop,
        be_gap_buffer_mult >= config.overnight_pass_gap_buffer_mult,
        be_expected_move_buffer_mult >= config.overnight_pass_expected_move_buffer_mult,
        state.event_risk.severity != "HIGH",
        not conflict,
        not futures_conflict,
    ]

    if open_survival_prob < config.overnight_pass_open_survival:
        reasons.append(f"next-open survival only {open_survival_prob*100:.0f}%")
    if no_breach_prob < config.overnight_pass_no_breach:
        reasons.append(f"analog no-breach only {no_breach_prob*100:.0f}%")
    if be_gap_buffer_mult < config.overnight_pass_gap_buffer_mult:
        reasons.append(f"BE cushion only {be_gap_buffer_mult:.2f}x p90 gap")
    if be_expected_move_buffer_mult < config.overnight_pass_expected_move_buffer_mult:
        reasons.append(f"BE cushion only {be_expected_move_buffer_mult:.2f}x implied move")
    if state.event_risk.severity == "HIGH":
        reasons.append("HIGH scheduled-event risk")

    if any(fail_checks) or (conflict and open_survival_prob < config.overnight_pass_open_survival):
        label = "FAIL"
    elif all(pass_checks):
        label = "PASS"
    else:
        label = "CAUTION"

    if not reasons:
        reasons.append("direction, gap cushion, probability, and no-breach checks align")
    return label, "; ".join(reasons[:5])


def same_day_intraday(m5: pd.DataFrame) -> pd.DataFrame:
    d = m5.copy().dropna(subset=["Close"])
    if d.empty:
        return d
    return d[d.index.date == d.index[-1].date()].copy()


def intraday_vwap(day: pd.DataFrame) -> float:
    if day.empty:
        return float("nan")
    tp = (day["High"] + day["Low"] + day["Close"]) / 3.0
    vol = day["Volume"].fillna(0)
    denom = vol.cumsum().replace(0, np.nan)
    vwap = (tp * vol).cumsum() / denom
    if pd.isna(vwap.iloc[-1]):
        return float(day["Close"].iloc[-1])
    return float(vwap.iloc[-1])


# ---------------------------- Event awareness ----------------------------

class _TextExtractor(HTMLParser):
    """Small stdlib HTML-to-text helper; avoids another dependency."""
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag in {"br", "p", "div", "h1", "h2", "h3", "h4", "h5", "li", "tr", "td", "th"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "div", "h1", "h2", "h3", "h4", "h5", "li", "tr", "td", "th"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


_FOMC_FALLBACK = {
    2026: [
        (1, 28), (3, 18), (4, 29), (6, 17),
        (7, 29), (9, 16), (10, 28), (12, 9),
    ],
    2027: [
        (1, 27), (3, 17), (4, 28), (6, 9),
        (7, 28), (9, 15), (10, 27), (12, 8),
    ],
}

_MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def _parse_fomc_calendar_html(html: str, years: Iterable[int]) -> List[dt.date]:
    parser = _TextExtractor()
    parser.feed(html)
    lines = [re.sub(r"\s+", " ", x).strip() for x in parser.text().splitlines()]
    lines = [x for x in lines if x]
    wanted = set(years)
    out: List[dt.date] = []
    current_year: Optional[int] = None
    for i, line in enumerate(lines):
        m_year = re.search(r"\b(20\d{2})\s+FOMC Meetings\b", line)
        if m_year:
            y = int(m_year.group(1))
            current_year = y if y in wanted else None
            continue
        if current_year is None or line not in _MONTHS:
            continue
        month = _MONTHS[line]
        # The official calendar places the date/range right after the month label.
        for nxt in lines[i + 1:i + 5]:
            m_date = re.fullmatch(r"(\d{1,2})(?:\s*-\s*(\d{1,2}))?\*?", nxt)
            if not m_date:
                continue
            decision_day = int(m_date.group(2) or m_date.group(1))
            try:
                out.append(dt.date(current_year, month, decision_day))
            except ValueError:
                pass
            break
    return sorted(set(out))


def get_fomc_dates(config: Config) -> List[dt.date]:
    """Best-effort official Fed calendar fetch, cached; falls back to embedded known dates."""
    now_ts = time.time()
    cached_time = float(FOMC_CACHE.get("time", 0.0))
    cached_dates = FOMC_CACHE.get("dates", [])
    if isinstance(cached_dates, list) and cached_dates and now_ts - cached_time < config.fed_data_cache_seconds:
        return cached_dates

    today = now_local(config.market_timezone).date()
    years = [today.year, today.year + 1]
    dates: List[dt.date] = []
    try:
        resp = requests.get(
            "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
            headers={"User-Agent": "Mozilla/5.0 trading-alert-bot/1.0"},
            timeout=12,
        )
        resp.raise_for_status()
        dates = _parse_fomc_calendar_html(resp.text, years)
    except Exception as exc:
        print(f"[warn] Federal Reserve FOMC calendar unavailable; using fallback when possible: {exc}", file=sys.stderr)

    if not dates:
        for year in years:
            for month, day in _FOMC_FALLBACK.get(year, []):
                dates.append(dt.date(year, month, day))

    dates = sorted(set(dates))
    FOMC_CACHE["time"] = now_ts
    FOMC_CACHE["dates"] = dates
    return dates


def _fred_latest(series_id: str, timeout: int = 10) -> Optional[float]:
    try:
        resp = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv",
            params={"id": series_id},
            headers={"User-Agent": "Mozilla/5.0 trading-alert-bot/1.0"},
            timeout=timeout,
        )
        resp.raise_for_status()
        frame = pd.read_csv(io.StringIO(resp.text))
        if frame.empty or series_id not in frame.columns:
            return None
        vals = pd.to_numeric(frame[series_id], errors="coerce").dropna()
        return float(vals.iloc[-1]) if not vals.empty else None
    except Exception:
        return None


def get_fed_rate_note(config: Config) -> str:
    """Current effective fed-funds rate plus target range, when FRED is reachable."""
    if not config.show_fed_rate_context:
        return ""
    now_ts = time.time()
    cached_time = float(FED_RATE_CACHE.get("time", 0.0))
    cached_note = str(FED_RATE_CACHE.get("note", ""))
    if cached_note and now_ts - cached_time < config.fed_data_cache_seconds:
        return cached_note

    effective = _fred_latest("DFF")
    lower = _fred_latest("DFEDTARL")
    upper = _fred_latest("DFEDTARU")
    pieces: List[str] = []
    if effective is not None:
        pieces.append(f"effective fed funds {effective:.2f}%")
    if lower is not None and upper is not None:
        pieces.append(f"target {lower:.2f}-{upper:.2f}%")
    note = " | ".join(pieces)
    FED_RATE_CACHE["time"] = now_ts
    FED_RATE_CACHE["note"] = note
    return note


def next_fomc_note(config: Config) -> Tuple[Optional[int], str]:
    if not config.auto_fomc_events:
        return None, ""
    today = now_local(config.market_timezone).date()
    future = [d for d in get_fomc_dates(config) if d >= today]
    if not future:
        return None, ""
    nxt = min(future)
    days = (nxt - today).days
    label = "today" if days == 0 else "tomorrow" if days == 1 else f"in {days}d"
    return days, f"FOMC rate decision {label} ({nxt.isoformat()})"


def parse_manual_events(config: Config, symbol: str) -> List[Tuple[dt.date, str]]:
    """
    Accepts either:
      MANUAL_EVENTS=2026-08-25:CPI;2026-08-27:FOMC
    or scoped format:
      MANUAL_EVENTS=2026-08-25|ALL|CPI;2026-08-27|SPY|FOMC
    """
    events: List[Tuple[dt.date, str]] = []
    raw = (config.manual_events or "").strip()
    if not raw:
        return events
    for chunk in raw.split(";"):
        item = chunk.strip()
        if not item:
            continue
        try:
            if "|" in item:
                date_part, scope, label = item.split("|", 2)
                scope = scope.strip().upper()
                if scope not in {"ALL", symbol.upper()}:
                    continue
            elif ":" in item:
                date_part, label = item.split(":", 1)
            else:
                continue
            event_date = dt.datetime.strptime(date_part.strip()[:10], "%Y-%m-%d").date()
            events.append((event_date, label.strip()))
        except Exception:
            continue
    return events


def parse_timed_manual_events(config: Config, symbol: str) -> List[Tuple[dt.datetime, str]]:
    """Parse manual events with optional clock times.

    Supported examples (market timezone is used when no offset is supplied):
      2026-09-21 05:30|ALL|CPI
      2026-09-20 13:05|SPY|AAPL product event
      2026-09-20|SPY|Unknown-time company event

    Date-only entries are placed at 12:00 local for display purposes, but callers can
    treat them as timing-uncertain when the date overlaps the overnight holding window.
    """
    raw = (config.manual_events or "").strip()
    if not raw:
        return []
    tz = ZoneInfo(config.market_timezone)
    out: List[Tuple[dt.datetime, str]] = []
    for chunk in raw.split(";"):
        item = chunk.strip()
        if not item:
            continue
        try:
            if "|" in item:
                when_part, scope, label = item.split("|", 2)
                scope = scope.strip().upper()
                if scope not in {"ALL", symbol.upper()}:
                    continue
            elif ":" in item and len(item.split(":", 1)[0].strip()) == 10:
                # Preserve legacy date:label format. The first colon separates the label,
                # not a clock time, so this remains a date-only event.
                when_part, label = item.split(":", 1)
            else:
                continue

            raw_when = when_part.strip()
            parsed: Optional[dt.datetime] = None
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
                try:
                    parsed = dt.datetime.strptime(raw_when, fmt)
                    if fmt == "%Y-%m-%d":
                        parsed = parsed.replace(hour=12, minute=0)
                    break
                except ValueError:
                    pass
            if parsed is None:
                continue
            out.append((parsed.replace(tzinfo=tz), label.strip()))
        except Exception:
            continue
    return sorted(out, key=lambda x: x[0])


def intended_exit_datetime(entry_dt: dt.datetime, config: Config) -> dt.datetime:
    """Configured next-session exit time, normally shortly after the 06:30 PT open."""
    nxt_open = next_regular_open(entry_dt, config.market_timezone)
    exit_t = _parse_hhmm(config.intended_exit_time)
    candidate = dt.datetime.combine(nxt_open.date(), exit_t, tzinfo=ZoneInfo(config.market_timezone))
    # If a user sets an exit before the normal open, keep it on the same next trading day.
    return candidate


def hold_window_event_notes(
    symbol: str,
    entry_dt: dt.datetime,
    exit_dt: dt.datetime,
    config: Config,
) -> List[str]:
    """Best-effort events that may occur between a late-day entry and next-morning exit."""
    notes: List[str] = []

    # Exact-time manual events are the most reliable way to capture CPI/product events
    # and similar catalysts not covered by the free earnings feeds.
    for when, label in parse_timed_manual_events(config, symbol):
        if entry_dt <= when <= exit_dt:
            notes.append(f"{label} at {when.strftime('%m-%d %H:%M')} PT")

    # FOMC decisions are normally released at 11:00 PT / 14:00 ET. This usually falls
    # outside a close-to-next-open hold, but checking the actual window prevents a generic
    # 'tomorrow' label from overstating risk for a position intended to exit at the open.
    if config.auto_fomc_events:
        tz = ZoneInfo(config.market_timezone)
        for d in get_fomc_dates(config):
            when = dt.datetime.combine(d, dt.time(11, 0), tzinfo=tz)
            if entry_dt <= when <= exit_dt:
                notes.append(f"FOMC rate decision at {when.strftime('%m-%d %H:%M')} PT")

    # Earnings feeds are often date-only. If a watched mega-cap reports on either the
    # entry date or next-open date, flag the timing as uncertain rather than pretending
    # we know whether it is BMO or AMC.
    if symbol.upper() in {"SPY", "QQQ", "XSP", "SPX"}:
        alpha_dates = alpha_vantage_earnings_dates(config)
        watchlist = [x.strip().upper() for x in config.major_earnings_watchlist.split(",") if x.strip()]
        relevant_dates = {entry_dt.date(), exit_dt.date()}
        for name in watchlist:
            ed = alpha_dates.get(name) or yfinance_next_earnings_date(name)
            if ed in relevant_dates:
                notes.append(f"{name} earnings date {ed.isoformat()} (release time unknown)")

    return list(dict.fromkeys(notes))


def _extract_calendar_earnings_date(cal: object) -> Optional[dt.date]:
    try:
        if isinstance(cal, dict):
            for key, value in cal.items():
                if "earn" not in str(key).lower():
                    continue
                vals = value if isinstance(value, (list, tuple)) else [value]
                for raw in vals:
                    if raw is not None:
                        return pd.to_datetime(raw).date()
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            for col in cal.columns:
                if "earn" in str(col).lower():
                    raw = cal.iloc[0][col]
                    if pd.notna(raw):
                        return pd.to_datetime(raw).date()
    except Exception:
        return None
    return None


def yfinance_next_earnings_date(symbol: str) -> Optional[dt.date]:
    now_ts = time.time()
    cached = EARNINGS_CACHE.get(symbol)
    if cached and now_ts - cached[0] < 6 * 3600:
        return cached[1]

    result: Optional[dt.date] = None
    try:
        ticker = yf.Ticker(symbol)
        result = _extract_calendar_earnings_date(getattr(ticker, "calendar", None))
        if result is None:
            try:
                ed = ticker.get_earnings_dates(limit=12)
                if ed is not None and not ed.empty:
                    today = dt.date.today()
                    future = [pd.to_datetime(x).date() for x in ed.index if pd.to_datetime(x).date() >= today]
                    if future:
                        result = min(future)
            except Exception:
                pass
    except Exception:
        result = None

    EARNINGS_CACHE[symbol] = (now_ts, result)
    return result


def alpha_vantage_earnings_dates(config: Config) -> Dict[str, dt.date]:
    """Optional second source. If no API key is set, returns an empty dict."""
    if not config.alpha_vantage_api_key:
        return {}
    now_ts = time.time()
    cached_time = float(ALPHA_EARNINGS_CACHE.get("time", 0.0))
    cached_dates = ALPHA_EARNINGS_CACHE.get("dates", {})
    if isinstance(cached_dates, dict) and now_ts - cached_time < 6 * 3600:
        return cached_dates

    dates: Dict[str, dt.date] = {}
    try:
        resp = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "EARNINGS_CALENDAR",
                "horizon": "3month",
                "apikey": config.alpha_vantage_api_key,
            },
            timeout=15,
        )
        resp.raise_for_status()
        reader = csv.DictReader(io.StringIO(resp.text))
        for row in reader:
            sym = (row.get("symbol") or "").strip().upper()
            raw = (row.get("reportDate") or "").strip()
            if sym and raw:
                try:
                    dates[sym] = dt.datetime.strptime(raw, "%Y-%m-%d").date()
                except ValueError:
                    pass
    except Exception as exc:
        print(f"[warn] Alpha Vantage earnings calendar unavailable: {exc}", file=sys.stderr)

    ALPHA_EARNINGS_CACHE["time"] = now_ts
    ALPHA_EARNINGS_CACHE["dates"] = dates
    return dates


def get_event_risk(symbol: str, config: Config) -> EventRisk:
    if not config.event_guard_enabled:
        return EventRisk(note="Event guard disabled", severity="LOW", block=False)

    today = now_local(config.market_timezone).date()
    items: List[Tuple[int, str]] = []

    fomc_days, fomc_note = next_fomc_note(config)
    fed_rate_note = get_fed_rate_note(config)
    if fomc_days is not None and 0 <= fomc_days <= config.event_lookahead_days:
        items.append((fomc_days, fomc_note))

    # Earnings are relevant mainly for index ETFs. GLD is driven more by macro events.
    watchlist: List[str] = []
    if symbol.upper() in {"SPY", "QQQ", "XSP", "SPX"}:
        watchlist = [x.strip().upper() for x in config.major_earnings_watchlist.split(",") if x.strip()]

    alpha_dates = alpha_vantage_earnings_dates(config)
    for name in watchlist:
        ed = alpha_dates.get(name) or yfinance_next_earnings_date(name)
        if ed is None:
            continue
        days = (ed - today).days
        if 0 <= days <= config.event_lookahead_days:
            wording = "today" if days == 0 else "tomorrow" if days == 1 else f"in {days}d"
            items.append((days, f"{name} earnings {wording}"))

    for event_date, label in parse_manual_events(config, symbol):
        days = (event_date - today).days
        if 0 <= days <= config.event_lookahead_days:
            wording = "today" if days == 0 else "tomorrow" if days == 1 else f"in {days}d"
            items.append((days, f"{label} {wording}"))

    if not items:
        display_fomc = fomc_note if fomc_days is not None and fomc_days <= config.fomc_display_lookahead_days else ""
        return EventRisk(next_fomc=display_fomc, fed_rate=fed_rate_note)

    items = sorted(set(items), key=lambda x: (x[0], x[1]))
    nearest = items[0][0]
    if nearest == 0:
        severity = "HIGH"
        block = config.block_same_day_events
    elif nearest == 1:
        severity = "HIGH"
        block = config.block_next_day_events
    else:
        severity = "MEDIUM"
        block = config.block_same_week_events

    note = " | ".join(label for _, label in items[:8])
    display_fomc = fomc_note if fomc_days is not None and fomc_days <= config.fomc_display_lookahead_days else ""
    return EventRisk(note=note, severity=severity, block=block, next_fomc=display_fomc, fed_rate=fed_rate_note)


# ---------------------------- Market regime ----------------------------

def _intraday_tail_return(day: pd.DataFrame, minutes: int) -> float:
    if day.empty or len(day) < 2:
        return 0.0
    bars = max(1, minutes // 5)
    if len(day) <= bars:
        return float(day["Close"].iloc[-1] / day["Open"].iloc[0] - 1.0)
    return float(day["Close"].iloc[-1] / day["Close"].iloc[-(bars + 1)] - 1.0)


def large_day_overnight_context(
    daily: pd.DataFrame,
    day: pd.DataFrame,
    atr14: float,
    vix: pd.DataFrame,
    config: Config,
) -> LargeDayOvernight:
    """SPY-specific close->next-open analog model emphasizing gap/session decomposition.

    Unlike the broad k-NN model, this explicitly distinguishes overnight gap from
    regular-session momentum and asks what happened at the *next open* after days
    with a similar total move, gap contribution, session contribution, ATR size,
    closing location, and volatility regime.
    """
    if not config.large_day_model_enabled or day.empty or len(daily) < 100:
        return LargeDayOvernight()

    # Explicitly strip any forming Yahoo daily candle before choosing yesterday's close.
    d = _completed_daily_before_live_day(daily[["Open", "High", "Low", "Close"]], day)
    if d.empty:
        return LargeDayOvernight(note="no completed daily bar before live session")
    prev_close_live = float(d["Close"].iloc[-1])
    live_open = float(day["Open"].iloc[0])
    live_close = float(day["Close"].iloc[-1])
    live_high = float(day["High"].max())
    live_low = float(day["Low"].min())
    total_ret = live_close / prev_close_live - 1.0
    gap_ret = live_open / prev_close_live - 1.0
    session_ret = live_close / live_open - 1.0
    move_atr = (live_close - prev_close_live) / max(atr14, 1e-9)
    range_pos = (live_close - live_low) / max(live_high - live_low, 1e-9)
    final30 = _intraday_tail_return(day, 30)
    final60 = _intraday_tail_return(day, 60)

    prev_close = d["Close"].shift(1)
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - prev_close).abs(),
        (d["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(config.atr_period, min_periods=max(5, config.atr_period // 2)).mean()
    rng = (d["High"] - d["Low"]).replace(0, np.nan)

    f = pd.DataFrame(index=d.index)
    f["total"] = d["Close"] / prev_close - 1.0
    f["gap"] = d["Open"] / prev_close - 1.0
    f["session"] = d["Close"] / d["Open"] - 1.0
    f["atrmove"] = (d["Close"] - prev_close) / atr.replace(0, np.nan)
    f["rangepos"] = (d["Close"] - d["Low"]) / rng
    if config.vix_regime_enabled and vix is not None and not vix.empty:
        f["vixpct"] = vix_regime_series(d.index, vix, config.vix_regime_lookback_days)
    else:
        f["vixpct"] = 0.5

    nxt = d["Open"].shift(-1) / d["Close"] - 1.0
    train = f.copy()
    train["next_open"] = nxt
    train = train.iloc[:-1].replace([np.inf, -np.inf], np.nan).dropna()
    if len(train) < config.large_day_min_samples:
        return LargeDayOvernight(
            total_return=total_ret, gap_return=gap_ret, session_return=session_ret,
            move_atr=move_atr, range_position=range_pos,
            final_30m_return=final30, final_60m_return=final60,
            note="insufficient large-day analog sample",
        )

    cols = ["total", "gap", "session", "atrmove", "rangepos", "vixpct"]
    current = pd.Series({
        "total": total_ret, "gap": gap_ret, "session": session_ret,
        "atrmove": move_atr, "rangepos": range_pos,
        "vixpct": float(train["vixpct"].iloc[-1]) if "vixpct" in train else 0.5,
    })
    means = train[cols].mean()
    stds = train[cols].std().replace(0, 1.0).fillna(1.0)
    zt = (train[cols] - means) / stds
    zc = (current - means) / stds
    # Total move and ATR extension matter most for this specialized model.
    zt["total"] *= 1.6; zc["total"] *= 1.6
    zt["atrmove"] *= 1.4; zc["atrmove"] *= 1.4
    zt["gap"] *= 1.2; zc["gap"] *= 1.2
    dist = np.sqrt(((zt - zc) ** 2).mean(axis=1))
    n = min(max(config.large_day_min_samples, config.large_day_neighbors), len(train))
    idx = dist.nsmallest(n).index
    chosen = train.loc[idx]
    dd = dist.loc[idx].to_numpy(float)
    scale = max(float(np.median(dd)) if len(dd) else 1.0, 0.25)
    w = np.exp(-dd / scale)
    w = w / max(float(w.sum()), 1e-12)
    opens = chosen["next_open"].to_numpy(float)
    p_up = _weighted_mean((opens > 0).astype(float), w)
    direction = 1.0 if total_ret >= 0 else -1.0
    reversal = _weighted_mean(((opens * direction) < 0).astype(float), w)
    avg_open = _weighted_mean(opens, w)
    large = abs(total_ret) >= config.large_day_min_abs_return
    note = (
        f"{'large-day' if large else 'normal-day'} analogs: P(up next open)={p_up*100:.0f}%, "
        f"reversal={reversal*100:.0f}%, avg gap={avg_open*100:+.2f}%"
    )
    return LargeDayOvernight(
        valid=True, samples=len(chosen), total_return=total_ret, gap_return=gap_ret,
        session_return=session_ret, move_atr=move_atr, range_position=range_pos,
        final_30m_return=final30, final_60m_return=final60,
        p_up_next_open=p_up, p_reversal_next_open=reversal,
        avg_next_open_return=avg_open, note=note,
    )


def _late_tape_probability(large: LargeDayOvernight, config: Config) -> float:
    """Map the final 30/60 minute tape into a deliberately bounded probability input."""
    if not large.valid:
        return 0.50
    adj = 0.0
    adj += max(
        -config.final_30m_weight,
        min(config.final_30m_weight, large.final_30m_return / 0.01 * config.final_30m_weight),
    )
    adj += max(
        -config.final_60m_weight,
        min(config.final_60m_weight, large.final_60m_return / 0.01 * config.final_60m_weight),
    )
    return max(0.35, min(0.65, 0.50 + adj))


def _technical_probability(daily_signal: int, m30_signal: int, m5_signal: int) -> float:
    """Bound technical alignment to 42%-58%; it is evidence, never a literal probability."""
    score = 2 * daily_signal + 2 * m30_signal + m5_signal
    normalized = max(-1.0, min(1.0, score / 5.0))
    return 0.50 + 0.08 * normalized


def _vix_regime_next_open_probability(
    daily: pd.DataFrame,
    vix: pd.DataFrame,
    current_bucket: str,
    config: Config,
) -> Tuple[Optional[float], int]:
    """Simple same-VIX-bucket next-open prior, kept separate and low-weighted."""
    if daily is None or daily.empty or not {"Open", "Close"}.issubset(daily.columns):
        return None, 0
    d = daily[["Open", "Close"]].dropna().copy()
    if not config.vix_regime_enabled or vix is None or vix.empty or len(d) < 80:
        return None, 0
    pct = vix_regime_series(d.index, vix, config.vix_regime_lookback_days)
    buckets = pct.apply(vix_bucket_label)
    nxt = d["Open"].shift(-1) / d["Close"] - 1.0
    mask = (buckets == current_bucket) & nxt.notna()
    vals = nxt[mask]
    if current_bucket == "unknown" or len(vals) < max(25, config.probability_min_samples):
        return None, int(len(vals))
    return float((vals > 0).mean()), int(len(vals))


def _combined_effective_n(general: HistoricalProbability, large: LargeDayOvernight) -> float:
    vals: List[float] = []
    if general.valid and general.weights:
        w = np.asarray(general.weights, dtype=float)
        denom = float(np.square(w).sum())
        if denom > 0:
            vals.append(float((w.sum() ** 2) / denom))
    if large.valid and large.samples > 0:
        vals.append(float(large.samples))
    if not vals:
        return 0.0
    return float(sum(vals))


def build_spy_consensus_forecast(
    general: HistoricalProbability,
    large: LargeDayOvernight,
    daily: pd.DataFrame,
    vix: pd.DataFrame,
    vix_bucket: str,
    daily_signal: int,
    m30_signal: int,
    m5_signal: int,
    config: Config,
) -> ConsensusForecast:
    """Create the ONE next-open directional probability used by v13.

    The components answer the same question: whether SPY will open above the late-day
    reference/close. Technical alignment is bounded to a very small contribution.
    Options POP remains spread-specific and is intentionally not repurposed as a
    directional SPY forecast.
    """
    components: Dict[str, float] = {}
    weights: Dict[str, float] = {}

    if general.valid:
        components["Historical analogs"] = float(general.p_up_next_open)
        weights["Historical analogs"] = max(0.0, config.consensus_general_weight)
    if large.valid:
        components["Large-day analogs"] = float(large.p_up_next_open)
        weights["Large-day analogs"] = max(0.0, config.consensus_large_day_weight)
        components["Late-day tape"] = float(_late_tape_probability(large, config))
        weights["Late-day tape"] = max(0.0, config.consensus_late_tape_weight)

    vix_p, vix_n = _vix_regime_next_open_probability(daily, vix, vix_bucket, config)
    if vix_p is not None:
        components["VIX-regime prior"] = vix_p
        weights["VIX-regime prior"] = max(0.0, config.consensus_vix_regime_weight)

    components["Technical contribution"] = _technical_probability(daily_signal, m30_signal, m5_signal)
    weights["Technical contribution"] = max(0.0, config.consensus_technical_weight)

    valid_names = [k for k in components if weights.get(k, 0.0) > 0]
    if not valid_names:
        return ConsensusForecast(note="insufficient consensus evidence")

    total_w = sum(weights[k] for k in valid_names)
    norm_w = {k: weights[k] / total_w for k in valid_names}
    raw_p = sum(components[k] * norm_w[k] for k in valid_names)
    variance = sum(norm_w[k] * (components[k] - raw_p) ** 2 for k in valid_names)
    model_std = math.sqrt(max(variance, 0.0))

    if model_std <= config.consensus_agreement_high_max_std:
        agreement = "HIGH"
    elif model_std <= config.consensus_agreement_moderate_max_std:
        agreement = "MODERATE"
    else:
        agreement = "LOW"

    # Local calibration is intentionally a gentle shrink, not a replacement for the model.
    calibrated, cal_n, _ = calibration_for_probability("SPY", raw_p, config)
    cal_weight = 0.0
    final_p = raw_p
    if config.consensus_use_local_calibration and calibrated is not None and cal_n >= config.calibration_min_resolved:
        ramp = min(1.0, cal_n / max(float(config.calibration_min_resolved * 4), 1.0))
        cal_weight = min(max(config.consensus_max_calibration_weight, 0.0), 0.75) * ramp
        final_p = (1.0 - cal_weight) * raw_p + cal_weight * calibrated

    final_p = max(0.02, min(0.98, float(final_p)))
    edge = final_p - 0.50
    if edge >= config.consensus_direction_min_edge:
        direction = "bullish"
    elif edge <= -config.consensus_direction_min_edge:
        direction = "bearish"
    else:
        direction = "neutral"
    strength = int(min(100, abs(edge) / 0.25 * 100))

    n_eff = _combined_effective_n(general, large)
    # Heuristic range combines sampling uncertainty with inter-model disagreement.
    if n_eff > 1:
        sample_se = math.sqrt(max(final_p * (1.0 - final_p), 1e-9) / n_eff)
    else:
        sample_se = 0.12
    half_width = min(0.30, max(0.05, 1.645 * sample_se + 0.75 * model_std))
    lo = max(0.0, final_p - half_width)
    hi = min(1.0, final_p + half_width)

    detail = ", ".join(f"{k}={components[k]*100:.0f}%" for k in valid_names)
    if calibrated is not None and cal_weight > 0:
        detail += f"; local calibration={calibrated*100:.0f}% (n={cal_n}, weight={cal_weight:.0%})"
    if vix_p is not None:
        detail += f"; VIX-regime n={vix_n}"

    return ConsensusForecast(
        raw_p_up=float(raw_p),
        p_up=final_p,
        p_down=1.0-final_p,
        direction=direction,
        strength=strength,
        agreement=agreement,
        weighted_std=float(model_std),
        range_low=float(lo),
        range_high=float(hi),
        components=components,
        component_weights=norm_w,
        effective_n=n_eff,
        calibrated_p_up=calibrated,
        calibration_n=cal_n,
        calibration_weight=cal_weight,
        note=detail,
    )


def analyze_market_state(symbol: str, daily: pd.DataFrame, m30: pd.DataFrame, m5: pd.DataFrame, config: Config) -> MarketState:
    daily = daily.copy().dropna(subset=["Close"])
    m30 = m30.copy().dropna(subset=["Close"])
    m5 = m5.copy().dropna(subset=["Close"])

    day = same_day_intraday(m5)
    spot = float(day["Close"].iloc[-1] if not day.empty else m5["Close"].iloc[-1])

    daily_signal, _, _, _ = ema_signal(
        daily["Close"], config.daily_ema_fast, config.daily_ema_slow, config.daily_ema_long
    )
    m30_signal, _, _, _ = ema_signal(
        m30["Close"], config.trend_ema_fast, config.trend_ema_slow
    )

    vwap = intraday_vwap(day)
    if not day.empty and len(day) >= config.intraday_ema_slow:
        m5_signal_raw, _, _, _ = ema_signal(
            day["Close"], config.intraday_ema_fast, config.intraday_ema_slow
        )
        if m5_signal_raw > 0 and spot > vwap:
            m5_signal = 1
        elif m5_signal_raw < 0 and spot < vwap:
            m5_signal = -1
        else:
            m5_signal = 0
    else:
        m5_signal = 0

    # Weighted agreement: daily and 30m matter more than the 5m tape.
    regime_score = 2 * daily_signal + 2 * m30_signal + m5_signal
    if regime_score >= 3:
        bias = "bullish"
    elif regime_score <= -3:
        bias = "bearish"
    else:
        bias = "neutral"

    confidence = int(min(100, abs(regime_score) / 5.0 * 100))

    # Even when the strict bias is neutral, preserve the direction of the weighted score
    # as a lower-conviction "lean" so the bot can show a cautious idea instead of silence.
    if regime_score > 0:
        directional_lean = "bullish"
    elif regime_score < 0:
        directional_lean = "bearish"
    else:
        # Tie-breaker uses the shortest available tape signal, then the 1-day move later below.
        directional_lean = "bullish" if m5_signal > 0 else "bearish" if m5_signal < 0 else "none"
    lean_confidence = confidence

    recent = m30.tail(min(config.support_resistance_lookback, len(m30)))
    support = float(recent["Low"].min())
    resistance = float(recent["High"].max())

    atr14 = calc_atr(daily, config.atr_period)
    ret_1d, ret_3d, ret_5d = recent_returns(daily)
    if directional_lean == "none" and abs(ret_1d) >= 0.001:
        directional_lean = "bullish" if ret_1d > 0 else "bearish"
        lean_confidence = max(20, min(35, int(abs(ret_1d) * 10000)))

    vix_daily = get_vix_context(config)
    # v10: regime-conditioned gap percentile needs a longer trailing window than the
    # v9 90-day slice so each VIX tertile still has enough observations; falls back
    # to the plain (unconditioned) stat automatically if the bucket sample is thin.
    avg_abs_gap, pctl_abs_gap, vix_bucket, vix_pctile = overnight_gap_stats_regime(
        daily.tail(max(750, config.vix_regime_lookback_days + 90)), vix_daily, config
    )
    vix_level = float(vix_daily["vix_close"].iloc[-1]) if not vix_daily.empty else None

    day_move_atr = 0.0
    if not day.empty and atr14 > 0:
        day_open = float(day["Open"].iloc[0])
        day_move_atr = (spot - day_open) / atr14

    # Historical IV uses daily close-to-close returns.
    daily_returns = daily["Close"].pct_change().dropna()
    historical_iv = float(daily_returns.std() * math.sqrt(252)) if not daily_returns.empty else config.iv_floor

    reasons = [
        f"TF signals D/30m/5m={daily_signal:+d}/{m30_signal:+d}/{m5_signal:+d}",
        f"1d/3d/5d={ret_1d*100:+.2f}%/{ret_3d*100:+.2f}%/{ret_5d*100:+.2f}%",
        f"day move={day_move_atr:+.2f} ATR",
    ]

    # Late reversal guard: do not keep calling continuation when the tape has flipped.
    if not day.empty and len(day) >= config.reversal_guard_bars + 1:
        recent_return = float(day["Close"].iloc[-1] / day["Close"].iloc[-(config.reversal_guard_bars + 1)] - 1.0)
        if bias == "bearish" and recent_return >= config.reversal_guard_threshold_pct and spot > vwap:
            bias = "neutral"
            confidence = min(confidence, 45)
            directional_lean = "bullish"
            lean_confidence = max(lean_confidence, 45)
            reasons.append(f"late upside reversal guard ({recent_return*100:+.2f}%, above VWAP)")
        elif bias == "bullish" and recent_return <= -config.reversal_guard_threshold_pct and spot < vwap:
            bias = "neutral"
            confidence = min(confidence, 45)
            directional_lean = "bearish"
            lean_confidence = max(lean_confidence, 45)
            reasons.append(f"late downside reversal guard ({recent_return*100:+.2f}%, below VWAP)")

    # Overextension guard: prevents chasing a large move with a same-direction credit spread.
    if config.block_overextended_continuation and abs(day_move_atr) >= config.overextension_atr_threshold:
        if bias == "bearish" and day_move_atr < 0:
            bias = "neutral"
            confidence = min(confidence, 50)
            directional_lean = "bullish"
            lean_confidence = max(lean_confidence, 40)
            reasons.append("selloff already overextended; rebound risk elevated; lean flipped bullish for rebound-risk context")
        elif bias == "bullish" and day_move_atr > 0:
            bias = "neutral"
            confidence = min(confidence, 50)
            directional_lean = "bearish"
            lean_confidence = max(lean_confidence, 40)
            reasons.append("rally already overextended; pullback risk elevated; lean flipped bearish for pullback-risk context")

    probability = historical_probability_context(daily, config, vix_daily, day=day)
    if probability.valid:
        regime_note = f", regime={vix_bucket}" if vix_bucket != "unknown" else ""
        reasons.append(
            f"analog P(up) next-open/1d/2d={probability.p_up_next_open*100:.0f}%/"
            f"{probability.p_up_1d_close*100:.0f}%/{probability.p_up_2d_close*100:.0f}% "
            f"(n={probability.samples}{regime_note})"
        )

    large_day = large_day_overnight_context(daily, day, atr14, vix_daily, config)
    consensus = (
        build_spy_consensus_forecast(
            probability, large_day, _completed_daily_before_live_day(daily, day),
            vix_daily, vix_bucket, daily_signal, m30_signal, m5_signal, config
        )
        if symbol.upper() == "SPY" and config.spy_overnight_model_enabled
        else ConsensusForecast(
            raw_p_up=probability.p_up_next_open if probability.valid else 0.50,
            p_up=probability.p_up_next_open if probability.valid else 0.50,
            p_down=1.0-(probability.p_up_next_open if probability.valid else 0.50),
            direction=directional_lean, strength=lean_confidence, note="legacy direction"
        )
    )
    overnight_p_up = consensus.p_up
    overnight_direction = consensus.direction
    overnight_direction_strength = consensus.strength
    overnight_direction_note = consensus.note
    if large_day.valid:
        reasons.append(
            f"large-day total/gap/session={large_day.total_return*100:+.2f}%/"
            f"{large_day.gap_return*100:+.2f}%/{large_day.session_return*100:+.2f}%; "
            f"final30/final60={large_day.final_30m_return*100:+.2f}%/{large_day.final_60m_return*100:+.2f}%"
        )
        reasons.append(large_day.note)
    reasons.append(
        f"OVERNIGHT direction={overnight_direction.upper()} P(up)={overnight_p_up*100:.1f}% "
        f"({overnight_direction_note})"
    )

    # Reference point is the most recent *completed* daily close, not the live
    # intraday spot, so this reads as "how far has drifted since the last settle"
    # both during the day and, more importantly, when checked after the close /
    # before the next open — which is when this number actually matters for a
    # position being held overnight.
    completed_daily = _completed_daily_before_live_day(daily, day)
    last_completed_close = float(completed_daily["Close"].iloc[-1]) if not completed_daily.empty else spot
    futures_symbol, futures_move_pct, futures_note = get_futures_overnight_move(
        symbol, last_completed_close, config
    )
    if futures_move_pct is not None:
        reasons.append(f"futures context ({futures_symbol}) move from last regular close={futures_move_pct*100:+.2f}%")

    event_risk = get_event_risk(symbol, config)
    if event_risk.severity != "LOW":
        reasons.append(f"event risk {event_risk.severity}: {event_risk.note}")

    # v6: event risk is a warning by default. It becomes a hard block only when explicitly requested.
    event_hard_block = (
        event_risk.block
        and (config.event_guard_mode == "block" or not config.allow_event_risk_alerts)
    )
    block = event_hard_block

    if bias == "neutral":
        lean_ok = (
            config.allow_neutral_directional_ideas
            and directional_lean in {"bullish", "bearish"}
            and lean_confidence >= int(config.neutral_min_lean_confidence * 100)
        )
        if not lean_ok and not config.allow_neutral_alerts:
            block = True
    elif confidence < config.min_market_confidence and not config.allow_neutral_alerts:
        block = True

    return MarketState(
        last_price=spot,
        bias=bias,
        confidence=confidence,
        directional_lean=directional_lean,
        lean_confidence=lean_confidence,
        reason="; ".join(reasons),
        support=support,
        resistance=resistance,
        atr14=atr14,
        historical_iv=historical_iv,
        daily_signal=daily_signal,
        m30_signal=m30_signal,
        m5_signal=m5_signal,
        ret_1d=ret_1d,
        ret_3d=ret_3d,
        ret_5d=ret_5d,
        day_move_atr=day_move_atr,
        avg_abs_gap=avg_abs_gap,
        pctl_abs_gap=pctl_abs_gap,
        vwap=vwap,
        event_risk=event_risk,
        probability=probability,
        vix_level=vix_level,
        vix_percentile=vix_pctile,
        vix_bucket=vix_bucket,
        futures_symbol=futures_symbol,
        futures_overnight_move_pct=futures_move_pct,
        futures_note=futures_note,
        large_day=large_day,
        overnight_p_up=overnight_p_up,
        overnight_direction=overnight_direction,
        overnight_direction_strength=overnight_direction_strength,
        overnight_direction_note=overnight_direction_note,
        overnight_consensus=consensus,
        block_new_alerts=block,
    )


# ---------------------------- Option math / spread filters ----------------------------

def year_fraction(expiration_date: dt.date, tz_name: str) -> float:
    now = now_local(tz_name)
    expiry_dt = dt.datetime.combine(expiration_date, dt.time(hour=13, minute=0), tzinfo=ZoneInfo(tz_name))
    seconds = max((expiry_dt - now).total_seconds(), 60)
    return seconds / (365.0 * 24 * 60 * 60)


def bs_delta(spot: float, strike: float, t: float, r: float, iv: float, option_type: str) -> float:
    iv = max(iv, 1e-6)
    t = max(t, 1e-6)
    if spot <= 0 or strike <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    return norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0


def bs_probability_above(spot: float, strike: float, t: float, r: float, iv: float) -> float:
    """Risk-neutral Black-Scholes probability that S_T is above strike (N(d2))."""
    iv = max(iv, 1e-6)
    t = max(t, 1e-6)
    if spot <= 0 or strike <= 0:
        return 0.50
    d2 = (math.log(spot / strike) + (r - 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    return float(norm.cdf(d2))


def bs_price(spot: float, strike: float, t: float, r: float, iv: float, option_type: str) -> float:
    """European Black-Scholes mark used only for scenario repricing.

    SPY/GLD options are American-style, so this is an approximation. For the user's
    close-to-next-open workflow it is used as a consistent scenario mark, not a fill quote.
    """
    if spot <= 0 or strike <= 0:
        return 0.0
    t = max(t, 1e-8)
    iv = max(iv, 1e-8)
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    disc = math.exp(-r * t)
    if option_type == "call":
        return float(spot * norm.cdf(d1) - strike * disc * norm.cdf(d2))
    return float(strike * disc * norm.cdf(-d2) - spot * norm.cdf(-d1))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if len(values) == 0 or len(weights) == 0:
        return 0.0
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    total = float(w.sum())
    if total <= 0:
        return float(np.quantile(v, q))
    cdf = np.cumsum(w) / total
    return float(np.interp(min(max(q, 0.0), 1.0), cdf, v))


def simulate_next_open_spread_pl(
    state: MarketState,
    spread_type: str,
    spot: float,
    short_strike: float,
    long_strike: float,
    credit: float,
    short_iv: float,
    long_iv: float,
    expiration: str,
    config: Config,
) -> Dict[str, float]:
    """Estimate the next-open P/L distribution from historical analog overnight returns.

    The historical analogs supply the underlying's close-to-next-open return scenarios.
    Each scenario then reprices both option legs at the next regular-session open. Two
    versions are calculated: unchanged IV and a conservative IV-up stress.
    """
    prob = state.probability
    if not prob.valid or not prob.weights or not prob.next_open:
        return {
            "profit_prob": 0.50,
            "profit_prob_stress": 0.50,
            "median_pl": 0.0,
            "p25_pl": 0.0,
            "p10_pl": 0.0,
            "loss25_prob": 0.50,
            "loss50_prob": 0.25,
        }

    w = np.asarray(prob.weights, dtype=float)
    returns = np.asarray(prob.next_open, dtype=float)
    if len(w) != len(returns) or len(w) == 0:
        return {
            "profit_prob": 0.50,
            "profit_prob_stress": 0.50,
            "median_pl": 0.0,
            "p25_pl": 0.0,
            "p10_pl": 0.0,
            "loss25_prob": 0.50,
            "loss50_prob": 0.25,
        }
    w = w / max(float(w.sum()), 1e-12)

    now = now_local(config.market_timezone)
    open_dt = next_regular_open(now, config.market_timezone)
    exp_date = dt.datetime.strptime(expiration, "%Y-%m-%d").date()
    expiry_dt = dt.datetime.combine(exp_date, dt.time(13, 0), tzinfo=ZoneInfo(config.market_timezone))
    t_open = max((expiry_dt - open_dt).total_seconds(), 60.0) / (365.0 * 24 * 60 * 60)
    option_type = "put" if spread_type == "bull_put" else "call"
    width = abs(short_strike - long_strike)
    max_loss_dollars = max(width - credit, 0.0) * 100.0

    def scenario_pl(iv_mult: float) -> np.ndarray:
        out: List[float] = []
        for r_open in returns:
            s = max(0.01, spot * (1.0 + float(r_open)))
            short_val = bs_price(s, short_strike, t_open, config.risk_free_rate, short_iv * iv_mult, option_type)
            long_val = bs_price(s, long_strike, t_open, config.risk_free_rate, long_iv * iv_mult, option_type)
            close_cost = min(max(short_val - long_val, 0.0), width)
            out.append((credit - close_cost) * 100.0)
        return np.asarray(out, dtype=float)

    pl = scenario_pl(1.0)
    pl_stress = scenario_pl(max(config.morning_iv_stress_mult, 0.01))
    profit_prob = _weighted_mean((pl > 0.0).astype(float), w)
    profit_prob_stress = _weighted_mean((pl_stress > 0.0).astype(float), w)
    loss25_prob = _weighted_mean((pl < -0.25 * max_loss_dollars).astype(float), w) if max_loss_dollars > 0 else 0.0
    loss50_prob = _weighted_mean((pl < -0.50 * max_loss_dollars).astype(float), w) if max_loss_dollars > 0 else 0.0

    return {
        "profit_prob": profit_prob,
        "profit_prob_stress": profit_prob_stress,
        "median_pl": _weighted_quantile(pl, w, 0.50),
        "p25_pl": _weighted_quantile(pl, w, 0.25),
        "p10_pl": _weighted_quantile(pl, w, 0.10),
        "loss25_prob": loss25_prob,
        "loss50_prob": loss50_prob,
    }


def consensus_probability(option_pop: float, historical_pop: float, prob_valid: bool, config: Config) -> float:
    if not prob_valid:
        return option_pop
    ow = max(config.option_pop_weight, 0.0)
    hw = max(config.historical_pop_weight, 0.0)
    total = ow + hw
    if total <= 0:
        return option_pop
    return float((ow * option_pop + hw * historical_pop) / total)


def estimate_iv(row: pd.Series, config: Config, historical_iv: float) -> float:
    iv = row.get("impliedVolatility", np.nan)
    if pd.isna(iv) or iv <= 0:
        return max(historical_iv, config.iv_floor)
    return float(max(iv, config.iv_floor))


def clean_option_chain(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["contractSymbol", "strike", "bid", "ask", "volume", "openInterest", "impliedVolatility", "lastPrice"]
    out = df.copy()
    for col in needed:
        if col not in out.columns:
            out[col] = np.nan
    out["mid"] = np.where(
        (out["bid"].fillna(0) > 0) & (out["ask"].fillna(0) > 0),
        (out["bid"].fillna(0) + out["ask"].fillna(0)) / 2.0,
        out["lastPrice"].fillna(0),
    )
    return out


def relative_bid_ask_width(bid: float, ask: float, mid: float) -> float:
    if mid <= 0 or ask < bid:
        return float("inf")
    return max(0.0, (ask - bid) / mid)


def get_candidate_expirations(ticker: yf.Ticker, config: Config) -> List[Tuple[str, int]]:
    today = now_local(config.market_timezone).date()
    results: List[Tuple[str, int]] = []
    for exp in ticker.options:
        exp_date = dt.datetime.strptime(exp, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if config.target_dte_min <= dte <= config.target_dte_max:
            results.append((exp, dte))
    return results


def candidate_grade(score: float, confidence: int, event_severity: str) -> str:
    penalty = 10 if event_severity == "MEDIUM" else 25 if event_severity == "HIGH" else 0
    adjusted = score + confidence * 0.25 - penalty
    if adjusted >= 70:
        return "A"
    if adjusted >= 50:
        return "B"
    return "C"


def sr_location_metrics(
    state: MarketState, spread_type: str, short_strike: float, config: Config
) -> Tuple[float, float, float, bool]:
    """Return (clearance, required, shortfall_atr, passes_normal_sr_rule).

    Positive clearance means the short strike sits beyond support/resistance in the
    safer direction. A negative value means the strike is still inside the recent
    S/R area.
    """
    if state.atr14 <= 0:
        return 0.0, 0.0, 0.0, True
    required = state.atr14 * config.min_sr_buffer_atr
    clearance = (
        state.support - short_strike
        if spread_type == "bull_put"
        else short_strike - state.resistance
    )
    shortfall = max(0.0, required - clearance)
    shortfall_atr = shortfall / max(state.atr14, 1e-9)
    return clearance, required, shortfall_atr, clearance >= required


def sr_override_qualifies(
    *,
    state: MarketState,
    overnight_eligibility: str,
    morning_conservative: float,
    open_survival_prob: float,
    consensus_pop: float,
    be_gap_buffer_mult: float,
    sr_shortfall_atr: float,
    config: Config,
) -> bool:
    """Whether exceptionally strong overnight evidence may override an S/R miss.

    This is intentionally demanding: S/R is not ignored. It is only demoted from a
    hard veto when the close-to-open statistics are unusually strong and the strike
    is not absurdly far inside the technical S/R zone.
    """
    return (
        overnight_eligibility == "PASS"
        and state.event_risk.severity != "HIGH"
        and morning_conservative >= config.sr_override_min_morning_profit_prob
        and open_survival_prob >= config.sr_override_min_open_survival
        and consensus_pop >= config.sr_override_min_consensus_pop
        and be_gap_buffer_mult >= config.sr_override_min_be_gap_mult
        and sr_shortfall_atr <= config.sr_override_max_shortfall_atr
    )


def make_spreads(
    chain: pd.DataFrame,
    option_type: str,
    spot: float,
    expiration: str,
    dte: int,
    config: Config,
    state: MarketState,
) -> List[SpreadCandidate]:
    rows = clean_option_chain(chain)
    rows = rows[
        (rows["openInterest"].fillna(0) >= config.min_open_interest)
        & (rows["volume"].fillna(0) >= config.min_volume)
    ]
    if rows.empty:
        return []

    exp_date = dt.datetime.strptime(expiration, "%Y-%m-%d").date()
    t = year_fraction(exp_date, config.market_timezone)
    rows = rows.sort_values("strike").reset_index(drop=True)
    rows["delta"] = [
        bs_delta(
            spot,
            float(r["strike"]),
            t,
            config.risk_free_rate,
            estimate_iv(r, config, state.historical_iv),
            option_type,
        )
        for _, r in rows.iterrows()
    ]
    rows["abs_delta"] = rows["delta"].abs()

    # Scheduled-event ideas are intentionally pushed farther OTM by tightening max delta.
    active_delta_max = config.short_delta_max
    if state.event_risk.severity in {"MEDIUM", "HIGH"} and config.allow_event_risk_alerts:
        active_delta_max = min(active_delta_max, config.event_short_delta_max)

    if option_type == "put":
        spread_type = "bull_put"
        shorts = rows[
            (rows["abs_delta"] >= config.short_delta_min)
            & (rows["abs_delta"] <= active_delta_max)
            & (rows["strike"] < spot)
        ]
    else:
        spread_type = "bear_call"
        shorts = rows[
            (rows["abs_delta"] >= config.short_delta_min)
            & (rows["abs_delta"] <= active_delta_max)
            & (rows["strike"] > spot)
        ]

    candidates: List[SpreadCandidate] = []
    gap_dollars = spot * state.pctl_abs_gap

    for _, short in shorts.iterrows():
        short_strike = float(short["strike"])
        sbid, sask, smid = float(short["bid"]), float(short["ask"]), float(short["mid"])
        if relative_bid_ask_width(sbid, sask, smid) > config.max_relative_bid_ask_width:
            continue

        # v11.2: measure S/R location now, but delay the hard decision until the
        # next-morning probability model has been evaluated. This allows a genuinely
        # exceptional close-to-open setup to receive a weighted S/R warning instead
        # of being discarded before its overnight evidence is even calculated.
        sr_clearance, sr_required, sr_shortfall_atr, sr_pass = sr_location_metrics(
            state, spread_type, short_strike, config
        )

        # Historical overnight-gap guard. Particularly relevant for 1-DTE holds.
        distance = (spot - short_strike) if option_type == "put" else (short_strike - spot)
        if config.require_overnight_gap_buffer and gap_dollars > 0:
            if distance < gap_dollars * config.min_overnight_gap_buffer_mult:
                continue

        if option_type == "put":
            longs = rows[
                (rows["strike"] < short_strike)
                & ((short_strike - rows["strike"]) <= config.max_spread_width)
            ]
        else:
            longs = rows[
                (rows["strike"] > short_strike)
                & ((rows["strike"] - short_strike) <= config.max_spread_width)
            ]

        for _, long in longs.iterrows():
            long_strike = float(long["strike"])
            lbid, lask, lmid = float(long["bid"]), float(long["ask"]), float(long["mid"])
            if relative_bid_ask_width(lbid, lask, lmid) > config.max_relative_bid_ask_width:
                continue

            # Conservative executable-credit estimate: short bid minus long ask.
            # This is less optimistic than assuming both legs fill at midpoint.
            credit = sbid - lask
            width = abs(short_strike - long_strike)
            if width <= 0 or credit < config.min_credit or credit >= width:
                continue

            max_loss = width - credit
            ror = credit / max_loss
            if ror < config.min_ror:
                continue

            short_delta = abs(float(short["delta"]))
            long_delta = abs(float(long["delta"]))

            breakeven = short_strike - credit if spread_type == "bull_put" else short_strike + credit
            short_iv = estimate_iv(short, config, state.historical_iv)
            long_iv = estimate_iv(long, config, state.historical_iv)
            p_above_be = bs_probability_above(spot, breakeven, t, config.risk_free_rate, short_iv)
            option_pop = p_above_be if spread_type == "bull_put" else 1.0 - p_above_be
            horizon_sessions = 1 if dte <= 1 else 2
            be_return = breakeven / spot - 1.0
            historical_pop, no_breach_prob = candidate_historical_probability(
                state.probability, spread_type, be_return, horizon_sessions
            )
            consensus_pop = consensus_probability(option_pop, historical_pop, state.probability.valid, config)
            probability_disagreement = abs(option_pop - historical_pop) if state.probability.valid else 0.0
            expected_move_pct = short_iv * math.sqrt(t)
            breakeven_distance_pct = abs(spot - breakeven) / max(spot, 1e-9)
            be_gap_buffer_mult = breakeven_distance_pct / max(state.pctl_abs_gap, 1e-9)
            be_expected_move_buffer_mult = breakeven_distance_pct / max(expected_move_pct, 1e-9)
            open_survival_prob = historical_open_survival_probability(
                state.probability, spread_type, be_return
            )

            # v11: this is the probability model that is most directly aligned with the
            # intended workflow: enter near the close and buy the spread back next morning.
            morning = simulate_next_open_spread_pl(
                state=state,
                spread_type=spread_type,
                spot=spot,
                short_strike=short_strike,
                long_strike=long_strike,
                credit=credit,
                short_iv=short_iv,
                long_iv=long_iv,
                expiration=expiration,
                config=config,
            )
            morning_profit_prob = float(morning["profit_prob"])
            morning_profit_prob_stress = float(morning["profit_prob_stress"])
            morning_conservative = min(morning_profit_prob, morning_profit_prob_stress)

            overnight_eligibility, overnight_reasons = evaluate_overnight_eligibility(
                state, spread_type, consensus_pop, no_breach_prob, open_survival_prob,
                be_gap_buffer_mult, be_expected_move_buffer_mult, config,
                breakeven_distance_pct=breakeven_distance_pct,
            )
            morning_reason = (
                f"next-open spread profit {morning_profit_prob*100:.0f}% "
                f"(IV stress {morning_profit_prob_stress*100:.0f}%)"
            )
            if morning_conservative < config.morning_caution_profit_prob:
                overnight_eligibility = "FAIL"
                overnight_reasons = f"{overnight_reasons}; {morning_reason}"
            elif morning_conservative < config.morning_pass_profit_prob and overnight_eligibility == "PASS":
                overnight_eligibility = "CAUTION"
                overnight_reasons = f"{overnight_reasons}; {morning_reason}"
            elif morning_reason not in overnight_reasons:
                overnight_reasons = f"{overnight_reasons}; {morning_reason}"

            # v11.2 weighted S/R handling. Strict mode preserves the old veto.
            # Weighted mode only permits a miss when the overnight/morning evidence is
            # exceptional. Warn mode never hard-rejects solely for S/R location.
            sr_override_used = False
            sr_mode = (config.sr_filter_mode or "weighted").strip().lower()
            if config.require_strike_beyond_sr and not sr_pass:
                if sr_mode == "strict":
                    continue
                if sr_mode == "weighted":
                    sr_override_used = sr_override_qualifies(
                        state=state,
                        overnight_eligibility=overnight_eligibility,
                        morning_conservative=morning_conservative,
                        open_survival_prob=open_survival_prob,
                        consensus_pop=consensus_pop,
                        be_gap_buffer_mult=be_gap_buffer_mult,
                        sr_shortfall_atr=sr_shortfall_atr,
                        config=config,
                    )
                    if not sr_override_used:
                        continue
                elif sr_mode == "warn":
                    sr_override_used = True
                else:
                    # Unknown values fail safe as strict mode.
                    continue

                sr_warning = (
                    f"S/R WARNING: clearance {sr_clearance:.2f} vs required {sr_required:.2f} "
                    f"({sr_shortfall_atr:.2f} ATR shortfall); overnight evidence override applied"
                )
                overnight_reasons = f"{overnight_reasons}; {sr_warning}"

            if state.event_risk.severity == "HIGH" or no_breach_prob < 0.65 or consensus_pop < 0.68 or morning_conservative < config.morning_caution_profit_prob:
                overnight_risk = "HIGH"
            elif state.event_risk.severity == "MEDIUM" or no_breach_prob < 0.75 or probability_disagreement >= config.probability_disagreement_threshold:
                overnight_risk = "MEDIUM"
            else:
                overnight_risk = "LOW"

            if config.require_probability_filter and consensus_pop < config.min_consensus_pop:
                continue
            if config.require_morning_profit_filter and morning_conservative < config.min_morning_profit_prob:
                continue
            if config.block_overnight_fail_alerts and overnight_eligibility == "FAIL":
                continue

            ev = credit * consensus_pop - max_loss * (1.0 - consensus_pop)

            gap_mult = distance / gap_dollars if gap_dollars > 0 else 0.0
            liquidity = max(0.0, 1.0 - relative_bid_ask_width(sbid, sask, smid))

            # A failed S/R rule no longer gets an automatic zero-or-reject outcome in
            # weighted mode. Instead, the size of the miss becomes a bounded penalty.
            sr_penalty = 0.0
            if sr_override_used and not sr_pass:
                sr_penalty = min(
                    config.sr_penalty_cap,
                    sr_shortfall_atr * config.sr_penalty_per_atr_shortfall,
                )

            # v11 quality score is deliberately close-to-open focused. Morning P/L
            # probability and next-open survival dominate; expiration POP is secondary.
            score = (
                state.confidence * 0.15
                + morning_conservative * 35
                + open_survival_prob * 15
                + consensus_pop * 12
                + no_breach_prob * 5
                + min(ror / 0.20, 1.5) * 10
                + min(max(sr_clearance, 0.0) / max(state.atr14, 0.01), 1.5) * 6
                + min(gap_mult, 2.0) * 5
                + liquidity * 5
                + max(min(ev, 0.25), -0.25) * 2
            )
            if probability_disagreement >= config.probability_disagreement_threshold:
                score -= config.probability_disagreement_penalty
            if sr_penalty > 0:
                score -= sr_penalty
            if state.event_risk.severity == "MEDIUM":
                score -= 10
            elif state.event_risk.severity == "HIGH":
                score -= 25

            grade = candidate_grade(score, state.confidence, state.event_risk.severity)
            # An S/R override can be a valid overnight idea, but it should never look
            # identical to a fully location-confirmed A setup. Cap it at B.
            if sr_override_used and grade == "A":
                grade = "B"
            rationale = (
                f"Signal {state.bias} strength {state.confidence}/100; lean {state.directional_lean} {state.lean_confidence}/100. "
                f"Morning P(profit)={morning_profit_prob*100:.0f}% (IV-stress={morning_profit_prob_stress*100:.0f}%); "
                f"POP option={option_pop*100:.0f}%, analog={historical_pop*100:.0f}%, consensus={consensus_pop*100:.0f}%; "
                f"historical no-breach={no_breach_prob*100:.0f}% over ~{horizon_sessions} session(s). "
                f"Short strike is {distance:.2f} from spot; S/R clearance={sr_clearance:.2f} "
                f"(required {sr_required:.2f}, {'OVERRIDE' if sr_override_used else 'PASS'}); "
                f"gap cushion={gap_mult:.2f}x the recent {int(config.overnight_gap_percentile*100)}th-percentile overnight gap. "
                f"Expected move through expiry ~{expected_move_pct*100:.2f}%. Overnight risk: {overnight_risk}. "
                f"Overnight eligibility: {overnight_eligibility} ({overnight_reasons}). "
                f"Event risk: {state.event_risk.severity}."
            )

            candidates.append(
                SpreadCandidate(
                    spread_type=spread_type,
                    expiration=expiration,
                    dte=dte,
                    short_strike=short_strike,
                    long_strike=long_strike,
                    width=width,
                    short_delta=short_delta,
                    long_delta=long_delta,
                    credit=credit,
                    max_loss=max_loss,
                    option_pop=option_pop,
                    historical_pop=historical_pop,
                    consensus_pop=consensus_pop,
                    no_breach_prob=no_breach_prob,
                    expected_move_pct=expected_move_pct,
                    probability_disagreement=probability_disagreement,
                    breakeven=breakeven,
                    overnight_risk=overnight_risk,
                    overnight_eligibility=overnight_eligibility,
                    overnight_reasons=overnight_reasons,
                    open_survival_prob=open_survival_prob,
                    breakeven_distance_pct=breakeven_distance_pct,
                    be_gap_buffer_mult=be_gap_buffer_mult,
                    be_expected_move_buffer_mult=be_expected_move_buffer_mult,
                    ror=ror,
                    ev_heuristic=ev,
                    score=score,
                    grade=grade,
                    rationale=rationale,
                    short_iv=short_iv,
                    long_iv=long_iv,
                    morning_profit_prob=morning_profit_prob,
                    morning_profit_prob_stress=morning_profit_prob_stress,
                    morning_median_pl=float(morning["median_pl"]),
                    morning_p25_pl=float(morning["p25_pl"]),
                    morning_p10_pl=float(morning["p10_pl"]),
                    morning_loss25_prob=float(morning["loss25_prob"]),
                    morning_loss50_prob=float(morning["loss50_prob"]),
                    sr_clearance=sr_clearance,
                    sr_required=sr_required,
                    sr_shortfall_atr=sr_shortfall_atr,
                    sr_override_used=sr_override_used,
                    sr_penalty=sr_penalty,
                )
            )

    return candidates


def rank_candidates(candidates: Iterable[SpreadCandidate], state: MarketState, config: Config) -> List[SpreadCandidate]:
    pool = list(candidates)

    # v13: ONE final next-open consensus chooses direction. If the consensus is
    # neutral, the bot does not force a directional credit spread from technicals.
    preferred = state.overnight_direction if config.spy_overnight_model_enabled else state.bias
    if preferred == "bullish":
        filtered = [c for c in pool if c.spread_type == "bull_put"]
    elif preferred == "bearish":
        filtered = [c for c in pool if c.spread_type == "bear_call"]
    else:
        filtered = []

    # Normal scans require A/B. During scheduled-event risk, v6 can surface a C-grade
    # idea as explicitly speculative if it still clears a minimum raw score.
    if state.event_risk.severity in {"MEDIUM", "HIGH"} and config.allow_event_risk_alerts:
        filtered = [c for c in filtered if c.grade in {"A", "B"} or c.score >= config.min_event_candidate_score]
    elif state.bias == "neutral" and config.allow_neutral_directional_ideas:
        filtered = [c for c in filtered if c.grade in {"A", "B"} or c.score >= config.min_neutral_idea_score]
    else:
        filtered = [c for c in filtered if c.grade in {"A", "B"}]

    return sorted(filtered, key=lambda c: (c.score, c.ror, c.credit), reverse=True)



def favored_spread_type(state: MarketState, config: Config) -> Optional[str]:
    """Return only the spread side favored by the v13 final consensus."""
    preferred = state.overnight_direction if config.spy_overnight_model_enabled else state.bias
    if preferred == "bullish":
        return "bull_put"
    if preferred == "bearish":
        return "bear_call"
    return None


def _rank_grade_passes(candidate: SpreadCandidate, state: MarketState, config: Config) -> bool:
    if state.event_risk.severity in {"MEDIUM", "HIGH"} and config.allow_event_risk_alerts:
        return candidate.grade in {"A", "B"} or candidate.score >= config.min_event_candidate_score
    if state.bias == "neutral" and config.allow_neutral_directional_ideas:
        return candidate.grade in {"A", "B"} or candidate.score >= config.min_neutral_idea_score
    return candidate.grade in {"A", "B"}


def _near_miss_failures(candidate: SpreadCandidate, state: MarketState, config: Config) -> Tuple[List[str], float]:
    """Explain which *normal* filters a relaxed candidate misses.

    miss_distance is a dimensionless shortfall score; lower means closer to qualifying.
    """
    failed: List[str] = []
    miss = 0.0

    if candidate.credit < config.min_credit:
        shortfall = (config.min_credit - candidate.credit) / max(config.min_credit, 1e-9)
        failed.append(f"credit ${candidate.credit:.2f} < ${config.min_credit:.2f}")
        miss += shortfall

    if candidate.ror < config.min_ror:
        shortfall = (config.min_ror - candidate.ror) / max(config.min_ror, 1e-9)
        failed.append(f"RoR {candidate.ror*100:.1f}% < {config.min_ror*100:.1f}%")
        miss += shortfall

    if config.require_strike_beyond_sr and state.atr14 > 0:
        clearance, required, shortfall_atr, sr_pass = sr_location_metrics(
            state, candidate.spread_type, candidate.short_strike, config
        )
        if not sr_pass:
            mode = (config.sr_filter_mode or "weighted").strip().lower()
            morning_conservative_for_sr = min(
                candidate.morning_profit_prob, candidate.morning_profit_prob_stress
            )
            override_ok = mode == "warn" or (
                mode == "weighted"
                and sr_override_qualifies(
                    state=state,
                    overnight_eligibility=candidate.overnight_eligibility,
                    morning_conservative=morning_conservative_for_sr,
                    open_survival_prob=candidate.open_survival_prob,
                    consensus_pop=candidate.consensus_pop,
                    be_gap_buffer_mult=candidate.be_gap_buffer_mult,
                    sr_shortfall_atr=shortfall_atr,
                    config=config,
                )
            )
            if not override_ok:
                failed.append(
                    f"S/R clearance {clearance:.2f} < required {required:.2f} "
                    f"({shortfall_atr:.2f} ATR shortfall; override criteria not met)"
                )
                # Scale the diagnostic miss by ATR rather than dividing by the tiny
                # required buffer. This prevents S/R from numerically dwarfing every
                # other near-miss criterion.
                miss += min(shortfall_atr, 3.0)

    if config.require_overnight_gap_buffer and state.pctl_abs_gap > 0:
        short_distance_pct = (
            (state.last_price - candidate.short_strike) / max(state.last_price, 1e-9)
            if candidate.spread_type == "bull_put"
            else (candidate.short_strike - state.last_price) / max(state.last_price, 1e-9)
        )
        gap_mult = short_distance_pct / max(state.pctl_abs_gap, 1e-9)
        if gap_mult < config.min_overnight_gap_buffer_mult:
            failed.append(
                f"short-strike gap cushion {gap_mult:.2f}x < {config.min_overnight_gap_buffer_mult:.2f}x"
            )
            miss += (config.min_overnight_gap_buffer_mult - gap_mult) / max(config.min_overnight_gap_buffer_mult, 1e-9)

    if config.require_probability_filter and candidate.consensus_pop < config.min_consensus_pop:
        failed.append(
            f"consensus POP {candidate.consensus_pop*100:.1f}% < {config.min_consensus_pop*100:.1f}%"
        )
        miss += (config.min_consensus_pop - candidate.consensus_pop) / max(config.min_consensus_pop, 1e-9)

    morning_conservative = min(candidate.morning_profit_prob, candidate.morning_profit_prob_stress)
    if config.require_morning_profit_filter and morning_conservative < config.min_morning_profit_prob:
        failed.append(
            f"next-open P(profit) {morning_conservative*100:.1f}% < {config.min_morning_profit_prob*100:.1f}%"
        )
        miss += (config.min_morning_profit_prob - morning_conservative) / max(config.min_morning_profit_prob, 1e-9)

    if config.block_overnight_fail_alerts and candidate.overnight_eligibility == "FAIL":
        failed.append("overnight eligibility = FAIL")
        miss += 1.0

    if not _rank_grade_passes(candidate, state, config):
        failed.append(f"grade/score gate not met (grade {candidate.grade}, score {candidate.score:.1f})")
        miss += 0.50

    return failed, miss


def find_near_miss_candidates(
    chain_cache: List[Tuple[str, int, pd.DataFrame, pd.DataFrame]],
    state: MarketState,
    config: Config,
) -> List[NearMissCandidate]:
    """Find close, direction-aligned candidates without weakening liquidity/delta rules.

    We rerun spread construction with only selected strategy gates relaxed. This means a
    displayed near miss still uses the configured OI, volume, bid/ask, width, and delta
    constraints, but may narrowly miss credit, RoR, S/R, gap, POP, morning-profit, or
    grade thresholds.
    """
    if not config.show_near_miss_candidates:
        return []
    favored = favored_spread_type(state, config)
    if favored is None:
        return []

    relaxed = dataclasses.replace(
        config,
        min_credit=max(0.01, min(config.min_credit, config.near_miss_min_credit)),
        min_ror=max(0.0, min(config.min_ror, config.near_miss_min_ror)),
        require_strike_beyond_sr=False,
        require_overnight_gap_buffer=False,
        require_probability_filter=False,
        require_morning_profit_filter=False,
        block_overnight_fail_alerts=False,
    )

    pool: List[SpreadCandidate] = []
    for expiration, dte, puts, calls in chain_cache:
        pool.extend(make_spreads(puts, "put", state.last_price, expiration, dte, relaxed, state))
        pool.extend(make_spreads(calls, "call", state.last_price, expiration, dte, relaxed, state))

    seen = set()
    evaluated: List[NearMissCandidate] = []
    for c in pool:
        if c.spread_type != favored:
            continue
        if c.key("_") in seen:
            continue
        seen.add(c.key("_"))

        morning_conservative = min(c.morning_profit_prob, c.morning_profit_prob_stress)
        if c.credit < config.near_miss_min_credit:
            continue
        if c.ror < config.near_miss_min_ror:
            continue
        if c.consensus_pop < config.near_miss_min_consensus_pop:
            continue
        if morning_conservative < config.near_miss_min_morning_profit_prob:
            continue

        failures, distance = _near_miss_failures(c, state, config)
        if not failures:
            # This can occur when the strict pool was empty for a transient rank/cooldown
            # reason. It is still useful, but label it as a zero-filter near miss.
            failures = ["no strategy filter miss detected; likely ranking/availability edge case"]
        if len(failures) > config.near_miss_max_failed_filters:
            continue
        evaluated.append(NearMissCandidate(candidate=c, failed_filters=failures, miss_distance=distance))

    evaluated.sort(
        key=lambda x: (
            len(x.failed_filters),
            x.miss_distance,
            -min(x.candidate.morning_profit_prob, x.candidate.morning_profit_prob_stress),
            -x.candidate.score,
        )
    )
    return evaluated[: max(0, config.max_near_miss_candidates)]


def print_near_miss_candidates(symbol: str, state: MarketState, config: Config, near_misses: List[NearMissCandidate]) -> None:
    if not near_misses:
        return
    print("NEAR-MISS / FAVORED-DIRECTION IDEAS")
    print("  These are informational only: they did NOT pass every normal alert filter.")
    print(f"  Favored direction  : {(favored_spread_type(state, config) or 'n/a').replace('_', ' ').upper()}")
    print(divider())
    for idx, item in enumerate(near_misses, 1):
        c = item.candidate
        morning_conservative = min(c.morning_profit_prob, c.morning_profit_prob_stress)
        print(
            f"  #{idx} {symbol} {c.title().upper()}  Sell {c.short_strike:.2f} / Buy {c.long_strike:.2f}  "
            f"({c.expiration}, {c.dte} DTE)"
        )
        print(
            f"     Credit ${c.credit:.2f} | RoR {c.ror*100:.1f}% | Delta {c.short_delta:.3f} | "
            f"Grade {c.grade} | Overnight {c.overnight_eligibility}"
        )
        print(
            f"     Morning P(profit) {morning_conservative*100:.1f}% "
            f"(base {c.morning_profit_prob*100:.1f}%, IV-stress {c.morning_profit_prob_stress*100:.1f}%) | "
            f"Spread POP blend {c.consensus_pop*100:.1f}%"
        )
        print(
            f"     Open survival {c.open_survival_prob*100:.1f}% | BE {c.breakeven:.2f} | "
            f"BE/p90 gap {c.be_gap_buffer_mult:.2f}x"
        )
        print("     Missed filter(s):")
        for reason in item.failed_filters:
            print(f"       - {reason}")
        print(f"     Near-miss distance: {item.miss_distance:.2f} (lower = closer to qualifying)")
        if idx != len(near_misses):
            print()
    print(divider())


# ---------------------------- v11 persistent overnight tracking ----------------------------

def _tracker_path(config: Config) -> str:
    path = config.track_state_file.strip() or "overnight_tracked_ideas.json"
    return os.path.abspath(path)


def load_tracker(config: Config) -> Dict[str, Any]:
    path = _tracker_path(config)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if not isinstance(data.get("ideas"), dict):
                data["ideas"] = {}
            if not isinstance(data.get("markets"), dict):
                data["markets"] = {}
            return data
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[warn] could not read overnight tracker {path}: {exc}", file=sys.stderr)
    return {"ideas": {}, "markets": {}}


def save_tracker(config: Config, tracker: Dict[str, Any]) -> None:
    path = _tracker_path(config)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(tracker, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception as exc:
        print(f"[warn] could not save overnight tracker {path}: {exc}", file=sys.stderr)



def _forecast_log_path(config: Config) -> str:
    path = config.forecast_log_file.strip() or "overnight_forecast_history.json"
    return os.path.abspath(path)


def load_forecast_log(config: Config) -> Dict[str, Any]:
    path = _forecast_log_path(config)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("forecasts"), dict):
            return data
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[warn] could not read forecast log {path}: {exc}", file=sys.stderr)
    return {"forecasts": {}}


def save_forecast_log(config: Config, log: Dict[str, Any]) -> None:
    path = _forecast_log_path(config)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception as exc:
        print(f"[warn] could not save forecast log {path}: {exc}", file=sys.stderr)


def _market_watch_key(symbol: str, target_open: dt.datetime) -> str:
    return f"{symbol.upper()}|{target_open.date().isoformat()}"


def _record_forecast_log_entry(watch: Dict[str, Any], config: Config) -> None:
    symbol = str(watch.get("symbol", "")).upper()
    target_open = str(watch.get("target_open", ""))
    if not symbol or not target_open:
        return
    key = f"{symbol}|{target_open[:10]}"
    log = load_forecast_log(config)
    forecasts = log.setdefault("forecasts", {})
    old = forecasts.get(key, {}) if isinstance(forecasts.get(key), dict) else {}
    forecasts[key] = {
        "symbol": symbol,
        "reference_price": float(watch.get("reference_price", 0.0) or 0.0),
        "reference_time": str(watch.get("reference_time", "")),
        "target_open": target_open,
        "raw_p_up": float(watch.get("entry_p_up", 0.50) or 0.50),
        "forecast_direction": str(watch.get("forecast_direction", "neutral")),
        "technical_alignment": int(watch.get("technical_alignment", 0) or 0),
        "event_severity": str(watch.get("event_severity", "LOW")),
        "model_version": str(watch.get("model_version", "v13")),
        "resolved": bool(old.get("resolved", False)),
        "actual_open": old.get("actual_open"),
        "actual_return": old.get("actual_return"),
        "outcome_up": old.get("outcome_up"),
        "resolved_at": old.get("resolved_at", ""),
    }
    save_forecast_log(config, log)


def resolve_forecast_log(config: Config) -> None:
    """Resolve prior close->next-open forecasts once the target regular session has opened."""
    log = load_forecast_log(config)
    forecasts = log.get("forecasts", {})
    if not isinstance(forecasts, dict) or not forecasts:
        return
    now = now_local(config.market_timezone)
    changed = False
    by_symbol: Dict[str, pd.DataFrame] = {}
    for key, rec in list(forecasts.items()):
        if not isinstance(rec, dict) or rec.get("resolved"):
            continue
        try:
            target = dt.datetime.fromisoformat(str(rec.get("target_open")))
        except Exception:
            continue
        if now < target + dt.timedelta(minutes=1):
            continue
        symbol = str(rec.get("symbol", "")).upper()
        ref = float(rec.get("reference_price", 0.0) or 0.0)
        if not symbol or ref <= 0:
            continue
        try:
            if symbol not in by_symbol:
                frame = yf.Ticker(symbol).history(period="30d", interval="1d", auto_adjust=False)
                frame = _normalize_history_index(frame.dropna(subset=["Open"]), config.market_timezone)
                by_symbol[symbol] = frame
            frame = by_symbol[symbol]
            rows = frame[frame.index.date == target.date()]
            if rows.empty:
                continue
            actual_open = float(rows["Open"].iloc[0])
            rec["actual_open"] = actual_open
            rec["actual_return"] = actual_open / ref - 1.0
            rec["outcome_up"] = bool(actual_open > ref)
            rec["resolved"] = True
            rec["resolved_at"] = now.isoformat()
            changed = True
        except Exception as exc:
            print(f"[warn] forecast resolution failed for {symbol}: {exc}", file=sys.stderr)
    if changed:
        save_forecast_log(config, log)


def calibration_for_probability(symbol: str, raw_p: float, config: Config) -> Tuple[Optional[float], int, Optional[float]]:
    """Empirical local calibration from this bot's own resolved forecasts.

    Uses a probability neighborhood, not a formal isotonic/logistic calibrator. It becomes
    informative only after the user has accumulated enough resolved forecasts.
    Returns (empirical P(up), n, Brier score for the neighborhood).
    """
    log = load_forecast_log(config)
    forecasts = log.get("forecasts", {})
    half = max(config.calibration_bin_width, 0.02) / 2.0
    rows = []
    if isinstance(forecasts, dict):
        for rec in forecasts.values():
            if not isinstance(rec, dict) or not rec.get("resolved"):
                continue
            if str(rec.get("symbol", "")).upper() != symbol.upper():
                continue
            if str(rec.get("model_version", "")) != "v13":
                continue
            p = float(rec.get("raw_p_up", 0.50) or 0.50)
            if abs(p - raw_p) <= half and rec.get("outcome_up") is not None:
                rows.append((p, 1.0 if bool(rec.get("outcome_up")) else 0.0))
    n = len(rows)
    if n < config.calibration_min_resolved:
        return None, n, None
    empirical = float(np.mean([y for _, y in rows]))
    brier = float(np.mean([(p - y) ** 2 for p, y in rows]))
    return empirical, n, brier


def _dynamic_market_p_up(watch: Dict[str, Any], proxy_price: float, now: dt.datetime) -> float:
    """Heuristic live estimate that next open finishes above the frozen reference price.

    The original historical analog forecast is blended with what the market has actually
    done overnight. Remaining uncertainty shrinks with sqrt(time-to-open). This is a
    monitoring heuristic and is deliberately labeled as such in console/Discord output.
    """
    entry = float(watch.get("reference_price", 0.0) or 0.0)
    original = float(watch.get("entry_p_up", 0.50) or 0.50)
    p90 = float(watch.get("entry_p90_gap", 0.0) or 0.0)
    if entry <= 0 or proxy_price <= 0 or p90 <= 0:
        return original
    try:
        ref_dt = dt.datetime.fromisoformat(str(watch.get("reference_time")))
        open_dt = dt.datetime.fromisoformat(str(watch.get("target_open")))
    except Exception:
        return original
    total = max((open_dt - ref_dt).total_seconds(), 1.0)
    remaining = max((open_dt - now).total_seconds(), 0.0)
    frac_remaining = min(max(remaining / total, 0.02), 1.0)
    sigma_full = max(p90 / 1.6448536269514722, 1e-5)
    sigma_remaining = max(sigma_full * math.sqrt(frac_remaining), 1e-5)
    current_r = proxy_price / entry - 1.0
    live_only = float(norm.cdf(current_r / sigma_remaining))
    elapsed = 1.0 - frac_remaining
    live_weight = 0.35 + 0.65 * elapsed
    return min(max(live_weight * live_only + (1.0 - live_weight) * original, 0.0), 1.0)


def _market_thesis_status(watch: Dict[str, Any], updated_p_up: float, live_move: float, config: Config) -> Tuple[str, List[str]]:
    direction = str(watch.get("forecast_direction", "neutral"))
    p90 = float(watch.get("entry_p90_gap", 0.0) or 0.0)
    favorable = updated_p_up if direction == "bullish" else (1.0 - updated_p_up) if direction == "bearish" else max(updated_p_up, 1.0 - updated_p_up)
    adverse_move = (-live_move) if direction == "bullish" else live_move if direction == "bearish" else 0.0
    reasons: List[str] = []
    if direction == "neutral":
        status = "CAUTION"
        reasons.append("original next-open direction was neutral")
    elif favorable < config.live_forecast_invalidate_prob:
        status = "INVALIDATED"
        reasons.append(f"live favorable-direction probability fell to {favorable*100:.0f}%")
    elif favorable < config.live_forecast_caution_prob:
        status = "CAUTION"
        reasons.append(f"live favorable-direction probability only {favorable*100:.0f}%")
    else:
        status = "PASS"
    if p90 > 0 and adverse_move > 0:
        adverse_mult = adverse_move / p90
        if adverse_mult >= config.live_forecast_adverse_gap_invalidate_mult:
            status = "INVALIDATED"
            reasons.append(f"adverse live move is {adverse_mult:.2f}x the entry p90 overnight gap")
        elif adverse_mult >= config.live_forecast_adverse_gap_caution_mult and status == "PASS":
            status = "CAUTION"
            reasons.append(f"adverse live move is {adverse_mult:.2f}x the entry p90 overnight gap")
    if not reasons:
        reasons.append("live overnight move still supports the frozen close-time thesis")
    return status, reasons


def track_market_forecast(symbol: str, state: MarketState, config: Config, force: bool = False) -> None:
    """Freeze the latest close-time market forecast independently of option alerts."""
    if not config.always_track_market_overnight:
        return
    now = now_local(config.market_timezone)
    if not force and not _in_clock_window(now, config.entry_window_start, config.entry_window_end):
        return
    if not state.probability.valid:
        return
    reference_time = most_recent_regular_close(now, config.market_timezone) if force else now
    target_open = next_regular_open(reference_time, config.market_timezone)
    lo, hi, n_eff = probability_uncertainty_90(state.probability)
    calibrated, cal_n, cal_brier = calibration_for_probability(symbol, state.overnight_p_up, config)
    tracker = load_tracker(config)
    markets = tracker.setdefault("markets", {})
    key = _market_watch_key(symbol, target_open)
    old = markets.get(key, {}) if isinstance(markets.get(key), dict) else {}
    markets[key] = {
        "symbol": symbol.upper(),
        "reference_price": state.last_price,
        "reference_time": reference_time.isoformat(),
        "target_open": target_open.isoformat(),
        "intended_exit": intended_exit_datetime(reference_time, config).isoformat(),
        "entry_p_up": state.overnight_p_up,
        "entry_p_up_lo90": state.overnight_consensus.range_low,
        "entry_p_up_hi90": state.overnight_consensus.range_high,
        "entry_effective_n": state.overnight_consensus.effective_n or n_eff,
        "entry_analog_n": state.probability.samples,
        "forecast_direction": _forecast_direction_from_state(state),
        "technical_bias": state.bias,
        "technical_alignment": state.confidence,
        "entry_p90_gap": state.pctl_abs_gap,
        "entry_vix": state.vix_level,
        "entry_vix_bucket": state.vix_bucket,
        "event_severity": state.event_risk.severity,
        "event_note": state.event_risk.note,
        "calibrated_p_up": calibrated,
        "calibration_n": cal_n,
        "calibration_brier": cal_brier,
        "last_status": old.get("last_status", "ENTRY"),
        "last_live_p_up": old.get("last_live_p_up", state.overnight_p_up),
        "last_live_move": old.get("last_live_move", 0.0),
        "last_alert_time": old.get("last_alert_time", ""),
        "morning_report_date": old.get("morning_report_date", ""),
        "bootstrapped": bool(force),
        "model_version": "v13",
        "model_agreement": state.overnight_consensus.agreement,
        "consensus_components": state.overnight_consensus.components,
    }
    save_tracker(config, tracker)
    _record_forecast_log_entry(markets[key], config)


def bootstrap_market_watches(symbols: List[str], config: Config) -> None:
    if not config.always_track_market_overnight or not config.bootstrap_market_watch_after_close:
        return
    tracker = load_tracker(config)
    markets = tracker.get("markets", {}) if isinstance(tracker, dict) else {}
    now = now_local(config.market_timezone)
    target = next_regular_open(now, config.market_timezone)
    missing = [s for s in symbols if _market_watch_key(s, target) not in markets]
    for symbol in missing:
        try:
            daily, m30, m5 = get_histories(symbol, config)
            state = analyze_market_state(symbol, daily, m30, m5, config)
            track_market_forecast(symbol, state, config, force=True)
            print(f"[monitor] Bootstrapped {symbol} market watch after close; reference may be less precise than a 12:30-12:58 PT snapshot.")
        except Exception as exc:
            print(f"[warn] {symbol}: could not bootstrap overnight market watch: {exc}", file=sys.stderr)


def _send_market_watch_discord(
    config: Config,
    watch: Dict[str, Any],
    status: str,
    proxy_price: float,
    source: str,
    live_move: float,
    updated_p_up: float,
    reasons: List[str],
    fut_note: str,
    ext_note: str,
    morning_report: bool,
) -> None:
    if not config.discord_webhook_url or not config.live_forecast_alerts:
        return
    symbol = str(watch.get("symbol", ""))
    original = float(watch.get("entry_p_up", 0.50) or 0.50)
    lo = float(watch.get("entry_p_up_lo90", original) or original)
    hi = float(watch.get("entry_p_up_hi90", original) or original)
    cal = watch.get("calibrated_p_up")
    cal_txt = "not enough local history"
    if cal is not None:
        cal_txt = f"{float(cal)*100:.1f}% (n={int(watch.get('calibration_n',0) or 0)})"
    prefix = "MORNING MARKET REPORT" if morning_report else "LIVE OVERNIGHT FORECAST"
    color = {"PASS": 5763719, "CAUTION": 16705372, "INVALIDATED": 15548997}.get(status, 3447003)
    fields = [
        {"name": "Thesis status", "value": status, "inline": True},
        {"name": "Live indication", "value": f"{proxy_price:.2f} ({live_move*100:+.2f}%) via {source}", "inline": True},
        {"name": "Forecast direction", "value": str(watch.get("forecast_direction", "neutral")).upper(), "inline": True},
        {"name": "Original forecast", "value": (
            f"Reference {float(watch.get('reference_price',0)):.2f}\n"
            f"Historical P(up next open) {original*100:.1f}%\n"
            f"Approx 90% range {lo*100:.0f}-{hi*100:.0f}% | eff n {float(watch.get('entry_effective_n',0)):.1f}\n"
            f"Technical alignment {str(watch.get('technical_bias','neutral')).upper()} {int(watch.get('technical_alignment',0))}/100"
        )[:1024], "inline": False},
        {"name": "Live overnight state", "value": (
            f"Heuristic P(open above reference) {updated_p_up*100:.1f}%\n"
            f"P(open below reference) {(1.0-updated_p_up)*100:.1f}%\n"
            f"Local calibration near original: {cal_txt}\n"
            f"Why: {'; '.join(reasons)}"
        )[:1024], "inline": False},
        {"name": "Futures", "value": (fut_note or "n/a")[:1024], "inline": False},
        {"name": "ETF extended hours", "value": (ext_note or "n/a")[:1024], "inline": False},
        {"name": "Event context", "value": f"{watch.get('event_severity','LOW')} | {watch.get('event_note','')}"[:1024], "inline": False},
    ]
    payload = {"embeds": [{
        "title": f"{prefix} — {symbol}",
        "description": "Original close-time forecast is frozen; live probability is a separate heuristic update using actual overnight movement.",
        "color": color,
        "fields": fields,
    }]}
    try:
        resp = requests.post(config.discord_webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[warn] failed to send market-watch Discord update: {exc}", file=sys.stderr)


def monitor_market_watches(config: Config, symbols: Optional[List[str]] = None) -> None:
    resolve_forecast_log(config)
    if symbols:
        bootstrap_market_watches(symbols, config)
    tracker = load_tracker(config)
    markets: Dict[str, Any] = tracker.get("markets", {}) if isinstance(tracker, dict) else {}
    if not markets:
        print("[market] No frozen late-day market forecasts are active.")
        return
    now = now_local(config.market_timezone)
    changed = False
    remove: List[str] = []
    print("\n" + divider("="))
    print(f"LIVE OVERNIGHT MARKET WATCH | {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(divider("="))
    for key, watch in list(markets.items()):
        try:
            target_open = dt.datetime.fromisoformat(str(watch.get("target_open")))
            intended_exit = dt.datetime.fromisoformat(str(watch.get("intended_exit")))
        except Exception:
            remove.append(key)
            continue
        if now > intended_exit + dt.timedelta(hours=2):
            remove.append(key)
            continue
        symbol = str(watch.get("symbol", ""))
        ref_price = float(watch.get("reference_price", 0.0) or 0.0)
        ref_dt = dt.datetime.fromisoformat(str(watch.get("reference_time")))
        ext_price, ext_move, ext_ts, ext_note = get_extended_hours_snapshot(symbol, ref_price, config)
        fut_symbol, _, _, fut_move, fut_note = get_futures_reference_snapshot(symbol, ref_dt, config)
        ext_fresh = ext_ts is not None and now - ext_ts <= dt.timedelta(minutes=max(1, config.extended_quote_stale_minutes))
        if ext_price is not None and ext_move is not None and ext_fresh:
            proxy_price, live_move, source = ext_price, ext_move, "ETF extended-hours"
        elif fut_move is not None and ref_price > 0:
            proxy_price, live_move, source = ref_price * (1.0 + fut_move), fut_move, fut_symbol or "futures proxy"
        elif ext_price is not None and ext_move is not None:
            proxy_price, live_move, source = ext_price, ext_move, "stale ETF extended-hours"
        else:
            print(f"  {symbol}: no usable overnight quote yet.")
            continue
        updated_p_up = _dynamic_market_p_up(watch, proxy_price, now)
        status, reasons = _market_thesis_status(watch, updated_p_up, live_move, config)
        original = float(watch.get("entry_p_up", 0.50) or 0.50)
        lo = float(watch.get("entry_p_up_lo90", original) or original)
        hi = float(watch.get("entry_p_up_hi90", original) or original)
        cal = watch.get("calibrated_p_up")
        print(f"{symbol} MARKET THESIS")
        print(f"  Frozen reference    : {ref_price:.2f} at {ref_dt.strftime('%m-%d %H:%M')}")
        print(f"  Forecast direction  : {str(watch.get('forecast_direction','neutral')).upper()}")
        print(f"  Technical alignment : {str(watch.get('technical_bias','neutral')).upper()} {int(watch.get('technical_alignment',0))}/100")
        print(f"  Original P(up open) : {original*100:.1f}%  (approx 90% range {lo*100:.0f}-{hi*100:.0f}%, eff n {float(watch.get('entry_effective_n',0)):.1f})")
        if cal is None:
            print(f"  Local calibration   : not enough resolved forecasts (n={int(watch.get('calibration_n',0) or 0)})")
        else:
            print(f"  Local calibration   : {float(cal)*100:.1f}% actual up near this forecast (n={int(watch.get('calibration_n',0) or 0)})")
        print(f"  Live indication     : {proxy_price:.2f} ({live_move*100:+.2f}%) via {source}")
        print(f"  LIVE P(up open)*    : {updated_p_up*100:.1f}%")
        print(f"  LIVE P(down open)*  : {(1.0-updated_p_up)*100:.1f}%")
        print(f"  Thesis status       : {status}")
        print(f"  Why                 : {'; '.join(reasons)}")
        print(f"  Futures             : {fut_note}")
        print(f"  Extended quote      : {ext_note}")
        print("  * Live probabilities are heuristic updates, not guaranteed/calibrated forecasts.")
        print(divider())

        prev_status = str(watch.get("last_status", "ENTRY"))
        prev_prob = float(watch.get("last_live_p_up", original) or original)
        prev_move = float(watch.get("last_live_move", 0.0) or 0.0)
        status_changed = status != prev_status
        prob_changed = abs(updated_p_up - prev_prob) >= config.live_forecast_material_prob_change
        move_changed = abs(live_move - prev_move) >= config.monitor_material_move_pct
        morning_report = _in_clock_window(now, config.morning_report_start, config.morning_report_end) and watch.get("morning_report_date") != now.date().isoformat()
        if status_changed or prob_changed or move_changed or morning_report:
            _send_market_watch_discord(config, watch, status, proxy_price, source, live_move, updated_p_up, reasons, fut_note, ext_note, morning_report)
            watch["last_alert_time"] = now.isoformat()
            if morning_report:
                watch["morning_report_date"] = now.date().isoformat()
        watch["last_status"] = status
        watch["last_live_p_up"] = updated_p_up
        watch["last_live_move"] = live_move
        watch["last_proxy_price"] = proxy_price
        watch["last_proxy_source"] = source
        changed = True
    for key in remove:
        markets.pop(key, None)
        changed = True
    if changed:
        tracker["markets"] = markets
        save_tracker(config, tracker)


def _tracked_key(symbol: str, candidate: SpreadCandidate) -> str:
    return f"{symbol}|{candidate.spread_type}|{candidate.expiration}|{candidate.short_strike:.4f}|{candidate.long_strike:.4f}"


def track_alerted_idea(symbol: str, state: MarketState, candidate: SpreadCandidate, config: Config) -> None:
    """Persist an alerted late-day idea so the same setup can be watched overnight.

    The tracker records ALERTS, not confirmed brokerage positions. If the user did not
    take an alert, its overnight updates can simply be ignored or TRACK_ALERTED_IDEAS can
    be disabled.
    """
    if not config.track_alerted_ideas:
        return
    now = now_local(config.market_timezone)
    if config.track_only_entry_window and not _in_clock_window(now, config.entry_window_start, config.entry_window_end):
        return

    exit_dt = intended_exit_datetime(now, config)
    hold_events = hold_window_event_notes(symbol, now, exit_dt, config)
    fut_symbol, _, fut_latest, _, fut_note = get_futures_reference_snapshot(symbol, now, config)

    tracker = load_tracker(config)
    ideas = tracker.setdefault("ideas", {})
    key = _tracked_key(symbol, candidate)
    ideas[key] = {
        "symbol": symbol,
        "spread_type": candidate.spread_type,
        "expiration": candidate.expiration,
        "short_strike": candidate.short_strike,
        "long_strike": candidate.long_strike,
        "width": candidate.width,
        "credit": candidate.credit,
        "max_loss": candidate.max_loss,
        "breakeven": candidate.breakeven,
        "entry_spot": state.last_price,
        "entry_time": now.isoformat(),
        "intended_exit": exit_dt.isoformat(),
        "short_iv": candidate.short_iv,
        "long_iv": candidate.long_iv,
        "entry_open_survival": candidate.open_survival_prob,
        "entry_morning_profit_prob": candidate.morning_profit_prob,
        "entry_morning_profit_prob_stress": candidate.morning_profit_prob_stress,
        "entry_overnight_eligibility": candidate.overnight_eligibility,
        "entry_p90_gap": state.pctl_abs_gap,
        "entry_expected_move_pct": candidate.expected_move_pct,
        "entry_event_severity": state.event_risk.severity,
        "hold_window_events": hold_events,
        "futures_symbol": fut_symbol,
        "futures_entry_price": fut_latest,
        "futures_entry_note": fut_note,
        "last_status": candidate.overnight_eligibility,
        "last_updated_survival": candidate.open_survival_prob,
        "last_live_move": 0.0,
        "last_alert_time": now.isoformat(),
        "morning_report_date": "",
    }
    save_tracker(config, tracker)
    print(f"[track] {symbol} {candidate.title()} saved for overnight monitoring through {exit_dt.strftime('%Y-%m-%d %H:%M %Z')}.")


def _dynamic_open_survival(
    idea: Dict[str, Any],
    proxy_price: float,
    now: dt.datetime,
    config: Config,
) -> float:
    """Heuristic real-time update of next-open survival as the overnight path unfolds.

    The original empirical open-survival estimate is blended with a live cushion model.
    Remaining uncertainty is scaled by sqrt(time remaining) from the entry-time p90 gap.
    This is a monitoring heuristic, not a calibrated probability guarantee.
    """
    entry = float(idea.get("entry_spot", 0.0) or 0.0)
    be = float(idea.get("breakeven", 0.0) or 0.0)
    entry_survival = float(idea.get("entry_open_survival", 0.50) or 0.50)
    p90 = float(idea.get("entry_p90_gap", 0.0) or 0.0)
    if entry <= 0 or proxy_price <= 0 or p90 <= 0:
        return entry_survival

    try:
        entry_dt = dt.datetime.fromisoformat(str(idea["entry_time"]))
        open_dt = next_regular_open(entry_dt, config.market_timezone)
    except Exception:
        open_dt = next_regular_open(now, config.market_timezone)
        entry_dt = now

    total = max((open_dt - entry_dt).total_seconds(), 1.0)
    remaining = max((open_dt - now).total_seconds(), 0.0)
    frac_remaining = min(max(remaining / total, 0.02), 1.0)
    sigma_full = max(p90 / 1.6448536269514722, 1e-5)  # p90(|N(0,sigma)|) ~= 1.645 sigma
    sigma_remaining = max(sigma_full * math.sqrt(frac_remaining), 1e-5)

    current_r = proxy_price / entry - 1.0
    be_r = be / entry - 1.0
    if str(idea.get("spread_type")) == "bull_put":
        live_prob = float(norm.cdf((current_r - be_r) / sigma_remaining))
    else:
        live_prob = float(norm.cdf((be_r - current_r) / sigma_remaining))

    elapsed = 1.0 - frac_remaining
    live_weight = 0.40 + 0.60 * elapsed
    return min(max(live_weight * live_prob + (1.0 - live_weight) * entry_survival, 0.0), 1.0)


def _monitor_status(
    idea: Dict[str, Any],
    proxy_price: float,
    updated_survival: float,
    mega_avg: Optional[float],
    mega_moves: Dict[str, float],
    config: Config,
) -> Tuple[str, float, float, List[str]]:
    entry = float(idea.get("entry_spot", 0.0) or 0.0)
    be = float(idea.get("breakeven", 0.0) or 0.0)
    spread_type = str(idea.get("spread_type", ""))
    initial_cushion = abs(entry - be)
    remaining_cushion = (proxy_price - be) if spread_type == "bull_put" else (be - proxy_price)
    consumed = 1.0 - remaining_cushion / max(initial_cushion, 1e-9)
    consumed = max(0.0, consumed)
    reasons: List[str] = []

    if remaining_cushion <= 0:
        status = "FAIL"
        reasons.append("proxy price is beyond breakeven")
    elif consumed >= config.monitor_cushion_fail_fraction or updated_survival < 0.50:
        status = "FAIL"
        reasons.append(f"{consumed*100:.0f}% of breakeven cushion consumed")
    elif consumed >= config.monitor_cushion_warn_fraction or updated_survival < 0.65:
        status = "CAUTION"
        reasons.append(f"{consumed*100:.0f}% of breakeven cushion consumed")
    else:
        status = "PASS"

    events = list(idea.get("hold_window_events", []) or [])
    if events and status == "PASS":
        status = "CAUTION"
        reasons.append("scheduled event may occur inside holding window")

    # For SPY only, a broad mega-cap move against the spread or a very large single-name
    # move is useful overnight context. It is a warning, not an index-fair-value model.
    adverse_mega = False
    if mega_avg is not None:
        adverse_mega = (spread_type == "bull_put" and mega_avg < -0.003) or (spread_type == "bear_call" and mega_avg > 0.003)
    extreme_names = [k for k, v in mega_moves.items() if abs(v) >= config.mega_cap_move_alert_pct]
    if adverse_mega or extreme_names:
        if status == "PASS":
            status = "CAUTION"
        if adverse_mega:
            reasons.append(f"mega-cap basket moving against setup ({mega_avg*100:+.2f}%)")
        if extreme_names:
            reasons.append("large after-hours move in " + ", ".join(extreme_names[:4]))

    if not reasons:
        reasons.append("breakeven cushion and live overnight indicators remain supportive")
    return status, remaining_cushion, consumed, reasons


def _send_monitor_discord(
    config: Config,
    idea: Dict[str, Any],
    status: str,
    proxy_price: float,
    source: str,
    live_move: float,
    updated_survival: float,
    remaining_cushion: float,
    consumed: float,
    fut_note: str,
    ext_note: str,
    mega_note: str,
    reasons: List[str],
    morning_report: bool,
) -> None:
    if not config.discord_webhook_url:
        return
    symbol = str(idea.get("symbol", ""))
    title_prefix = "MORNING REPORT" if morning_report else "OVERNIGHT UPDATE"
    color = {"PASS": 5763719, "CAUTION": 16705372, "FAIL": 15548997}.get(status, 3447003)
    events = idea.get("hold_window_events", []) or []
    event_text = "\n".join(f"• {x}" for x in events) if events else "No known timed event inside intended hold window"
    fields = [
        {"name": "Status", "value": status, "inline": True},
        {"name": "Live proxy", "value": f"{proxy_price:.2f} ({live_move*100:+.2f}%) via {source}", "inline": True},
        {"name": "Updated open survival", "value": f"{updated_survival*100:.1f}%", "inline": True},
        {"name": "Tracked spread", "value": (
            f"{idea.get('spread_type')} | sell {float(idea.get('short_strike',0)):.2f} / buy {float(idea.get('long_strike',0)):.2f}\n"
            f"Entry spot {float(idea.get('entry_spot',0)):.2f} | BE {float(idea.get('breakeven',0)):.2f}\n"
            f"Entry next-open P(profit) {float(idea.get('entry_morning_profit_prob',0.5))*100:.1f}% "
            f"(IV stress {float(idea.get('entry_morning_profit_prob_stress',0.5))*100:.1f}%)"
        )[:1024], "inline": False},
        {"name": "Cushion", "value": f"Remaining ${remaining_cushion:.2f} | consumed {consumed*100:.0f}%", "inline": False},
        {"name": "Why", "value": "; ".join(reasons)[:1024], "inline": False},
        {"name": "Futures", "value": (fut_note or "n/a")[:1024], "inline": False},
        {"name": "ETF extended hours", "value": (ext_note or "n/a")[:1024], "inline": False},
        {"name": "Mega-cap sentiment", "value": (mega_note or "n/a")[:1024], "inline": False},
        {"name": "Events in hold window", "value": event_text[:1024], "inline": False},
    ]
    payload = {"embeds": [{
        "title": f"{title_prefix} — {symbol} {str(idea.get('spread_type','')).replace('_',' ').title()}",
        "description": "Tracked alert only — this is not confirmation that a position was opened.",
        "color": color,
        "fields": fields,
    }]}
    try:
        resp = requests.post(config.discord_webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[warn] failed to send overnight Discord update: {exc}", file=sys.stderr)


def monitor_tracked_ideas(config: Config, symbols: Optional[List[str]] = None) -> None:
    # v11.3: the underlying market watch runs even if no spread qualified.
    monitor_market_watches(config, symbols)

    tracker = load_tracker(config)
    ideas: Dict[str, Any] = tracker.get("ideas", {}) if isinstance(tracker, dict) else {}
    if not ideas:
        print("[monitor] No tracked spread alerts are active; underlying market watch is still running above.")
        return

    now = now_local(config.market_timezone)
    changed = False
    remove_keys: List[str] = []

    print("\n" + divider("="))
    print(f"OVERNIGHT MONITOR | {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(divider("="))

    for key, idea in list(ideas.items()):
        try:
            entry_dt = dt.datetime.fromisoformat(str(idea["entry_time"]))
            exit_dt = dt.datetime.fromisoformat(str(idea["intended_exit"]))
        except Exception:
            remove_keys.append(key)
            continue

        # Keep a short grace period after the intended exit, then retire the tracked idea.
        if now > exit_dt + dt.timedelta(hours=2):
            remove_keys.append(key)
            continue

        symbol = str(idea.get("symbol", ""))
        entry_spot = float(idea.get("entry_spot", 0.0) or 0.0)
        ext_price, ext_move, ext_ts, ext_note = get_extended_hours_snapshot(symbol, entry_spot, config)
        fut_symbol, _, _, fut_move, fut_note = get_futures_reference_snapshot(symbol, entry_dt, config)

        ext_fresh = False
        if ext_ts is not None:
            ext_fresh = now - ext_ts <= dt.timedelta(minutes=max(1, config.extended_quote_stale_minutes))
        if ext_price is not None and ext_move is not None and ext_fresh:
            proxy_price = ext_price
            live_move = ext_move
            source = "ETF extended-hours"
        elif fut_move is not None and entry_spot > 0:
            proxy_price = entry_spot * (1.0 + fut_move)
            live_move = fut_move
            source = fut_symbol or "futures proxy"
        elif ext_price is not None and ext_move is not None:
            proxy_price = ext_price
            live_move = ext_move
            source = "stale ETF extended-hours"
        else:
            print(f"  {symbol}: no usable extended-hours or futures proxy data yet.")
            continue

        mega_avg: Optional[float] = None
        mega_note = "not applicable"
        mega_moves: Dict[str, float] = {}
        if symbol.upper() == "SPY":
            mega_avg, mega_note, mega_moves = get_megacap_afterhours_sentiment(config)

        updated_survival = _dynamic_open_survival(idea, proxy_price, now, config)
        status, remaining_cushion, consumed, reasons = _monitor_status(
            idea, proxy_price, updated_survival, mega_avg, mega_moves, config
        )

        print(f"{symbol} {str(idea.get('spread_type','')).replace('_',' ').upper()}")
        print(f"  Proxy             : {proxy_price:.2f} ({live_move*100:+.2f}%) via {source}")
        print(f"  Breakeven         : {float(idea.get('breakeven',0)):.2f}")
        print(f"  Cushion remaining : ${remaining_cushion:.2f} | {consumed*100:.0f}% consumed")
        print(f"  Open survival est.: {updated_survival*100:.1f}% (entry {float(idea.get('entry_open_survival',0.5))*100:.1f}%)")
        print(f"  Status            : {status}")
        print(f"  Why               : {'; '.join(reasons)}")
        print(f"  Futures           : {fut_note}")
        print(f"  Extended quote    : {ext_note}")
        if symbol.upper() == "SPY":
            print(f"  Mega-cap sentiment: {mega_note}")
        if idea.get("hold_window_events"):
            print("  Hold-window events: " + " | ".join(idea["hold_window_events"]))
        print(divider())

        prev_status = str(idea.get("last_status", ""))
        prev_prob = float(idea.get("last_updated_survival", idea.get("entry_open_survival", 0.5)) or 0.5)
        prev_move = float(idea.get("last_live_move", 0.0) or 0.0)
        status_changed = status != prev_status
        prob_changed = abs(updated_survival - prev_prob) >= config.monitor_probability_change
        move_changed = abs(live_move - prev_move) >= config.monitor_material_move_pct
        morning_report = _in_clock_window(now, config.morning_report_start, config.morning_report_end) and idea.get("morning_report_date") != now.date().isoformat()

        if status_changed or prob_changed or move_changed or morning_report:
            _send_monitor_discord(
                config, idea, status, proxy_price, source, live_move, updated_survival,
                remaining_cushion, consumed, fut_note, ext_note, mega_note, reasons, morning_report
            )
            idea["last_alert_time"] = now.isoformat()
            if morning_report:
                idea["morning_report_date"] = now.date().isoformat()

        idea["last_status"] = status
        idea["last_updated_survival"] = updated_survival
        idea["last_live_move"] = live_move
        idea["last_proxy_price"] = proxy_price
        idea["last_proxy_source"] = source
        changed = True

    for key in remove_keys:
        ideas.pop(key, None)
        changed = True
    if changed:
        tracker["ideas"] = ideas
        save_tracker(config, tracker)


# ---------------------------- Discord / scanning ----------------------------

CONSOLE_WIDTH = 92


def divider(char: str = "-") -> str:
    return char * CONSOLE_WIDTH


def _signal_word(signal: int) -> str:
    return {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}.get(signal, "NEUTRAL")


def _analog_direction(prob: HistoricalProbability) -> str:
    if not prob.valid:
        return "UNKNOWN"
    if prob.p_up_next_open >= 0.55:
        return "BULLISH"
    if prob.p_up_next_open <= 0.45:
        return "BEARISH"
    return "NEUTRAL"


def _signal_conflict_text(state: MarketState) -> str:
    """Readable warning when technical direction and historical overnight analogs disagree."""
    analog = _analog_direction(state.probability)
    technical = state.bias.upper()
    if analog in {"BULLISH", "BEARISH"} and technical in {"BULLISH", "BEARISH"} and analog != technical:
        return (
            f"SIGNAL CONFLICT: technical={technical}, "
            f"historical next-open={analog} "
            f"(P(up)={state.probability.p_up_next_open*100:.1f}%)"
        )
    return ""


def print_market_snapshot(symbol: str, state: MarketState, config: Optional[Config] = None) -> None:
    """Print one scan in compact sections that are easy to read in Windows CMD/PowerShell."""
    print("\n" + divider("="))
    print(f"{symbol} MARKET SNAPSHOT")
    print(divider("="))

    print("TECHNICAL ALIGNMENT")
    print(
        f"  Spot              : {state.last_price:.2f}\n"
        f"  Alignment         : {state.bias.upper():<8}  ({state.confidence}/100 score; NOT a probability)\n"
        f"  Directional lean  : {state.directional_lean.upper():<8}  ({state.lean_confidence}/100 score)"
    )

    print(divider())
    print("TIMEFRAMES")
    print(
        f"  Daily             : {_signal_word(state.daily_signal)} ({state.daily_signal:+d})\n"
        f"  30 minute         : {_signal_word(state.m30_signal)} ({state.m30_signal:+d})\n"
        f"  5 minute          : {_signal_word(state.m5_signal)} ({state.m5_signal:+d})"
    )

    print(divider())
    print("RECENT MARKET MOVES")
    print(
        f"  1 day             : {state.ret_1d*100:+.2f}%\n"
        f"  3 days            : {state.ret_3d*100:+.2f}%\n"
        f"  5 days            : {state.ret_5d*100:+.2f}%\n"
        f"  Today's move      : {state.day_move_atr:+.2f} ATR\n"
        f"  90% overnight gap : {state.pctl_abs_gap*100:.2f}%  (regime={state.vix_bucket})"
    )

    print(divider())
    print("VOLATILITY REGIME / FUTURES CONTEXT")
    vix_txt = f"{state.vix_level:.2f}" if state.vix_level is not None else "n/a"
    vix_pctl_txt = f"{state.vix_percentile*100:.0f}th pct (2y)" if state.vix_percentile is not None else "n/a"
    print(f"  VIX               : {vix_txt}  ({vix_pctl_txt}, bucket={state.vix_bucket})")
    if state.futures_symbol is None:
        print("  Futures context    : disabled / no proxy for this symbol")
    elif state.futures_overnight_move_pct is None:
        print(f"  Futures context    : {state.futures_symbol} — {state.futures_note or 'no data'}")
    else:
        print(
            f"  Futures context    : {state.futures_symbol} "
            f"{state.futures_overnight_move_pct*100:+.2f}% vs last close"
        )

    print(divider())
    print("SPY NEXT-OPEN CONSENSUS FORECAST")
    cf = state.overnight_consensus
    print(
        f"  FINAL P(UP)       : {cf.p_up*100:5.1f}%\n"
        f"  FINAL P(DOWN)     : {cf.p_down*100:5.1f}%\n"
        f"  FORECAST          : {cf.direction.upper()}\n"
        f"  MODEL AGREEMENT   : {cf.agreement}\n"
        f"  Model range*      : {cf.range_low*100:5.1f}% - {cf.range_high*100:5.1f}% P(up)\n"
        f"  Direction strength: {cf.strength}/100"
    )
    if cf.calibrated_p_up is not None and cf.calibration_weight > 0:
        print(
            f"  Raw consensus     : {cf.raw_p_up*100:5.1f}%\n"
            f"  Local calibration : {cf.calibrated_p_up*100:5.1f}% "
            f"(n={cf.calibration_n}, blend weight={cf.calibration_weight:.0%})"
        )
    print("\n  COMPONENTS (diagnostics; FINAL above is the directional answer)")
    for name, prob in cf.components.items():
        wt = cf.component_weights.get(name, 0.0)
        print(f"    {name:<24}: {prob*100:5.1f}% P(up)  weight={wt:.0%}")
    if state.large_day.valid:
        print(
            f"\n  Today's structure\n"
            f"    Close-to-now move : {state.large_day.total_return*100:+.2f}% ({state.large_day.move_atr:+.2f} ATR)\n"
            f"    Opening gap       : {state.large_day.gap_return*100:+.2f}%\n"
            f"    Open -> current   : {state.large_day.session_return*100:+.2f}%\n"
            f"    Final 30m / 60m  : {state.large_day.final_30m_return*100:+.2f}% / {state.large_day.final_60m_return*100:+.2f}%\n"
            f"    Large-day reversal: {state.large_day.p_reversal_next_open*100:.1f}% (n={state.large_day.samples})"
        )
    print("\n  *Model range is a heuristic uncertainty band, not a statistical guarantee.")

    print(divider())
    print("SUB-MODEL / CALIBRATION DETAILS")
    if state.probability.valid:
        lo90, hi90, n_eff = probability_uncertainty_90(state.probability)
        print(
            f"  General historical analog P(up): {state.probability.p_up_next_open*100:.1f}% "
            f"(n={state.probability.samples}, effective n~{n_eff:.1f})\n"
            f"  General analog 90% band        : {lo90*100:.1f}% - {hi90*100:.1f}%"
        )
    else:
        print("  General historical analog model unavailable.")
    print(f"  Blend detail                    : {cf.note}")

    print(divider())
    print("EVENT / FED RISK")
    print(f"  Event risk        : {state.event_risk.severity}")
    print(f"  Event             : {state.event_risk.note}")
    if state.event_risk.next_fomc:
        print(f"  FOMC              : {state.event_risk.next_fomc}")
    if state.event_risk.fed_rate:
        print(f"  Fed rates         : {state.event_risk.fed_rate}")

    print(divider())
    print("MODEL REASON")
    print(f"  {state.reason}")
    print(divider("="))


def print_candidate_alert(symbol: str, candidate: SpreadCandidate, state: MarketState) -> None:
    print("\n" + divider("#"))
    print(f"TRADE IDEA | {symbol} {candidate.title().upper()} | GRADE {candidate.grade}")
    print(divider("#"))
    print(
        f"  Spot              : {state.last_price:.2f}\n"
        f"  Sell / Buy        : {candidate.short_strike:.2f} / {candidate.long_strike:.2f}\n"
        f"  Expiration / DTE  : {candidate.expiration} / {candidate.dte}\n"
        f"  Credit            : {candidate.credit:.2f}\n"
        f"  Short delta       : {candidate.short_delta:.3f}\n"
        f"  Return on risk    : {candidate.ror*100:.1f}%\n"
        f"  Max loss          : {candidate.max_loss:.2f}\n"
        f"  Breakeven         : {candidate.breakeven:.2f}\n"
        f"  S/R clearance     : {candidate.sr_clearance:.2f} vs {candidate.sr_required:.2f} required"
        f"{'  [WEIGHTED OVERRIDE]' if candidate.sr_override_used else ''}"
    )
    print(divider())
    print("OVERNIGHT ELIGIBILITY")
    print(
        f"  Decision          : {candidate.overnight_eligibility}\n"
        f"  Next-open survival: {candidate.open_survival_prob*100:.1f}%\n"
        f"  BE distance       : {candidate.breakeven_distance_pct*100:.2f}%\n"
        f"  BE / p90 gap      : {candidate.be_gap_buffer_mult:.2f}x\n"
        f"  BE / implied move : {candidate.be_expected_move_buffer_mult:.2f}x\n"
        f"  Reason            : {candidate.overnight_reasons}"
    )
    print(divider())
    print("NEXT-MORNING P/L MODEL")
    print(
        f"  P(profit at open) : {candidate.morning_profit_prob*100:.1f}%\n"
        f"  IV-stress P(profit): {candidate.morning_profit_prob_stress*100:.1f}%\n"
        f"  Median est. P/L   : ${candidate.morning_median_pl:+.0f} per spread\n"
        f"  25th pct P/L      : ${candidate.morning_p25_pl:+.0f} per spread\n"
        f"  10th pct P/L      : ${candidate.morning_p10_pl:+.0f} per spread\n"
        f"  P(loss >25% max)  : {candidate.morning_loss25_prob*100:.1f}%\n"
        f"  P(loss >50% max)  : {candidate.morning_loss50_prob*100:.1f}%"
    )
    print(divider())
    print("PROBABILITY / RISK")
    print(
        f"  Option POP (BE)   : {candidate.option_pop*100:.1f}%\n"
        f"  Historical POP    : {candidate.historical_pop*100:.1f}%\n"
        f"  Spread POP blend     : {candidate.consensus_pop*100:.1f}%\n"
        f"  No-breach history : {candidate.no_breach_prob*100:.1f}%\n"
        f"  Expected move     : +/-{candidate.expected_move_pct*100:.2f}%\n"
        f"  Overnight risk    : {candidate.overnight_risk}\n"
        f"  Event risk        : {state.event_risk.severity}"
    )
    print(divider())
    print(f"  Why: {candidate.rationale}")
    print(divider("#") + "\n")


def send_alert(config: Config, candidate: SpreadCandidate, symbol: str, state: MarketState) -> None:
    print_candidate_alert(symbol, candidate, state)
    if not config.discord_webhook_url:
        return

    grade_color = {"A": 5763719, "B": 16705372, "C": 9807270}.get(candidate.grade, 3447003)
    event_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(state.event_risk.severity, "⚪")

    event_prefix = "⚠️ EVENT-RISK IDEA — " if state.event_risk.severity in {"MEDIUM", "HIGH"} else ""
    speculative = state.event_risk.severity in {"MEDIUM", "HIGH"} and candidate.grade == "C"
    neutral_cautious = state.bias == "neutral" and state.event_risk.severity == "LOW"
    if speculative:
        status_text = "SPECULATIVE / EVENT RISK"
    elif candidate.sr_override_used and state.event_risk.severity in {"MEDIUM", "HIGH"}:
        status_text = "S/R OVERRIDE + EVENT RISK"
    elif candidate.sr_override_used:
        status_text = "S/R WEIGHTED OVERRIDE"
    elif state.event_risk.severity in {"MEDIUM", "HIGH"}:
        status_text = "EVENT RISK"
    elif neutral_cautious:
        status_text = "CAUTIOUS / NEUTRAL LEAN"
    else:
        status_text = "NORMAL"

    conflict = _signal_conflict_text(state)
    vix_line = f"**VIX:** {state.vix_level:.2f}" if state.vix_level is not None else "**VIX:** n/a"
    if state.futures_symbol is not None and state.futures_overnight_move_pct is not None:
        vix_line += f" | **{state.futures_symbol} context:** {state.futures_overnight_move_pct*100:+.2f}%"
    cf = state.overnight_consensus
    market_block = (
        f"**FINAL next-open forecast:** {cf.direction.upper()}\n"
        f"**FINAL P(up/down):** {cf.p_up*100:.1f}% / {cf.p_down*100:.1f}%\n"
        f"**Model agreement:** {cf.agreement} | **Range:** {cf.range_low*100:.1f}-{cf.range_high*100:.1f}% P(up)\n"
        f"**Spot:** {state.last_price:.2f}\n"
        f"**Technical context only:** {state.bias.upper()} ({state.confidence}/100; not a probability)\n"
        f"**TF D / 30m / 5m:** {state.daily_signal:+d} / {state.m30_signal:+d} / {state.m5_signal:+d}\n"
        f"**1d / 3d / 5d:** {state.ret_1d*100:+.2f}% / {state.ret_3d*100:+.2f}% / {state.ret_5d*100:+.2f}%\n"
        f"**Day move:** {state.day_move_atr:+.2f} ATR | **90% gap:** {state.pctl_abs_gap*100:.2f}% "
        f"(regime={state.vix_bucket})\n"
        f"{vix_line}"
    )

    setup_block = (
        f"**Sell / Buy:** {candidate.short_strike:.2f} / {candidate.long_strike:.2f}\n"
        f"**Expiration:** {candidate.expiration} ({candidate.dte} DTE)\n"
        f"**Credit:** {candidate.credit:.2f} | **Width:** {candidate.width:.2f}\n"
        f"**Short delta:** {candidate.short_delta:.3f} | **RoR:** {candidate.ror*100:.1f}%\n"
        f"**Breakeven:** {candidate.breakeven:.2f} | **Max loss:** {candidate.max_loss:.2f}\n"
        f"**S/R clearance:** {candidate.sr_clearance:.2f} vs {candidate.sr_required:.2f} required"
        f"{' ⚠️ WEIGHTED OVERRIDE' if candidate.sr_override_used else ''}"
    )

    morning_block = (
        f"**P(profit at next open):** {candidate.morning_profit_prob*100:.1f}%\n"
        f"**IV-stress P(profit):** {candidate.morning_profit_prob_stress*100:.1f}%\n"
        f"**Median est. P/L:** ${candidate.morning_median_pl:+.0f} per spread\n"
        f"**25th / 10th pct P/L:** ${candidate.morning_p25_pl:+.0f} / ${candidate.morning_p10_pl:+.0f}\n"
        f"**P(loss >25% max):** {candidate.morning_loss25_prob*100:.1f}%\n"
        f"**P(loss >50% max):** {candidate.morning_loss50_prob*100:.1f}%"
    )

    probability_block = (
        f"**Option POP (BE):** {candidate.option_pop*100:.1f}%\n"
        f"**Historical POP:** {candidate.historical_pop*100:.1f}%\n"
        f"**Spread POP blend:** {candidate.consensus_pop*100:.1f}%\n"
        f"**Overnight eligibility:** {candidate.overnight_eligibility}\n"
        f"**Next-open survival:** {candidate.open_survival_prob*100:.1f}%\n"
        f"**BE cushion / p90 gap:** {candidate.be_gap_buffer_mult:.2f}x\n"
        f"**BE cushion / implied move:** {candidate.be_expected_move_buffer_mult:.2f}x\n"
        f"**Historical no-breach:** {candidate.no_breach_prob*100:.1f}%\n"
        f"**Expected move:** +/-{candidate.expected_move_pct*100:.2f}%\n"
        f"**Overnight risk:** {candidate.overnight_risk}\n"
        f"**Why:** {candidate.overnight_reasons}"
    )

    component_lines = []
    for name, prob in state.overnight_consensus.components.items():
        wt = state.overnight_consensus.component_weights.get(name, 0.0)
        component_lines.append(f"**{name}:** {prob*100:.1f}% P(up), w={wt:.0%}")
    analog_block = "\n".join(component_lines) if component_lines else "Consensus components unavailable."


    fed_text = " | ".join(x for x in [state.event_risk.next_fomc, state.event_risk.fed_rate] if x)
    if not fed_text:
        fed_text = "No near-term FOMC/rate context available"

    # Keep this well below Discord's 25-field embed limit. The prior version used
    # many small fields; grouped sections are both easier to read and more reliable.
    fields = [
        {"name": "Status", "value": status_text, "inline": True},
        {"name": "Grade", "value": candidate.grade, "inline": True},
        {"name": f"{event_emoji} Event Risk", "value": state.event_risk.severity, "inline": True},
        {"name": "📊 Market", "value": market_block[:1024], "inline": False},
        {"name": "🧾 Spread Setup", "value": setup_block[:1024], "inline": False},
        {"name": "🌅 Next-Morning P/L", "value": morning_block[:1024], "inline": False},
        {"name": "🎯 Probability / Overnight", "value": probability_block[:1024], "inline": False},
        {"name": "🧠 Next-Open Consensus Inputs", "value": analog_block[:1024], "inline": False},
        {"name": f"{event_emoji} Scheduled Event", "value": state.event_risk.note[:1024], "inline": False},
        {"name": "🏦 Fed / Rates", "value": fed_text[:1024], "inline": False},
        {"name": "Why", "value": candidate.rationale[:1024], "inline": False},
    ]

    payload = {
        "embeds": [
            {
                "title": f"{event_prefix}{symbol} {candidate.title()} — Grade {candidate.grade}",
                "description": (
                    "Alert only — verify current price, scheduled events, and executable fill before placing an order."
                ),
                "color": grade_color,
                "fields": fields,
            }
        ]
    }
    try:
        resp = requests.post(config.discord_webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[warn] failed to send Discord alert: {exc}", file=sys.stderr)


def scan_symbol(symbol: str, config: Config) -> List[Tuple[str, MarketState, SpreadCandidate]]:
    daily, m30, m5 = get_histories(symbol, config)
    state = analyze_market_state(symbol, daily, m30, m5, config)

    # v11.3: freeze the market-level next-open forecast during the late-day entry window
    # even if no option spread ultimately passes every filter.
    track_market_forecast(symbol, state, config)

    print_market_snapshot(symbol, state, config)

    if state.block_new_alerts:
        print("RESULT")
        print(f"  No alert: regime/lean rules did not clear the configured threshold for {symbol}.")
        print(divider() + "\n")
        return []

    ticker = yf.Ticker(symbol)
    expirations = get_candidate_expirations(ticker, config)
    if not expirations:
        print("RESULT")
        print(f"  No alert: {symbol} has no expirations inside the configured DTE range.")
        print(divider() + "\n")
        return []

    all_candidates: List[SpreadCandidate] = []
    chain_cache: List[Tuple[str, int, pd.DataFrame, pd.DataFrame]] = []
    for expiration, dte in expirations:
        try:
            chain = ticker.option_chain(expiration)
        except Exception as exc:
            print(f"[warn] {symbol}: failed to load chain for {expiration}: {exc}", file=sys.stderr)
            continue
        chain_cache.append((expiration, dte, chain.puts.copy(), chain.calls.copy()))
        all_candidates.extend(make_spreads(chain.puts, "put", state.last_price, expiration, dte, config, state))
        all_candidates.extend(make_spreads(chain.calls, "call", state.last_price, expiration, dte, config, state))

    ranked = rank_candidates(all_candidates, state, config)
    if not ranked:
        print("RESULT")
        print(f"  No fully alert-qualified {symbol} spread matched every configured filter on this scan.")
        print("  Checking the closest favored-direction candidates...")
        near_misses = find_near_miss_candidates(chain_cache, state, config)
        if near_misses:
            print_near_miss_candidates(symbol, state, config, near_misses)
        else:
            print("  No reasonable near-miss candidate cleared the relaxed diagnostic floors either.")
            print("  This is a normal result, not an error.")
            print(divider())
        print()
        return []

    limit = config.max_alerts_per_scan
    if state.event_risk.severity in {"MEDIUM", "HIGH"} and config.allow_event_risk_alerts:
        limit = min(limit, config.max_event_alerts_per_scan)
    elif state.bias == "neutral" and config.allow_neutral_directional_ideas:
        limit = min(limit, config.max_neutral_alerts_per_scan)
    return [(symbol, state, c) for c in ranked[:limit]]


def should_alert(
    seen_alerts: Dict[Tuple[str, str, str, float, float], Dict[str, float]],
    candidate: SpreadCandidate,
    symbol: str,
    config: Config,
) -> bool:
    key = candidate.key(symbol)
    now_ts = time.time()
    prev = seen_alerts.get(key)
    if prev is None:
        seen_alerts[key] = {"time": now_ts, "score": candidate.score, "credit": candidate.credit}
        return True

    time_ok = now_ts - prev["time"] >= config.alert_cooldown_seconds
    score_ok = abs(candidate.score - prev["score"]) >= config.min_score_change_for_realert
    credit_ok = abs(candidate.credit - prev["credit"]) >= config.min_credit_change_for_realert
    if time_ok and (score_ok or credit_ok):
        seen_alerts[key] = {"time": now_ts, "score": candidate.score, "credit": candidate.credit}
        return True
    return False


def run_one_scan(config: Config, symbols: List[str], seen_alerts: Dict) -> None:
    """One pass: scan for entries during RTH, monitor live market + tracked spreads overnight."""
    resolve_forecast_log(config)
    is_rth = within_market_hours(config.market_timezone)
    if not is_rth:
        if config.overnight_monitor_enabled:
            monitor_tracked_ideas(config, symbols)
            return
        if config.regular_hours_only:
            stamp = now_local(config.market_timezone).strftime("%Y-%m-%d %H:%M:%S %Z")
            print(f"[{stamp}] Outside regular market hours. Overnight monitor disabled; skipping this pass.")
            return
    if config.late_day_only and is_rth and not _in_clock_window(
        now_local(config.market_timezone), config.entry_window_start, config.entry_window_end
    ):
        # During the first minutes after the bell, the user's tracked overnight idea may
        # still be open even though we do not want to search for new entries. Continue
        # monitoring it until the configured intended exit/grace period instead of going blind.
        if config.overnight_monitor_enabled:
            monitor_tracked_ideas(config, symbols)
            return
        stamp = now_local(config.market_timezone).strftime("%Y-%m-%d %H:%M:%S %Z")
        print(
            f"[{stamp}] Outside configured entry window "
            f"({config.entry_window_start}-{config.entry_window_end} PT). Skipping entry scan."
        )
        return

    stamp = now_local(config.market_timezone).strftime("%Y-%m-%d %H:%M:%S %Z")
    print("\n" + divider("="))
    print(f"SCAN START | {stamp}")
    print(divider("="))

    results: List[Tuple[str, MarketState, SpreadCandidate]] = []
    for symbol in symbols:
        try:
            results.extend(scan_symbol(symbol, config))
        except Exception as exc:
            print(f"[warn] {symbol}: scan failed: {exc}", file=sys.stderr)

    if not results:
        print(divider("="))
        print("SCAN RESULT: No alertable candidates this scan.")
        print(divider("="))
        return

    sent = False
    for symbol, state, candidate in results:
        if should_alert(seen_alerts, candidate, symbol, config):
            send_alert(config, candidate, symbol, state)
            track_alerted_idea(symbol, state, candidate, config)
            sent = True
    if not sent:
        print(divider("="))
        print("SCAN RESULT: Candidates were found, but all are inside cooldown / re-alert thresholds.")
        print(divider("="))


def main() -> None:
    run_once = os.getenv("RUN_ONCE", "false").lower() == "true" or "--once" in sys.argv

    config = Config.from_env()
    symbols = [s.strip().upper() for s in config.symbols.split(",") if s.strip()]
    # v13 is intentionally SPY-only so the next-open consensus can be specialized.
    symbols = ["SPY"]

    print("\n" + divider("="))
    print("SPY OVERNIGHT CREDIT SPREAD BOT v13 - UNIFIED NEXT-OPEN CONSENSUS")
    print(divider("="))
    print(f"Symbols           : {', '.join(symbols)}")
    print(f"DTE range         : {config.target_dte_min}-{config.target_dte_max}")
    print(f"Short delta range : {config.short_delta_min:.2f}-{config.short_delta_max:.2f}")
    print(f"Min consensus POP : {config.min_consensus_pop:.0%}")
    print(f"Min morning P(profit): {config.min_morning_profit_prob:.0%} (IV stress x{config.morning_iv_stress_mult:.2f})")
    print(
        "Consensus weights : "
        f"hist {config.consensus_general_weight:.0%} | large-day {config.consensus_large_day_weight:.0%} | "
        f"late tape {config.consensus_late_tape_weight:.0%} | VIX prior {config.consensus_vix_regime_weight:.0%} | "
        f"technical {config.consensus_technical_weight:.0%}"
    )
    print(f"S/R filter mode     : {config.sr_filter_mode} (override morning>={config.sr_override_min_morning_profit_prob:.0%}, open>={config.sr_override_min_open_survival:.0%})")
    print(f"Near-miss display   : {config.show_near_miss_candidates} (max {config.max_near_miss_candidates}, up to {config.near_miss_max_failed_filters} misses)")
    print(f"Overnight engine  : {config.overnight_eligibility_enabled} (FAIL blocks={config.block_overnight_fail_alerts})")
    print(f"Overnight monitor : {config.overnight_monitor_enabled} | track alerts={config.track_alerted_ideas}")
    print(f"Market watch      : {config.always_track_market_overnight} | bootstrap after close={config.bootstrap_market_watch_after_close}")
    print(f"Live thesis rules : caution <{config.live_forecast_caution_prob:.0%} favorable | invalidated <{config.live_forecast_invalidate_prob:.0%}")
    print(f"Entry window      : {config.entry_window_start}-{config.entry_window_end} PT | intended exit {config.intended_exit_time} PT")
    print(f"Event guard       : {config.event_guard_enabled} ({config.event_guard_mode})")
    print(f"Neutral ideas     : {config.allow_neutral_directional_ideas}")
    print(f"Regular hours only: {config.regular_hours_only}")
    print(f"Run mode          : {'single pass (--once / RUN_ONCE)' if run_once else f'internal loop, {config.scan_interval_seconds}s interval'}")
    print(divider("=") + "\n")

    seen_alerts: Dict[Tuple[str, str, str, float, float], Dict[str, float]] = {}

    if run_once:
        # Designed for an external scheduler (GitHub Actions cron, Heroku Scheduler,
        # a VPS cron job) to invoke this process repeatedly instead of paying for a
        # dyno/VPS that sits idle in an internal sleep() loop 24/7. Cooldown/re-alert
        # dedupe (seen_alerts) is in-memory only and does not persist between separate
        # --once invocations, so alert_cooldown_seconds effectively has no effect
        # across runs in this mode -- each run is independent.
        try:
            run_one_scan(config, symbols, seen_alerts)
        except Exception as exc:
            print(f"[error] {exc}", file=sys.stderr)
            sys.exit(1)
        return

    while True:
        try:
            run_one_scan(config, symbols, seen_alerts)
        except KeyboardInterrupt:
            print("\nStopping bot.")
            return
        except Exception as exc:
            print(f"[error] {exc}", file=sys.stderr)

        print(f"Next scan in {config.scan_interval_seconds}s.")
        time.sleep(config.scan_interval_seconds)


if __name__ == "__main__":
    main()
