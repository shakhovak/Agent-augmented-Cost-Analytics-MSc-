from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cost_agent_mvp.core.errors import ValidationError
from cost_agent_mvp.data.schema import (
    coerce_types,
    default_joint_schema,
    validate_columns,
)
from data.service_costs import add_services_and_costs_v2
from data.users_activity import build_service_user_state

# ----------------------------
# Config loading
# ----------------------------


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValidationError(f"Config at {p} must be a YAML mapping.")
    return data


# ----------------------------
# Helpers
# ----------------------------

_DOW_KEYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def date_range_days(start: str, end: str) -> list[dt.date]:
    start_d = dt.date.fromisoformat(start)
    end_d = dt.date.fromisoformat(end)
    if end_d < start_d:
        raise ValidationError("end_date must be >= start_date.")
    days = (end_d - start_d).days + 1
    return [start_d + dt.timedelta(days=i) for i in range(days)]


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def sample_unique_ints(
    rng: np.random.Generator,
    n: int,
    low: int,
    high: int,
) -> np.ndarray:
    # high is inclusive in config; numpy's integers high is exclusive
    if high <= low:
        raise ValidationError("account_id_max must be > account_id_min.")
    pool_size = high - low + 1
    if n > pool_size:
        raise ValidationError("Cannot sample n_accounts unique IDs from the given range.")
    return rng.choice(np.arange(low, high + 1, dtype=np.int64), size=n, replace=False)


def sample_chat_ids(
    rng: np.random.Generator,
    n: int,
    digits: int,
) -> list[str]:
    # 11-digit numeric strings, allow leading zeros
    # Example range: 0..(10^digits - 1)
    max_val = 10**digits
    vals = rng.integers(0, max_val, size=n, endpoint=False)
    return [f"{v:0{digits}d}" for v in vals]


def dow_key(d: dt.date) -> str:
    # Monday=0 .. Sunday=6
    return _DOW_KEYS[d.weekday()]


def lognormal_from_median(
    rng: np.random.Generator, median: float, sigma: float, size: int
) -> np.ndarray:
    if median <= 0:
        raise ValidationError("Median must be > 0 for lognormal.")
    mu = float(np.log(median))
    return rng.lognormal(mean=mu, sigma=float(sigma), size=size)


# ----------------------------
# Generation steps
# ----------------------------


@dataclass
class ChatState:
    chat_id: str
    chat_type: str
    weight: float
    remaining_life: int  # days left in active pool


def generate_accounts(cfg: dict[str, Any]) -> list[int]:
    rng = make_rng(int(cfg["dataset"]["seed"]))
    n_accounts = int(cfg["accounts"]["n_accounts"])
    a_min = int(cfg["accounts"]["account_id_min"])
    a_max = int(cfg["accounts"]["account_id_max"])
    return sample_unique_ints(rng, n=n_accounts, low=a_min, high=a_max).tolist()


def sample_lifetime_days(rng: np.random.Generator, cfg: dict[str, Any]) -> int:
    life_cfg = cfg["chat_lifecycle"]["lifetime"]
    dist = str(life_cfg["dist"]).lower()
    mean_days = float(life_cfg["mean_days"])

    if mean_days <= 1:
        return 1

    if dist == "geometric":
        # geometric over {1,2,...} with mean = 1/p  => p = 1/mean
        p = 1.0 / mean_days
        return int(rng.geometric(p))
    else:
        raise ValidationError(f"Unsupported lifetime dist: {dist}")


def create_new_chat(
    rng: np.random.Generator,
    cfg: dict[str, Any],
) -> ChatState:
    digits = int(cfg["chats"]["chat_id_digits"])
    chat_types: list[str] = list(cfg["chats"]["chat_types"])

    chat_id = sample_chat_ids(rng, n=1, digits=digits)[0]
    chat_type = str(rng.choice(chat_types))
    # popularity weight gives "favorite chats" but not immortal anymore
    sigma = float(cfg["activity"]["chat_popularity_lognorm_sigma"])
    weight = float(rng.lognormal(mean=0.0, sigma=sigma))
    life = sample_lifetime_days(rng, cfg)
    return ChatState(chat_id=chat_id, chat_type=chat_type, weight=weight, remaining_life=life)


def sample_active_chat_count(
    rng: np.random.Generator,
    m_total: int,
    d: dt.date,
    cfg: dict[str, Any],
) -> int:
    p_full = float(cfg["activity"]["p_full_day"])
    base_q = float(cfg["activity"]["base_q"])
    q_min = float(cfg["activity"]["q_clip_min"])
    q_max = float(cfg["activity"]["q_clip_max"])

    dow_mult = cfg["activity"]["dow_multiplier"]
    month_mult = cfg["activity"]["month_multiplier"]

    dk = dow_key(d)
    m_mult = float(month_mult[int(d.month)])
    d_mult = float(dow_mult[dk])

    q_t = base_q * m_mult * d_mult
    q_t = float(np.clip(q_t, q_min, q_max))

    if rng.random() < p_full:
        return m_total

    # Typical day: Binomial
    k = int(rng.binomial(n=m_total, p=q_t))
    if m_total > 0 and k == 0:
        k = 1
    # allow 0 active chats
    return k


def simulate_active_chat_days(cfg: dict[str, Any]) -> pd.DataFrame:
    rng = make_rng(int(cfg["dataset"]["seed"]))
    days = date_range_days(cfg["dataset"]["start_date"], cfg["dataset"]["end_date"])

    accounts = generate_accounts(cfg)
    max_concurrent = int(cfg["chats"]["max_concurrent_per_account"])

    lifecycle_enabled = bool(cfg["chat_lifecycle"]["enabled"])
    lam_new = float(cfg["chat_lifecycle"]["new_chats_lambda_per_account_day"])
    p_reactivate = float(cfg["chat_lifecycle"].get("reactivation_prob", 0.0))

    # Per-account state
    active: dict[int, dict[str, ChatState]] = {acc: {} for acc in accounts}
    inactive: dict[int, dict[str, ChatState]] = {acc: {} for acc in accounts}

    init_cfg = cfg["chat_lifecycle"]["initial_active_chats_per_account"]
    init_mean = float(init_cfg["mean"])
    init_min = int(init_cfg.get("min", 0))
    init_max = int(init_cfg.get("max", max_concurrent))

    for acc in accounts:
        k0 = int(rng.poisson(init_mean))
        k0 = max(init_min, min(init_max, k0))
        for _ in range(k0):
            if len(active[acc]) >= max_concurrent:
                break
            st = create_new_chat(rng, cfg)
            while st.chat_id in active[acc] or st.chat_id in inactive[acc]:
                st = create_new_chat(rng, cfg)
            active[acc][st.chat_id] = st

    records: list[dict[str, Any]] = []

    for d in days:
        for acc in accounts:
            # 1) Age out existing active chats
            to_expire: list[str] = []
            for cid, st in active[acc].items():
                st.remaining_life -= 1
                if st.remaining_life <= 0:
                    to_expire.append(cid)
            for cid in to_expire:
                inactive[acc][cid] = active[acc].pop(cid)

            # 2) New chats arrive (Poisson), only if lifecycle enabled
            if lifecycle_enabled:
                # You can optionally apply weekend/summer to arrival rate too:
                # lam_t = lam_new * month_multiplier * dow_multiplier
                dk = dow_key(d)
                dow_mult = float(cfg["activity"]["dow_multiplier"][dk])
                month_mult = float(cfg["activity"]["month_multiplier"][int(d.month)])
                lam_t = lam_new * dow_mult * month_mult

                n_new = int(rng.poisson(lam_t))
                for _ in range(n_new):
                    if len(active[acc]) >= max_concurrent:
                        break
                    st = create_new_chat(rng, cfg)
                    # ensure uniqueness within account (rare collision, but handle)
                    while st.chat_id in active[acc] or st.chat_id in inactive[acc]:
                        st = create_new_chat(rng, cfg)
                    active[acc][st.chat_id] = st

            # 3) Optional reactivation of inactive chats (rare)
            if p_reactivate > 0 and inactive[acc]:
                if rng.random() < p_reactivate and len(active[acc]) < max_concurrent:
                    # Reactivate one random inactive chat
                    cid = str(rng.choice(list(inactive[acc].keys())))
                    st = inactive[acc].pop(cid)
                    # refresh lifetime on reactivation
                    st.remaining_life = sample_lifetime_days(rng, cfg)
                    active[acc][cid] = st

            # 4) Pick daily active subset from currently active pool
            chats_today = list(active[acc].values())
            m_total = len(chats_today)
            if m_total == 0:
                continue

            k = sample_active_chat_count(rng, m_total=m_total, d=d, cfg=cfg)
            if k <= 0:
                continue
            if k >= m_total:
                chosen = chats_today
            else:
                weights = np.array([c.weight for c in chats_today], dtype=float)
                probs = weights / weights.sum()
                idx = rng.choice(m_total, size=k, replace=False, p=probs)
                chosen = [chats_today[int(i)] for i in idx.tolist()]

            # 5) Emit rows
            for st in chosen:
                records.append(
                    {
                        "account_id": int(acc),
                        "chat_id": st.chat_id,
                        "chat_type": st.chat_type,
                        "date": d.isoformat(),
                    }
                )

    return pd.DataFrame.from_records(records)


def generate_joint_dataset(cfg: dict) -> pd.DataFrame:
    base = simulate_active_chat_days(cfg)
    accounts = sorted(base["account_id"].unique().tolist())
    rng = np.random.default_rng(cfg["dataset"]["seed"] + 9)

    user_state = build_service_user_state(accounts, cfg, rng)

    end = dt.date.fromisoformat(cfg["dataset"]["end_date"])
    for svc in ["tasks", "classifications", "amocrm_call"]:
        deltas = []
        for s in user_state[svc].values():
            start = max(dt.date.fromisoformat(cfg["dataset"]["start_date"]), s.adoption_date)
            stop = end if s.churn_date is None else min(end, s.churn_date)
            deltas.append((stop - start).days + 1)
        deltas = sorted(deltas)
        print(svc, "smallest 10 windows:", deltas[:10], "largest 5:", deltas[-5:])

        # --- DEBUG: adopters + intensity stats (Step 2 output)
        print("ADOPTERS (from user_state):")
        for svc in ["tasks", "classifications", "amocrm_call"]:
            print(svc, len(user_state.get(svc, {})))

    print("INTENSITY stats:")
    for svc in ["tasks", "classifications", "amocrm_call"]:
        intens = np.array([float(s.intensity) for s in user_state.get(svc, {}).values()])
        if len(intens) == 0:
            continue
        print(
            svc,
            "mean=",
            float(intens.mean()),
            "median=",
            float(np.median(intens)),
            "p90=",
            float(np.percentile(intens, 90)),
            "max=",
            float(intens.max()),
        )

    full = add_services_and_costs_v2(base, cfg, user_state)

    # --- DEBUG: account-day usage summary (after Step 3)
    acc_day = full.groupby(["account_id", "date"], as_index=False)[
        ["has_tasks", "has_classifications", "has_amocrm_call", "num_services"]
    ].max()

    print("ACCOUNT-DAY num_services distribution:")
    print(acc_day["num_services"].value_counts(normalize=True).sort_index())

    print("ACCOUNT-DAY service-day counts:")
    print("tasks days:", int((acc_day["has_tasks"] == 1).sum()))
    print("classifications days:", int((acc_day["has_classifications"] == 1).sum()))
    print("calls days:", int((acc_day["has_amocrm_call"] == 1).sum()))

    # Validate against schema (required columns + coercion)
    schema = default_joint_schema()
    validate_columns(full, schema)
    full = coerce_types(full, schema)
    return full


def apply_cli_overrides(cfg: dict, args) -> dict:
    cfg = dict(cfg)  # shallow copy is fine if you only mutate a few leaves

    if args.seed is not None:
        cfg["dataset"]["seed"] = int(args.seed)

    if args.n_accounts is not None:
        cfg["accounts"]["n_accounts"] = int(args.n_accounts)

    if args.days is not None:
        start = dt.date.fromisoformat(cfg["dataset"]["start_date"])
        end = start + dt.timedelta(days=int(args.days) - 1)
        cfg["dataset"]["end_date"] = end.isoformat()

    if args.out is not None:
        cfg.setdefault("output", {})
        cfg["output"]["path"] = str(args.out)

    return cfg


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic joint usage-cost dataset.")
    parser.add_argument("--config", type=str, default="configs/synth_2025.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override dataset.seed")
    parser.add_argument(
        "--days", type=int, default=None, help="Override horizon length in days (from start_date)"
    )
    parser.add_argument("--n-accounts", type=int, default=None, help="Override accounts.n_accounts")
    parser.add_argument("--out", type=Path, default=None, help="Override output.path")

    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_cli_overrides(cfg, args)
    out_path = Path(cfg["output"]["path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_joint_dataset(cfg)
    df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} | rows={len(df):,} | cols={len(df.columns)}")


if __name__ == "__main__":
    main()
