#!/usr/bin/env python3
"""Exact local port of the make.fun bars-replay/v1 harness.

Re-implements sim/bars/engine.ts (BarsEngine + BarsPool + fill math) and the
bars half of sim/metrics.ts (computeBarsMetrics) in stdlib-only Python, so a
bot can be backtested on the frozen dev dataset without Docker.

Every amount is an integer (micro-USDC / scaled ints); every division
truncates toward zero, exactly like the harness's BigInt `/`. Python's `//`
floors, so all divisions go through `tdiv`.

Usage:
    python3 tools/replay.py --kit <kit-dir> --bot <bot-dir-or-main.py>
        [--fee-multiplier 1] [--out run.json]

The bot is driven as a subprocess exactly like the real runtime:
`python3 -P -s <main.py>` with PYTHONHASHSEED=0, NDJSON on stdin/stdout.
After the 1x run the recorded action log is replayed at 2x fees (with the
clampOrders set the harness uses) for the cost-sensitivity line.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

BPS = 10_000
SCALE9 = 1_000_000_000
SCALE18 = SCALE9 * SCALE9
USDC_DECIMALS = 6


def tdiv(a: int, b: int) -> int:
    """Truncating integer division (BigInt `/` semantics, not Python `//`)."""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


# ---------------------------------------------------------------- fill math


def bars_value(q: int, p: int, quote_scale: int, price_scale: int, qty_scale: int) -> int:
    return tdiv(q * p * 10**quote_scale, 10 ** (qty_scale + price_scale))


class BarsPool:
    """Constant-product pool of depth D = prod/sum, state (yn/yd, kn/kd).

    Mirrors sim/bars/engine.ts BarsPool exactly: the constructor gcd-reduces
    yn/yd but keeps kn/kd as given; amounts are scaled by 10^6 and depths by
    10^quote_scale, so both count 10^-(quote_scale + 6) USDC.
    """

    def __init__(self, yn: int, yd: int, kn: int, kd: int, unit: int = 10**USDC_DECIMALS):
        g = math.gcd(yn, yd)
        self.yn = yn // g
        self.yd = yd // g
        self.kn = kn
        self.kd = kd
        self.unit = unit

    @classmethod
    def of(cls, depth: list[int], quote_scale: int) -> "BarsPool":
        if not depth or any(d < 1 for d in depth):
            raise ValueError(f"a fill needs a depth above 0 on every hop, got {depth}")
        d = [x * 10**quote_scale for x in depth]
        prod = 1
        for x in d:
            prod *= x
        total = 0
        for i in range(len(d)):
            comp = 1
            for j, x in enumerate(d):
                if j != i:
                    comp *= x
            total += comp
        g = math.gcd(prod, total)
        return cls(prod // g, total // g, (prod // g) ** 2, (total // g) ** 2)

    def buy(self, spend: int):
        yn2 = self.yn + spend * self.unit * self.yd
        num = self.yn * yn2 * self.kd
        den = self.yd * self.yd * self.kn
        return num, den, BarsPool(yn2, self.yd, self.kn, self.kd, self.unit)

    def sell(self, value: int):
        t = self.kn * self.yd + value * self.unit * self.yn * self.kd
        num = t * self.yd
        den = self.kd * self.yn * self.yn
        return num, den, BarsPool(self.kn * self.yn, t, self.kn, self.kd, self.unit)


def bars_buy_fill(notional, open_price, pool, fee_bps, slip_bps, scales):
    """Returns (price, spend, fee, qty, after_pool)."""
    quote_scale, price_scale, qty_scale = scales
    if not isinstance(pool, BarsPool):
        pool = BarsPool.of(pool, quote_scale)
    spend = tdiv(notional * (BPS - fee_bps), BPS)
    num, den, after = pool.buy(spend)
    price = tdiv(open_price * (BPS + slip_bps) * num, BPS * den)
    qty = tdiv(spend * 10 ** (qty_scale + price_scale), price * 10**quote_scale)
    return price, spend, notional - spend, qty, after


def bars_sell_fill(qty, open_price, pool, fee_bps, slip_bps, scales):
    """Returns (price, gross, fee, proceeds, after_pool)."""
    quote_scale, price_scale, qty_scale = scales
    if not isinstance(pool, BarsPool):
        pool = BarsPool.of(pool, quote_scale)
    value = bars_value(qty, open_price, quote_scale, price_scale, qty_scale)
    num, den, after = pool.sell(value)
    slipped = tdiv(open_price * (BPS - slip_bps) * den, BPS * num)
    price = slipped if slipped >= 1 else 1
    gross = bars_value(qty, price, quote_scale, price_scale, qty_scale)
    proceeds = tdiv(gross * (BPS - fee_bps), BPS)
    return price, gross, gross - proceeds, proceeds, after


# ---------------------------------------------------------------- metrics


def isqrt(n: int) -> int:
    return math.isqrt(n)  # floor sqrt, like the harness's Newton isqrt


def ratio9(a: int, b: int) -> int:
    return tdiv(a * SCALE9, b)


def period_returns(marks: list[int]) -> list[int]:
    out = []
    for i in range(1, len(marks)):
        prev = marks[i - 1]
        out.append(0 if prev == 0 else ratio9(marks[i] - prev, prev))
    return out


def sharpe9(r: list[int]):
    n = len(r)
    if n < 2:
        return None
    s = sum(r)
    s2 = sum(x * x for x in r)
    sd18 = isqrt(tdiv((n * s2 - s * s) * SCALE18, n * (n - 1)))
    if sd18 == 0:
        return None
    return tdiv(s * SCALE18, n * sd18)


def sortino9(r: list[int]):
    n = len(r)
    if n < 2:
        return None
    s = sum(r)
    d2 = sum(x * x for x in r if x < 0)
    dd18 = isqrt(tdiv(d2 * SCALE18, n))
    if dd18 == 0:
        return None
    return tdiv(s * SCALE18, n * dd18)


def max_drawdown9(equity: list[int]) -> int:
    peak = 0
    worst = 0
    for e in equity:
        if e > peak:
            peak = e
        if peak > 0:
            dd = ratio9(peak - e, peak)
            if dd > worst:
                worst = dd
    return worst


def fmt9(x: int) -> str:
    neg = x < 0
    a = -x if neg else x
    return f"{'-' if neg else ''}{a // SCALE9}.{a % SCALE9:09d}"


def fmt_usdc(units: int) -> str:
    return f"{units / 1_000_000:.6f}"


# ---------------------------------------------------------------- dataset


def load_dataset(kit_dir: Path):
    manifest = json.loads((kit_dir / "dataset" / "manifest.json").read_text())
    sym = manifest["symbols"][0]
    bars = []
    with gzip.open(kit_dir / "dataset" / "bars" / "CBBTC.ndjson.gz", "rt") as f:
        for line in f:
            d = json.loads(line)
            bars.append(
                {
                    "t": d["t"],
                    "o": int(d["o"]),
                    "h": int(d["h"]),
                    "l": int(d["l"]),
                    "c": int(d["c"]),
                    "v": d["v"],
                    "o_depth": [int(x) for x in d["o_depth"]],
                    "c_depth": [int(x) for x in d["c_depth"]],
                    "wire": {
                        "symbol": "CBBTC",
                        "t": d["t"],
                        "o": d["o"],
                        "h": d["h"],
                        "l": d["l"],
                        "c": d["c"],
                        "v": d["v"],
                        "o_depth": [str(x) for x in d["o_depth"]],
                        "c_depth": [str(x) for x in d["c_depth"]],
                    },
                }
            )
    window = manifest["window"]
    return {
        "sha256": manifest["dataset_sha256"],
        "window": window,
        "bars": bars,
        "price_scale": sym["price_scale"],
        "qty_scale": sym["qty_scale"],
        "mint": sym["mint"],
    }


# ---------------------------------------------------------------- engine


class SimAbort(Exception):
    pass


class BarsEngine:
    """Port of sim/bars/engine.ts BarsEngine (frozen dataset, warmup 0)."""

    def __init__(self, dataset, fee_multiplier=1, clamp_orders=frozenset()):
        self.dataset = dataset
        self.window = dataset["window"]
        self.step = 60_000
        self.first_close = self.window["start_ms"] + self.step
        self.batch_count = self.window["bars"]
        self.warm = 0
        self.fee_bps = min(4 * fee_multiplier, BPS)
        self.slip_bps = 2
        self.min_notional = 1_000_000
        self.quote_scale = 6
        self.scales = (6, dataset["price_scale"], dataset["qty_scale"])
        self.caps = {"max_open_positions": 1, "max_orders_per_bar": 16, "max_weight_bps": 10_000}
        self.clamp = clamp_orders
        self.start_cash = 1_000_000_000
        self.cash = self.start_cash
        self.seq = 0
        self.phase = "ready"
        self.aborted = None
        self.by_t = {b["t"]: b for b in dataset["bars"]}
        # per-symbol runtime state (single symbol: CBBTC)
        self.sym = {
            "symbol": "CBBTC",
            "close": None,
            "close_depth": [],
            "qty": 0,
            "cost": 0,
            "pnl": 0,
            "trades": 0,
            "fees": 0,
            "open_rt": None,
        }
        self.pending = []
        self.used_ids = set()
        self.resolutions = []
        self.undelivered = []
        self.equity = []
        self.round_trips = []
        self.replies = []
        self.close_counter = 0

    # -- helpers ------------------------------------------------------
    def value(self, q, p):
        return bars_value(q, p, *self.scales)

    def equity_now(self):
        e = self.cash
        if self.sym["qty"] > 0:
            e += self.value(self.sym["qty"], self.sym["close"])
        return e

    def account(self):
        s = self.sym
        pos = []
        if s["qty"] > 0:
            pos.append({"symbol": "CBBTC", "qty": str(s["qty"]), "cost_quote_units": str(s["cost"])})
        return {
            "cash_quote_units": str(self.cash),
            "positions": pos,
            "equity_quote_units": str(self.equity_now()),
        }

    def flow(self, cash_delta, fee):
        s = self.sym
        s["pnl"] += cash_delta
        s["fees"] += fee
        if s["open_rt"] is not None:
            s["open_rt"]["pnl"] += cash_delta

    def close_round_trip(self, bar_t):
        rt = self.sym["open_rt"]
        if rt is None:
            raise SimAbort("emptied with no open round trip")
        self.round_trips.append(
            {"symbol": "CBBTC", "opened_bar": rt["opened_bar"], "closed_bar": bar_t, "pnl_quote_units": str(rt["pnl"])}
        )
        self.sym["open_rt"] = None

    def base_fill(self, p):
        return {
            "client_id": p["order"]["client_id"],
            "symbol": p["order"]["symbol"],
            "side": p["order"]["side"],
            "status": "rejected",
            "reason": None,
            "decided_bar": p["decided_bar"],
            "fill_bar": p["fill_bar"],
            "price": "0",
            "qty": "0",
            "notional_quote_units": "0",
            "fee_quote_units": "0",
            "kind": "order",
            "seq": p["seq"],
            "index": p["index"],
            "rejected_at": None,
            "position_before_qty": str(self.sym["qty"]),
            "cash_after_quote_units": str(self.cash),
        }

    def record(self, fill, deliver=True):
        self.resolutions.append(fill)
        if not deliver:
            return
        stripped = {k: v for k, v in fill.items() if k not in ("kind", "seq", "index", "rejected_at", "position_before_qty", "cash_after_quote_units")}
        self.undelivered.append(stripped)

    def record_reject(self, p, reason, at, deliver=True):
        f = self.base_fill(p)
        f["reason"] = reason
        f["rejected_at"] = at
        self.record(f, deliver)

    # -- stepping ------------------------------------------------------
    def step_to(self, k):
        t = self.first_close + k * self.step
        if k == self.warm:
            self.equity.append({"seq": self.warm - 1, "t": self.window["start_ms"], "cash_quote_units": str(self.cash), "equity_quote_units": str(self.cash)})
        bar = self.by_t.get(t)
        pools = {}
        for p in self.pending:
            self.resolve(p, bar, pools)
        self.pending = []
        if bar is not None:
            self.sym["close"] = bar["c"]
            self.sym["close_depth"] = bar["c_depth"]
        bars_wire = [bar["wire"]] if bar is not None else []
        fills = self.undelivered
        self.undelivered = []
        account = self.account()
        if k >= self.warm:
            self.equity.append({"seq": k, "t": t, "cash_quote_units": account["cash_quote_units"], "equity_quote_units": account["equity_quote_units"]})
        return {"type": "batch", "seq": k, "t": t, "bars": bars_wire, "fills": fills, "account": account}

    def next_batch(self):
        if self.phase == "awaiting_reply":
            raise SimAbort("batch has no reply yet")
        if self.phase == "finished":
            raise SimAbort("replay is finished")
        if self.aborted is not None or self.seq >= self.batch_count:
            return None
        batch = self.step_to(self.seq)
        self.phase = "awaiting_reply"
        return batch

    # -- deciding ------------------------------------------------------
    def decide(self, reply):
        caps = self.caps
        t = self.first_close + self.seq * self.step
        is_last = self.seq == self.batch_count - 1
        equity = self.equity_now()
        buys = 0
        bought_by = {}
        sold_by = {}
        open_set = {"CBBTC"} if self.sym["qty"] > 0 else set()
        for index, order in enumerate(reply["orders"]):
            side = order["side"]
            written = int(order["notional_quote_units"]) if side == "buy" else int(order["qty"])
            clamp = f"{self.seq}:{index}" in self.clamp
            amount = written
            reject = None
            s = self.sym if order["symbol"] == "CBBTC" else None
            if index >= caps["max_orders_per_bar"]:
                reject = "max_orders_per_bar"
            elif order["client_id"] in self.used_ids:
                reject = "duplicate_client_id"
            elif s is None:
                reject = "not_in_universe"
            elif written == 0:
                reject = "zero_amount"
            elif self.seq < self.warm:
                reject = "warmup"
            elif is_last:
                reject = "after_end"
            elif s["close"] is None:
                reject = "no_bar"
            elif side == "buy":
                cash_left = self.cash - buys
                held_value = self.value(s["qty"], s["close"])
                bought = bought_by.get("CBBTC", 0)
                if clamp:
                    room = tdiv(caps["max_weight_bps"] * equity, BPS) - held_value - bought
                    bound = min(cash_left, room)
                    if amount > bound:
                        amount = bound if bound > 0 else 0
                if amount == 0:
                    reject = "insufficient_cash" if cash_left <= 0 else "max_weight"
                elif amount < self.min_notional:
                    reject = "below_min_notional"
                elif amount > cash_left:
                    reject = "insufficient_cash"
                elif "CBBTC" not in open_set and len(open_set) >= caps["max_open_positions"]:
                    reject = "max_open_positions"
                elif (held_value + bought + amount) * BPS > caps["max_weight_bps"] * equity:
                    reject = "max_weight"
                if reject is None:
                    buys += amount
                    bought_by["CBBTC"] = bought + amount
                    open_set.add("CBBTC")
            else:  # sell
                sold = sold_by.get("CBBTC", 0)
                available = s["qty"] - sold
                if clamp and amount > available:
                    amount = available if available > 0 else 0
                if amount == 0 or amount > available:
                    reject = "insufficient_position"
                elif amount != available and self.value(amount, s["close"]) < self.min_notional:
                    reject = "below_min_notional"
                if reject is None:
                    sold_by["CBBTC"] = sold + amount
            self.used_ids.add(order["client_id"])
            self.pending.append(
                {
                    "order": order,
                    "seq": self.seq,
                    "index": index,
                    "decided_bar": t,
                    "fill_bar": t + self.step,
                    "reject": reject,
                    "amount": amount,
                    "close": s["close"] if s is not None else None,
                }
            )

    # -- resolving ------------------------------------------------------
    def resolve(self, p, bar, pools):
        if p["reject"] is not None:
            self.record_reject(p, p["reject"], "decision")
            return
        if bar is None:
            self.record_reject(p, "no_bar", "fill")
            return
        s = self.sym
        close = p["close"]
        max_slip = int(p["order"]["max_slippage_bps"])
        bps = (self.fee_bps, self.slip_bps)
        pool = pools.get("CBBTC")
        if pool is None:
            pool = BarsPool.of(bar["o_depth"], self.quote_scale)
        before = s["qty"]
        if p["order"]["side"] == "buy":
            price, spend, fee, qty, after = bars_buy_fill(p["amount"], bar["o"], pool, *bps, self.scales)
            if price * BPS > close * (BPS + max_slip):
                self.record_reject(p, "max_slippage", "fill")
                return
            if qty == 0:
                self.record_reject(p, "zero_amount", "fill")
                return
            if spend + fee != p["amount"]:
                raise SimAbort(f"buy spend {spend} + fee {fee} != notional {p['amount']}")
            pools["CBBTC"] = after
            self.cash -= p["amount"]
            if self.cash < 0:
                raise SimAbort(f"cash went negative ({self.cash})")
            if s["qty"] == 0:
                s["open_rt"] = {"pnl": 0, "opened_bar": bar["t"]}
            s["qty"] += qty
            s["cost"] += p["amount"]
            s["trades"] += 1
            self.flow(-p["amount"], fee)
            f = self.base_fill(p)
            f.update(
                {
                    "status": "filled",
                    "price": str(price),
                    "qty": str(qty),
                    "notional_quote_units": str(p["amount"]),
                    "fee_quote_units": str(fee),
                    "position_before_qty": str(before),
                    "cash_after_quote_units": str(self.cash),
                }
            )
            self.record(f, True)
            return
        price, gross, fee, proceeds, after = bars_sell_fill(p["amount"], bar["o"], pool, *bps, self.scales)
        if price * BPS < close * (BPS - max_slip):
            self.record_reject(p, "max_slippage", "fill")
            return
        if p["amount"] > s["qty"]:
            raise SimAbort(f"sell {p['amount']} exceeds {s['qty']} held")
        pools["CBBTC"] = after
        self.cash += proceeds
        s["cost"] = tdiv(s["cost"] * (s["qty"] - p["amount"]), s["qty"])
        s["qty"] -= p["amount"]
        s["trades"] += 1
        self.flow(proceeds, fee)
        if s["qty"] == 0:
            self.close_round_trip(bar["t"])
        f = self.base_fill(p)
        f.update(
            {
                "status": "filled",
                "price": str(price),
                "qty": str(p["amount"]),
                "notional_quote_units": str(proceeds),
                "fee_quote_units": str(fee),
                "position_before_qty": str(before),
                "cash_after_quote_units": str(self.cash),
            }
        )
        self.record(f, True)

    def flatten(self, last_bar):
        s = self.sym
        qty = s["qty"]
        price, gross, fee, proceeds, _after = bars_sell_fill(qty, s["close"], s["close_depth"], self.fee_bps, self.slip_bps, self.scales)
        self.cash += proceeds
        s["qty"] = 0
        s["cost"] = 0
        self.flow(proceeds, fee)
        self.close_round_trip(last_bar)
        self.record(
            {
                "client_id": f"_flatten_{self.close_counter}",
                "symbol": "CBBTC",
                "side": "sell",
                "status": "filled",
                "reason": None,
                "decided_bar": last_bar,
                "fill_bar": last_bar,
                "price": str(price),
                "qty": str(qty),
                "notional_quote_units": str(proceeds),
                "fee_quote_units": str(fee),
                "kind": "flatten",
                "seq": None,
                "index": None,
                "rejected_at": None,
                "position_before_qty": str(qty),
                "cash_after_quote_units": str(self.cash),
            },
            deliver=False,
        )
        self.close_counter += 1

    def finish(self):
        if self.phase == "awaiting_reply":
            raise SimAbort("batch has no reply yet")
        if self.phase == "finished":
            raise SimAbort("replay is finished")
        if self.aborted is None and self.seq < self.batch_count:
            raise SimAbort(f"finish() before the last batch ({self.seq} of {self.batch_count})")
        self.phase = "finished"
        last_bar = self.first_close + (self.batch_count - 1) * self.step
        if self.aborted is None:
            for p in self.pending:
                if p["reject"] is None:
                    raise SimAbort(f"order {p['order']['client_id']} of the last batch passed its decision checks")
                self.record_reject(p, p["reject"], "decision", deliver=False)
            self.pending = []
            if self.sym["qty"] > 0:
                self.flatten(last_bar)
            self.equity.append({"seq": self.seq, "t": last_bar, "cash_quote_units": str(self.cash), "equity_quote_units": str(self.cash)})
            # checkTotals
            if self.cash < 0:
                raise SimAbort(f"cash ended negative ({self.cash})")
            pnl = self.sym["pnl"]
            if self.sym["qty"] != 0 or self.sym["open_rt"] is not None:
                raise SimAbort("still held after flattening")
            if pnl != self.cash - self.start_cash:
                raise SimAbort(f"per-symbol PnL {pnl} != cash change {self.cash - self.start_cash}")
        return {
            "batches": self.seq,
            "equity": self.equity,
            "round_trips": self.round_trips,
            "resolutions": self.resolutions,
            "replies": self.replies,
            "final_cash": self.cash,
            "start_cash": self.start_cash,
            "per_symbol": {
                "symbol": "CBBTC",
                "pnl_quote_units": str(self.sym["pnl"]),
                "trades": self.sym["trades"],
                "fees_quote_units": str(self.sym["fees"]),
            },
        }


def decision_accepted(run):
    out = set()
    for f in run["resolutions"]:
        if f["kind"] == "order" and f["rejected_at"] != "decision":
            out.add(f"{f['seq']}:{f['index']}")
    return out


def compute_metrics(run, cost_run):
    start = run["start_cash"]
    net = run["final_cash"] - start
    cost_net = cost_run["final_cash"] - cost_run["start_cash"]
    marks = [int(p["equity_quote_units"]) for p in run["equity"]]
    returns = period_returns(marks)
    sharpe = sharpe9(returns)
    sortino = sortino9(returns)
    dd = max_drawdown9(marks)
    wins = sum(1 for t in run["round_trips"] if int(t["pnl_quote_units"]) > 0)
    trips = len(run["round_trips"])
    trades = 0
    rejected = 0
    fees = 0
    for f in run["resolutions"]:
        if f["status"] == "filled":
            fees += int(f["fee_quote_units"])
        if f["kind"] != "order":
            continue
        if f["status"] == "rejected":
            rejected += 1
        else:
            trades += 1
    # exposure: mean over scored batches of (equity - cash) / equity
    warm = 0
    batches = run["batches"]
    ssum = 0
    n = 0
    for p in run["equity"]:
        if p["seq"] < warm or p["seq"] >= batches:
            continue
        equity = int(p["equity_quote_units"])
        if equity > 0:
            ssum += ratio9(equity - int(p["cash_quote_units"]), equity)
        n += 1
    exposure = ssum // n if n else 0
    return {
        "net_pnl_quote_units": net,
        "net_pnl_usdc": net / 1_000_000,
        "return": fmt9(ratio9(net, start)) if start else "0.000000000",
        "sharpe": fmt9(sharpe) if sharpe is not None else None,
        "sortino": fmt9(sortino) if sortino is not None else None,
        "max_drawdown": fmt9(dd),
        "win_rate": fmt9(ratio9(wins, trips)) if trips else None,
        "round_trips": trips,
        "trades": trades,
        "rejected": rejected,
        "fees_quote_units": fees,
        "fees_usdc": fees / 1_000_000,
        "exposure": fmt9(exposure),
        "cost_2x_net_pnl_usdc": cost_net / 1_000_000,
        "guardrails": {
            "trades_ok": trades >= 1,
            "drawdown_ok": dd <= 250_000_000,
        },
    }


# ---------------------------------------------------------------- bot driver


def hello_msg(dataset):
    return {
        "type": "hello",
        "protocol": "agent-fun/bars-replay/v1",
        "seed": "1",
        "symbols": [
            {"symbol": "CBBTC", "mint": dataset["mint"], "price_scale": dataset["price_scale"], "qty_scale": dataset["qty_scale"]}
        ],
        "capital": {"quote": "USDC", "quote_scale": 6, "quote_units": "1000000000"},
        "bar_interval": "1m",
        "first_close_ms": dataset["window"]["start_ms"] + 60_000,
        "caps": {"max_open_positions": 1, "max_orders_per_bar": 16, "max_weight_bps": 10000, "shorts": False, "leverage": False},
        "fees": {"fee_bps": 4, "slippage_bps": 2, "min_notional_quote_units": "1000000"},
        "latency_bars": 1,
    }


def run_bot(dataset, bot_path, err_log):
    engine = BarsEngine(dataset)
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    proc = subprocess.Popen(
        [sys.executable, "-P", "-s", str(bot_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=open(err_log, "w"),
        env=env,
        text=True,
        bufsize=1,
    )
    try:
        proc.stdin.write(json.dumps(hello_msg(dataset)) + "\n")
        proc.stdin.flush()
        while True:
            batch = engine.next_batch()
            if batch is None:
                break
            proc.stdin.write(json.dumps(batch) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError(f"bot exited (rc={proc.poll()}) at batch {batch['seq']}; see {err_log}")
            reply = json.loads(line)
            if reply.get("type") != "reply" or reply.get("seq") != batch["seq"]:
                raise RuntimeError(f"bad reply at batch {batch['seq']}: {line[:200]}")
            if len(reply.get("orders", [])) > 64:
                raise RuntimeError(f"reply at batch {batch['seq']} carries {len(reply['orders'])} orders (max 64)")
            engine.replies.append(reply)
            engine.decide(reply)
            engine.seq += 1
            engine.phase = "ready"
        run = engine.finish()
        # The real harness sends done after the last batch; the SDK raises
        # ProtocolError on EOF before done, so send it, then close stdin.
        done = {
            "type": "done",
            "seq": run["batches"],
            "reason": "end_of_data",
            "account": {
                "cash_quote_units": str(run["final_cash"]),
                "positions": [],
                "equity_quote_units": str(run["final_cash"]),
            },
        }
        proc.stdin.write(json.dumps(done) + "\n")
        proc.stdin.flush()
        return run
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        proc.wait(timeout=30)


def replay_log(dataset, replies, fee_multiplier, clamp_orders):
    engine = BarsEngine(dataset, fee_multiplier=fee_multiplier, clamp_orders=clamp_orders)
    for reply in replies:
        batch = engine.next_batch()
        if batch is None:
            raise RuntimeError("action log ends before the window")
        engine.replies.append(reply)
        engine.decide(reply)
        engine.seq += 1
        engine.phase = "ready"
    return engine.finish()


def main():
    ap = argparse.ArgumentParser(description="Local bars-replay/v1 backtester (exact port of the kit harness).")
    ap.add_argument("--kit", required=True, help="kit directory (contains dataset/)")
    ap.add_argument("--bot", required=True, help="bot directory (with main.py) or main.py path")
    ap.add_argument("--out", default=None, help="write the run JSON here")
    args = ap.parse_args()

    kit_dir = Path(args.kit)
    bot_path = Path(args.bot)
    if bot_path.is_dir():
        bot_path = bot_path / "main.py"
    dataset = load_dataset(kit_dir)

    t0 = time.time()
    err_log = Path("/tmp/replay-bot-stderr.log")
    run = run_bot(dataset, bot_path, err_log)
    clamp = decision_accepted(run)
    cost_run = replay_log(dataset, run["replies"], 2, clamp)
    m = compute_metrics(run, cost_run)
    wall = time.time() - t0

    print(f"bot       {bot_path}")
    print(f"status    finished")
    print(f"net PnL   {fmt_usdc(m['net_pnl_quote_units'])} USDC  (return {m['return']})")
    print(
        f"trading   {m['trades']} trades over 1 symbols, win rate {m['win_rate']}, "
        f"max drawdown {m['max_drawdown']}, exposure {m['exposure']}, sharpe {m['sharpe']}, "
        f"{m['rejected']} rejected"
    )
    print(f"fees      {fmt_usdc(m['fees_quote_units'])} USDC; at 2x fees net PnL {m['cost_2x_net_pnl_usdc']:.6f} USDC")
    print(f"guardrails {'met' if (m['guardrails']['trades_ok'] and m['guardrails']['drawdown_ok']) else 'NOT MET'} "
          f"(trades ok, drawdown ok)")
    print(f"round trips {m['round_trips']}  (win rate {m['win_rate']})  wall {wall:.1f}s")
    print(f"dev gate  {'PASS (net PnL > 0)' if m['net_pnl_quote_units'] > 0 else 'FAIL (net PnL <= 0)'}")
    if args.out:
        Path(args.out).write_text(json.dumps({"metrics": m, "equity": run["equity"], "round_trips": run["round_trips"], "per_symbol": run["per_symbol"]}, indent=2))
        print(f"run json  {args.out}")


if __name__ == "__main__":
    main()
