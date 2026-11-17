"""Dip-and-trend cbBTC bot for make.fun's "Race a BTC trading bot live for 24 hours".

Protocol: agent-fun/bars-replay/v1 (stdin/stdout NDJSON). One asset, CBBTC (cbBTC/USDC,
Orca Whirlpool), long only, $1,000 paper USDC. The harness fills orders at the next
bar's open and charges 4 bps fee + 2 bps slippage + constant-product price impact,
so a full round trip costs ~26 bps: a trade only makes sense when it can catch a
move clearly bigger than that.

Strategy (trend + mean-reversion blend, all on 1-minute closes):
  * Trend filter: only buy when the trend is up (close above EMA-48 and EMA-12
    above EMA-48, with a real spread between them, so flat chop is skipped).
  * Entry: buy the pullback to EMA-12 inside an uptrend, or the breakout back
    above EMA-12, while RSI(14) is below 68 (not overbought).
  * Sizing: volatility-based. Risk 0.6% of equity per trade; the stop sits
    2.5x ATR(14) (min 0.40%) below the entry, and the notional is sized so the
    risk equals that budget, capped at 100% of equity and shrunk further if
    the modelled price impact would exceed 15 bps.
  * Exits: stop-loss at the stop, take-profit at 2x the stop distance, trend
    exit when the close falls back under EMA-48, or a 6-hour time exit.
  * Guard: if equity ever falls 5% below its peak, flatten and stop trading
    for the rest of the window (the ranking guardrail is 25%; this is stricter).

Deterministic by construction: pure float arithmetic on the closes, no clock,
no environment reads, no randomness. The same bars give the same orders on
every run, which the grader requires (two executions must match byte for byte).

Run it:  python3 -P -s main.py   (stdin: NDJSON hello/batches, stdout: replies)
Grade it: node <kit>/cli/makefun.mjs grade-local --kit <kit> --bot <repo> --driver process
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars  # noqa: E402  (vendored kit shim, byte-identical to sdk/makefun_bars.py)

SYMBOL = "CBBTC"

# --- strategy parameters (round numbers on purpose: nothing here is tuned to
# one window's noise; see README for the reasoning) ---
EMA_FAST = 12
EMA_SLOW = 48
RSI_LEN = 14
ATR_LEN = 14
RSI_OVERBOUGHT = 68.0
RSI_BREAKOUT_MIN = 45.0
MIN_TREND_SPREAD = 0.0005   # EMA12 must sit 5 bps above EMA48 to call it a trend
MIN_ATR_FRAC = 0.0002       # skip entries when 1-minute ATR is under 2 bps (dead tape)
RISK_FRAC = 0.006           # risk 0.6% of equity per trade
STOP_ATR_MULT = 2.5
STOP_MIN_FRAC = 0.004       # stop at least 0.40% below entry
TAKE_PROFIT_R = 2.0         # exit at 2x the stop distance
MAX_HOLD_BARS = 360         # 6-hour time exit
MAX_SLIPPAGE_BPS = 50       # 0.5% past the close the order was decided on
MAX_IMPACT_BPS = 15         # shrink the order if its modelled impact tops 15 bps
DD_HALT_FRAC = 0.05         # flatten and stop if equity falls 5% under its peak


def log(*parts):
    """Diagnostics go to stderr only: stdout is the protocol."""
    print("[bot]", *parts, file=sys.stderr, flush=True)


class Ema:
    """EMA seeded as the mean of its first span of closes (deterministic)."""

    def __init__(self, span):
        self.span = span
        self.k = 2.0 / (span + 1)
        self.warm = []
        self.value = None

    def update(self, close):
        if self.value is not None:
            self.value += self.k * (close - self.value)
            return self.value
        self.warm.append(close)
        if len(self.warm) == self.span:
            self.value = sum(self.warm) / self.span
        return self.value


class Rsi:
    """Wilder RSI, seeded with the mean gain/loss of the first span."""

    def __init__(self, span):
        self.span = span
        self.gains = []
        self.losses = []
        self.avg_gain = None
        self.avg_loss = None
        self.prev = None

    def update(self, close):
        if self.prev is None:
            self.prev = close
            return None
        gain = max(0.0, close - self.prev)
        loss = max(0.0, self.prev - close)
        self.prev = close
        if self.avg_gain is None:
            self.gains.append(gain)
            self.losses.append(loss)
            if len(self.gains) == self.span:
                self.avg_gain = sum(self.gains) / self.span
                self.avg_loss = sum(self.losses) / self.span
            else:
                return None
        else:
            self.avg_gain = (self.avg_gain * (self.span - 1) + gain) / self.span
            self.avg_loss = (self.avg_loss * (self.span - 1) + loss) / self.span
        if self.avg_loss == 0:
            return 100.0 if self.avg_gain > 0 else 50.0
        rs = self.avg_gain / self.avg_loss
        return 100.0 - 100.0 / (1.0 + rs)


class Atr:
    """Wilder ATR on true range of the 1-minute bars."""

    def __init__(self, span):
        self.span = span
        self.warm = []
        self.value = None
        self.prev_close = None

    def update(self, high, low, close):
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        self.prev_close = close
        if self.value is not None:
            self.value = (self.value * (self.span - 1) + tr) / self.span
            return self.value
        self.warm.append(tr)
        if len(self.warm) == self.span:
            self.value = sum(self.warm) / self.span
        return self.value


class Bot:
    def __init__(self):
        self.ema_fast = Ema(EMA_FAST)
        self.ema_slow = Ema(EMA_SLOW)
        self.rsi = Rsi(RSI_LEN)
        self.atr = Atr(ATR_LEN)
        self.prev_close = None
        self.prev_fast = None
        self.peak_equity = None
        self.halted = False
        self.hold_bars = 0
        self.stop_frac = None
        self.trades = 0

    def entry_price(self, state):
        """Break-even entry price (dollars) from the harness's position cost."""
        holding = state.positions.get(SYMBOL)
        if holding is None or holding.qty == 0:
            return None
        sym = state.symbols[SYMBOL]
        quote_scale = state.hello["capital"]["quote_scale"]
        price = holding.cost * 10 ** (sym["qty_scale"] + sym["price_scale"])
        price //= holding.qty * 10 ** quote_scale
        return price / 10 ** sym["price_scale"]

    def on_batch(self, state, batch):
        bar = None
        for b in batch["bars"]:
            if b["symbol"] == SYMBOL:
                bar = b
                break
        if bar is None:
            return []  # CBBTC absent this minute: a gap adds nothing
        if state.warming_up:
            self._feed(bar, state.symbols[SYMBOL]["price_scale"])
            return []
        close = int(bar["c"]) / 10 ** state.symbols[SYMBOL]["price_scale"]
        high = int(bar["h"]) / 10 ** state.symbols[SYMBOL]["price_scale"]
        low = int(bar["l"]) / 10 ** state.symbols[SYMBOL]["price_scale"]
        self._feed_bar(bar, close, high, low)

        if self.peak_equity is None:
            self.peak_equity = state.equity
        else:
            self.peak_equity = max(self.peak_equity, state.equity)

        holding = state.positions.get(SYMBOL)
        if holding is not None and holding.qty > 0:
            self.hold_bars += 1
            return self._exit(state, close, holding)
        self.hold_bars = 0
        self.stop_frac = None
        if self.halted:
            return []
        if state.equity < self.peak_equity * (1.0 - DD_HALT_FRAC):
            self.halted = True
            log(f"seq {state.seq}: drawdown guard tripped at equity "
                f"${state.equity / 1e6:.2f} (peak ${self.peak_equity / 1e6:.2f}); halted")
            return []
        return self._enter(state, close)

    def _feed(self, bar, price_scale):
        close = int(bar["c"]) / 10 ** price_scale
        high = int(bar["h"]) / 10 ** price_scale
        low = int(bar["l"]) / 10 ** price_scale
        self._feed_bar(bar, close, high, low)

    def _feed_bar(self, bar, close, high, low):
        self.ema_fast.update(close)
        self.ema_slow.update(close)
        self.rsi.update(close)
        self.atr.update(high, low, close)

    def _exit(self, state, close, holding):
        entry = self.entry_price(state)
        stop_frac = self.stop_frac if self.stop_frac is not None else STOP_MIN_FRAC
        fast = self.ema_fast.value
        slow = self.ema_slow.value
        reason = None
        if state.equity < self.peak_equity * (1.0 - DD_HALT_FRAC):
            reason = "drawdown guard"
            self.halted = True
        elif entry is not None and close <= entry * (1.0 - stop_frac):
            reason = f"stop-loss ({close:.2f} <= {entry * (1 - stop_frac):.2f})"
        elif entry is not None and close >= entry * (1.0 + TAKE_PROFIT_R * stop_frac):
            reason = f"take-profit ({close:.2f} >= {entry * (1 + TAKE_PROFIT_R * stop_frac):.2f})"
        elif slow is not None and close < slow:
            reason = f"trend exit (close {close:.2f} < EMA48 {slow:.2f})"
        elif self.hold_bars >= MAX_HOLD_BARS:
            reason = f"time exit ({self.hold_bars} bars held)"
        if reason is None:
            return []
        log(f"seq {state.seq}: SELL {holding.qty} ({reason}); "
            f"equity ${state.equity / 1e6:.2f}")
        self.trades += 1
        return [{"side": "sell", "symbol": SYMBOL, "qty": holding.qty,
                 "max_slippage_bps": MAX_SLIPPAGE_BPS}]

    def _enter(self, state, close):
        fast = self.ema_fast.value
        slow = self.ema_slow.value
        atr = self.atr.value
        rsi = self._rsi_now()
        if fast is None or slow is None or rsi is None or atr is None:
            return []
        if close <= slow or fast <= slow:
            return []  # no uptrend, no trade
        if (fast - slow) / close < MIN_TREND_SPREAD:
            return []  # chop: the trend spread is too small to pay for a trade
        if atr / close < MIN_ATR_FRAC:
            return []  # dead tape
        if rsi >= RSI_OVERBOUGHT:
            return []  # overbought: do not chase
        pullback = close <= fast * 1.0005
        breakout = (self.prev_close is not None and self.prev_fast is not None
                    and self.prev_close <= self.prev_fast and close > fast
                    and rsi >= RSI_BREAKOUT_MIN)
        if not (pullback or breakout):
            self.prev_close, self.prev_fast = close, fast
            return []
        self.prev_close, self.prev_fast = close, fast
        stop_frac = max(STOP_ATR_MULT * atr / close, STOP_MIN_FRAC)
        risk_units = int(state.equity * RISK_FRAC / stop_frac)
        notional = min(risk_units, state.cash)
        if notional < 1_000_000:  # below the $1 minimum notional
            return []
        impact = state.impact_bps(SYMBOL, notional)
        if impact is not None and impact > MAX_IMPACT_BPS:
            notional = int(notional * MAX_IMPACT_BPS / impact)
            if notional < 1_000_000:
                return []
        self.stop_frac = stop_frac
        log(f"seq {state.seq}: BUY ${notional / 1e6:.2f} @ ~${close:.2f} "
            f"({'pullback' if pullback else 'breakout'}, RSI {rsi:.1f}, "
            f"ATR {atr / close * 1e4:.1f} bps, stop {stop_frac * 1e4:.0f} bps); "
            f"equity ${state.equity / 1e6:.2f}")
        self.trades += 1
        return [{"side": "buy", "symbol": SYMBOL,
                 "notional_quote_units": notional,
                 "max_slippage_bps": MAX_SLIPPAGE_BPS}]

    def _rsi_now(self):
        """Current RSI value, or None while warming up."""
        r = self.rsi
        if r.avg_gain is None:
            return None
        if r.avg_loss == 0:
            return 100.0 if r.avg_gain > 0 else 50.0
        return 100.0 - 100.0 / (1.0 + r.avg_gain / r.avg_loss)


def on_hello(state):
    sym = state.symbols[SYMBOL]
    log(f"hello: {SYMBOL} mint {sym['mint'][:8]}..., capital "
        f"${int(state.hello['capital']['quote_units']) / 1e6:.0f}, "
        f"warmup {state.warmup_bars} bars, seed {state.hello.get('seed')}")


def on_done(state, done):
    log(f"done: final equity ${state.equity / 1e6:.2f}, "
        f"net PnL ${(state.equity - 1_000_000_000) / 1e6:+.2f}, "
        f"orders sent {BOT.trades}")


BOT = Bot()


def on_batch(state, batch):
    return BOT.on_batch(state, batch)


if __name__ == "__main__":
    makefun_bars.run(on_batch, on_hello=on_hello, on_done=on_done)
