"""Oversold-dip cbBTC bot for make.fun's "Race a BTC trading bot live for 24 hours".

Protocol: agent-fun/bars-replay/v1 (stdin/stdout NDJSON). One asset, CBBTC (cbBTC/USDC,
Orca Whirlpool), long only, $1,000 paper USDC. The harness fills orders at the next
bar's open and charges 4 bps fee + 2 bps slippage + constant-product price impact,
so a full round trip costs ~26 bps: a trade only makes sense when the bounce it
catches is clearly bigger than that.

Strategy (buy oversold dips, never buy a crash):
  * The dev window and three out-of-sample 24-hour windows (Coinbase BTC-USD,
    +0.36% chop, -4.35% crash, -1.37% drift-down) were backtested through an
    exact local port of the harness. Two earlier designs both failed there:
    a 1-minute trend system churned 16 round trips and lost $51.22 on dev, and
    a slow trend follower bought every local top and lost on all four windows.
    What survives on all four windows is buying oversold dips with a hard
    crash halt -- see README for the full backtest table.
  * Entry: RSI(14) under 35 (oversold on 1-minute bars) while the close is
    under its 1-hour mean, at most one entry per hour (60-bar cooldown).
  * Sizing: risk 0.50% of equity per trade; the stop sits 0.80% below the
    close the order was decided on, so the notional is ~62% of equity, capped
    at cash. The take-profit is +1.20% (1.5x the stop distance).
  * Exits: stop-loss at -0.80%, take-profit at +1.20%, or a 4-hour time exit.
    There is deliberately no mean-reversion exit: exiting the moment the close
    crosses its 1-hour mean was measured to destroy the strategy (it exits on
    the first tiny bounce and pays the round-trip cost every time).
  * Crash halt: if the 4-hour return falls under -1.50% -- about 3x the normal
    4-hour chop, a regime break rather than a dip -- stop trading for the rest
    of the window. On the -4.35% crash window this cut the loss from -$29.98
    to -$8.13; on the choppy windows it fires rarely or never. The halt is
    permanent on purpose: re-arming was measured to re-buy dips mid-crash and
    lose another ~$14.
  * Guard: if equity ever falls 5% below its peak, stop entering (the ranking
    guardrail is 25%; this is stricter).

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

# --- strategy parameters (round numbers on purpose: the backtest table in the
# README shows this plateau, not a single tuned spike) ---
EMA_MEAN = 60             # 1-hour mean the dip must sit under
RSI_LEN = 14
RSI_OVERSOLD = 35.0      # enter when RSI(14) falls under 35
RISK_FRAC = 0.005        # risk 0.50% of equity per trade
STOP_FRAC = 0.008        # stop 0.80% below the decision close
TAKE_PROFIT_FRAC = 0.012  # take profit 1.20% above the decision close
MAX_HOLD_BARS = 240      # 4-hour time exit
COOLDOWN_BARS = 60       # 1 hour between an exit and the next entry
CRASH_HALT_4H = -0.015   # halt for the window if the 4-hour return tops -1.50%
MAX_SLIPPAGE_BPS = 50    # 0.5% past the close the order was decided on
DD_HALT_FRAC = 0.05      # stop entering if equity falls 5% under its peak


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


class Bot:
    def __init__(self):
        self.ema_mean = Ema(EMA_MEAN)
        self.rsi = Rsi(RSI_LEN)
        self.closes = []           # every close, for the 4-hour return
        self.entry_close = None   # the close the open trade was decided on
        self.cooldown_until = -1
        self.crash_halted = False
        self.peak_equity = None
        self.halted = False
        self.hold_bars = 0
        self.trades = 0

    def ret_4h(self, close):
        """The 4-hour return (close vs the close 240 bars ago), or None."""
        if len(self.closes) <= 240:
            return None
        return close / self.closes[-241] - 1.0

    def on_batch(self, state, batch):
        bar = None
        for b in batch["bars"]:
            if b["symbol"] == SYMBOL:
                bar = b
                break
        if bar is None:
            return []  # CBBTC absent this minute: a gap adds nothing
        close = int(bar["c"]) / 10 ** state.symbols[SYMBOL]["price_scale"]
        if state.warming_up:
            self._feed(bar, close)
            return []
        self._feed(bar, close)

        if self.peak_equity is None:
            self.peak_equity = state.equity
        else:
            self.peak_equity = max(self.peak_equity, state.equity)

        holding = state.positions.get(SYMBOL)
        if holding is not None and holding.qty > 0:
            self.hold_bars += 1
            return self._exit(state, close, holding)
        self.hold_bars = 0
        if self.halted or self.crash_halted:
            return []
        if state.equity < self.peak_equity * (1.0 - DD_HALT_FRAC):
            self.halted = True
            log(f"seq {state.seq}: drawdown guard tripped at equity "
                f"${state.equity / 1e6:.2f} (peak ${self.peak_equity / 1e6:.2f}); halted")
            return []
        return self._enter(state, close)

    def _feed(self, bar, close):
        self.ema_mean.update(close)
        self.rsi.update(close)
        self.closes.append(close)

    def _exit(self, state, close, holding):
        entry = self.entry_close
        reason = None
        if entry is not None and close <= entry * (1.0 - STOP_FRAC):
            reason = f"stop-loss ({close:.2f} <= {entry * (1 - STOP_FRAC):.2f})"
        elif entry is not None and close >= entry * (1.0 + TAKE_PROFIT_FRAC):
            reason = f"take-profit ({close:.2f} >= {entry * (1 + TAKE_PROFIT_FRAC):.2f})"
        elif self.hold_bars >= MAX_HOLD_BARS:
            reason = f"time exit ({self.hold_bars} bars held)"
        if reason is None:
            return []
        log(f"seq {state.seq}: SELL {holding.qty} ({reason}); "
            f"equity ${state.equity / 1e6:.2f}")
        self.cooldown_until = state.seq + COOLDOWN_BARS
        self.entry_close = None
        self.trades += 1
        return [{"side": "sell", "symbol": SYMBOL, "qty": holding.qty,
                 "max_slippage_bps": MAX_SLIPPAGE_BPS}]

    def _enter(self, state, close):
        # The crash halt: a 4-hour fall past -1.50% is a regime break, not a
        # dip. It is permanent for the window; re-arming was measured to
        # re-buy dips mid-crash and lose ~$14 more on the crash window.
        ret4h = self.ret_4h(close)
        if ret4h is not None and ret4h < CRASH_HALT_4H:
            self.crash_halted = True
            log(f"seq {state.seq}: crash halt (4h return {ret4h * 100:+.2f}%); "
                f"no more entries this window")
            return []
        if state.seq < self.cooldown_until:
            return []  # cooling down after an exit
        rsi = self._rsi_now()
        if rsi is None:
            return []
        if rsi >= RSI_OVERSOLD:
            return []  # not oversold yet
        mean = self.ema_mean.value
        if mean is not None and close >= mean:
            return []  # above the 1-hour mean: not a dip
        notional = int(state.equity * RISK_FRAC / STOP_FRAC)
        notional = min(notional, state.cash)
        if notional < 1_000_000:  # below the $1 minimum notional
            return []
        self.entry_close = close
        ret4h_s = "n/a" if ret4h is None else f"{ret4h * 100:+.2f}%"
        log(f"seq {state.seq}: BUY ${notional / 1e6:.2f} @ ~${close:.2f} "
            f"(RSI {rsi:.1f}, 4h {ret4h_s}); "
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
