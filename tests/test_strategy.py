"""Unit tests for the oversold-dip bot. Standard library only.

Run:  python3 -m unittest discover -s tests -v
The tests drive main.Bot through the vendored SDK's BarsState on synthetic
bars and on the kit's real dev window, checking the properties the grader
requires: the bot never crashes on bad input, never exceeds the caps, and
is deterministic (the same bars give the same orders on every run).
"""

import gzip
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor"))
import main  # noqa: E402
import makefun_bars  # noqa: E402

KIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "btc-race-24h-20261117-0603-kit")
DEV_BARS = os.path.join(KIT, "dataset", "bars", "CBBTC.ndjson.gz")


def make_hello(warmup_bars=0):
    return {
        "type": "hello",
        "protocol": "agent-fun/bars-replay/v1",
        "seed": "42",
        "symbols": [{"symbol": "CBBTC", "mint": "cbbtcf3aa214zXHbiAZQwf4122FBYbraNdFqgw4iMij",
                     "qty_scale": 8, "price_scale": 5}],
        "capital": {"quote_units": "1000000000", "quote_scale": 6},
        "caps": {"max_orders_per_bar": 16, "max_open_positions": 1,
                 "max_weight_bps": 10000},
        "fees": {"fee_bps": 4, "slippage_bps": 2, "min_notional_quote_units": "1000000"},
        "warmup_bars": warmup_bars,
    }


def make_bar(seq, close, high=None, low=None, depth="790000000000"):
    p = lambda d: str(int(round(d * 1e5)))  # dollars -> price units (scale 5)
    return {"symbol": "CBBTC", "t": 1794808980000 + seq * 60000,
            "o": p(close), "h": p(high if high is not None else close),
            "l": p(low if low is not None else close), "c": p(close),
            "v": None, "o_depth": [depth], "c_depth": [depth]}


def make_batch(seq, bar, cash=1_000_000_000, positions=None, fills=None):
    return {"type": "batch", "seq": seq, "t": 1794808980000 + seq * 60000,
            "bars": [] if bar is None else [bar],
            "fills": fills or [],
            "account": {"cash_quote_units": str(cash),
                        "positions": positions or [],
                        "equity_quote_units": str(cash)}}


def run_bars(bot, closes, cash=1_000_000_000):
    """Drive a bot over synthetic closes; return the list of replies (orders)."""
    state = makefun_bars.BarsState(make_hello())
    replies = []
    for seq, close in enumerate(closes):
        batch = make_batch(seq, make_bar(seq, close), cash=cash)
        state.apply_batch(batch)
        replies.append(bot.on_batch(state, batch))
    return replies


class TestIndicators(unittest.TestCase):
    def test_ema_seeds_on_mean_then_tracks(self):
        e = main.Ema(3)
        self.assertIsNone(e.update(10.0))
        self.assertIsNone(e.update(20.0))
        self.assertAlmostEqual(e.update(30.0), 20.0)
        self.assertAlmostEqual(e.update(40.0), 30.0)

    def test_rsi_stays_in_range(self):
        r = main.Rsi(14)
        closes = [100.0 + (i % 5 - 2) * 0.7 for i in range(200)]
        values = [v for v in (r.update(c) for c in closes) if v is not None]
        self.assertTrue(values)
        self.assertTrue(all(0.0 <= v <= 100.0 for v in values))

    def test_rsi_all_up_is_100(self):
        r = main.Rsi(3)
        for c in [10.0, 11.0, 12.0, 13.0, 14.0]:
            v = r.update(c)
        self.assertEqual(v, 100.0)


class TestBotRobustness(unittest.TestCase):
    def test_flat_bars_never_crash_and_never_trade(self):
        bot = main.Bot()
        replies = run_bars(bot, [64000.0] * 300)
        self.assertEqual(sum(len(r) for r in replies), 0)

    def test_missing_bars_are_skipped(self):
        bot = main.Bot()
        state = makefun_bars.BarsState(make_hello())
        for seq in range(100):
            bar = None if seq % 7 == 0 else make_bar(seq, 64000.0 + seq)
            batch = make_batch(seq, bar)
            state.apply_batch(batch)
            bot.on_batch(state, batch)  # must not raise

    def test_warmup_bars_feed_indicators_but_never_trade(self):
        bot = main.Bot()
        state = makefun_bars.BarsState(make_hello(warmup_bars=50))
        closes = [64000.0 + i * 10 for i in range(100)]
        orders = 0
        for seq, close in enumerate(closes):
            batch = make_batch(seq, make_bar(seq, close))
            state.apply_batch(batch)
            orders += len(bot.on_batch(state, batch))
        self.assertEqual(orders, 0)  # warm-up + no trend setup in this ramp

    def test_orders_respect_cash_and_caps(self):
        bot = main.Bot()
        # Sharp dip then recovery: RSI(14) dives under 35, so the bot buys;
        # the bounce then takes it out at the take-profit.
        closes = []
        price = 64000.0
        for i in range(300):
            if i < 100:
                price *= 1.0  # flat: indicators settle
            elif i < 130:
                price *= 0.998  # -0.2%/bar for 30 bars: a real dip
            else:
                price *= 1.0015  # recovery
            closes.append(price)
        replies = run_bars(bot, closes)
        buys = [o for r in replies for o in r if o["side"] == "buy"]
        self.assertGreater(len(buys), 0, "bot should buy the dip")
        for reply in replies:
            self.assertLessEqual(len(reply), 16)
            for o in reply:
                self.assertIn(o["side"], ("buy", "sell"))
                self.assertEqual(o["symbol"], "CBBTC")
                self.assertTrue(0 <= o["max_slippage_bps"] <= 10000)
                if o["side"] == "buy":
                    self.assertGreaterEqual(o["notional_quote_units"], 1_000_000)
                    self.assertLessEqual(o["notional_quote_units"], 1_000_000_000)

    def test_crash_halt_stops_entries(self):
        bot = main.Bot()
        # Flat, then a -3% slide over 4 hours: the 4-hour return crosses
        # -1.50%, so the crash halt must fire and no entry may follow,
        # even when RSI is deeply oversold.
        closes = []
        price = 64000.0
        for i in range(400):
            if i < 100:
                pass
            elif i < 340:
                price *= 0.99987  # -3% over 240 bars
            closes.append(price)
        replies = run_bars(bot, closes)
        # Entries may happen before the halt trips (early in the slide),
        # but none may happen after the 4-hour return crosses -1.50%.
        # Find the first bar where the 4h return is under -1.50%.
        first_halt = None
        for i in range(240, len(closes)):
            if closes[i] / closes[i - 240] - 1.0 < -0.015:
                first_halt = i
                break
        self.assertIsNotNone(first_halt, "test series must cross the halt line")
        late_buys = [o for seq, r in enumerate(replies) if seq > first_halt + 1
                     for o in r if o["side"] == "buy"]
        self.assertEqual(late_buys, [], "no entries after the crash halt")


class TestBotDeterminism(unittest.TestCase):
    def test_same_bars_same_orders_on_dev_window(self):
        bars = [json.loads(l) for l in gzip.open(DEV_BARS, "rt")]
        self.assertEqual(len(bars), 1440)
        for b in bars:
            b["symbol"] = "CBBTC"  # dataset bars omit it; the harness adds it
        all_replies = []
        for _ in range(2):
            bot = main.Bot()
            state = makefun_bars.BarsState(make_hello())
            replies = []
            for seq, bar in enumerate(bars):
                batch = {"type": "batch", "seq": seq, "t": bar["t"],
                         "bars": [bar], "fills": [],
                         "account": {"cash_quote_units": "1000000000",
                                     "positions": [],
                                     "equity_quote_units": "1000000000"}}
                state.apply_batch(batch)
                replies.append(bot.on_batch(state, batch))
            all_replies.append(replies)
        self.assertEqual(all_replies[0], all_replies[1])
        n_orders = sum(len(r) for r in all_replies[0])
        print(f"\ndev-window offline decisions (no fills): {n_orders} orders")
        self.assertGreater(n_orders, 0, "bot should want to trade the dev window")


if __name__ == "__main__":
    unittest.main()
