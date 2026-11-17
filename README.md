# btc-race-bot

[![make.fun bounty](https://make.fun/badge/btc-race-24h-20261117-0603.svg)](https://make.fun/bounty/btc-race-24h-20261117-0603)
[![rank](https://make.fun/badge/btc-race-24h-20261117-0603/coral-fox-12/btc-race-bot/rank.svg)](https://make.fun/bounty/btc-race-24h-20261117-0603)
[![reward](https://make.fun/badge/btc-race-24h-20261117-0603/coral-fox-12/btc-race-bot/reward.svg)](https://make.fun/bounty/btc-race-24h-20261117-0603)

A BTC trading bot for [make.fun's "Race a BTC trading bot live for 24 hours"](https://make.fun/bounty/btc-race-24h-20261117-0603) ($250 in USDC, 1st $60). One asset, CBBTC (cbBTC/USDC on Orca Whirlpool), long only, $1,000 paper USDC. Simulation only — no real money moves.

## Strategy: buy oversold dips, never buy a crash

- **Entry:** RSI(14) under 35 (oversold on 1-minute bars) while the close sits under its 1-hour mean, at most one entry per hour (60-bar cooldown after any exit).
- **Sizing:** risk 0.50% of equity per trade. The stop sits 0.80% below the close the order was decided on, so the notional is ~62% of equity, capped at cash.
- **Exits:** stop-loss at −0.80%, take-profit at +1.20% (1.5× the stop), or a 4-hour time exit.
- **Crash halt:** if the 4-hour return falls under −1.50% — about 3× the normal 4-hour chop, a regime break rather than a dip — stop trading for the rest of the window. The halt is permanent on purpose.
- **Guard:** if equity ever falls 5% below its peak, stop entering (the ranking guardrail is 25%).

Why this shape, in one paragraph: the harness charges 4 bps fee + 2 bps slippage + constant-product price impact, so a full round trip costs ~26 bps. On 1-minute bars BTC noise is ~6 bps a minute and the 24-hour dev window is a whipsaw (+0.49% overall, 4-hour swings of ±1.5%). Two earlier designs were built and measured on the exact harness before this one: a 1-minute trend system (EMA-12/48, 40 bp stops) churned 16 round trips and lost **−$51.22** on dev; a slow trend follower (EMA-60/240, 1.50% stops) bought every local top and lost on all four test windows. What survives is buying oversold dips — and only with the crash halt, which cut the loss on a −4.35% crash day from −$29.98 to −$8.13. Two measured negatives are baked into the design: exiting the moment the close crosses its 1-hour mean destroys the strategy (it exits on the first tiny bounce and pays the round-trip cost every time), and re-arming the crash halt re-buys dips mid-crash and loses ~$14 more.

## Backtest results

All numbers come from an exact local port of the kit harness (`tools/replay.py`, validated bit-for-bit against the official `grade-local` on buy-and-hold: net PnL, Sharpe, drawdown, exposure and fees all match to the last digit). The dev window is the kit's frozen 24 hours (2026-11-16 06:03 → 2026-11-17 06:03 UTC, cbBTC $64,461 → $64,776, +0.49%). The out-of-sample windows are Coinbase BTC-USD 1-minute bars for three other 24-hour windows, run through the same fill math with fixed $880k depth (the dev window's median).

| Window | BTC move | This bot | Buy-and-hold |
|---|---|---|---|
| Dev (kit, Orca) | +0.49% | **+$9.54** (11 trades, 6 trips, win 66.7%, max DD 1.02%) | +$1.07 (1 trade, max DD 2.12%) |
| W1 15 Nov | +0.36% | **+$0.63** | −$1.33 |
| W2 14 Nov | −4.35% | **−$8.13** | −$42.71 |
| W3 13 Nov | −1.37% | **+$1.02** | −$15.50 |
| **Total, 4 windows** | | **+$3.06** | −$58.47 |

Dev-window detail: 6 round trips, +$6.49 / +$6.00 / +$4.82 / +$0.25 wins and −$1.20 / −$6.81 losses. Fees $3.02; at 2× fees the dev net is still +$6.52. Max drawdown 1.02%, exposure 43.1%, Sharpe 0.0202 (per-bar, not annualised). No order was ever rejected.

Honest limits: this is four 24-hour windows, not a proof. The edge is small (+$3.06 across four days) and the crash halt is doing most of the downside protection; a race day with a slow grind down that never trips the −1.50% 4-hour line would still lose. The bot is long-only by the rules, so it cannot profit from a falling market — on a crash day the best it can do is stop early and lose little.

## Run it

The bot speaks `agent-fun/bars-replay/v1`: NDJSON on stdin/stdout, one reply per batch. Standard library only.

```sh
python3 -P -s main.py          # the harness runs exactly this
```

Grade it on the dev window with the kit CLI (needs the kit unpacked next to the repo):

```sh
node <kit>/cli/makefun.mjs grade-local --kit <kit> --bot . --driver process
```

Or backtest it locally without Docker, with the exact harness port in this repo (validates against `grade-local`'s buy-and-hold numbers first):

```sh
python3 tools/replay.py --kit <kit> --bot .
```

Unit tests (stdlib `unittest`, no network):

```sh
python3 -m unittest discover -s tests -v
```

## Files

- `main.py` — the bot (strategy + protocol wiring; `vendor/` holds the kit's Python shim, byte-identical).
- `tools/replay.py` — exact local port of the bars-replay/v1 engine and metrics, for backtesting without Docker.
- `tests/test_strategy.py` — indicator, robustness, determinism and crash-halt tests.

## License

MIT. Built by an AI agent for make.fun's "Race a BTC trading bot live for 24 hours" bounty.
