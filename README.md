[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Auttzy/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Auttzy/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy-Rarity Forecaster

Entry for [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies, its earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles (5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale.

## Method

`main.ts` (Node 22, stdlib only, no network, deterministic) reads every `*.in` case and writes one `*.out` per case. Per trophy, in the logit scale:

- **History is a staircase.** Exophase refreshes a game's counts on its own schedule, hours apart, so consecutive reads are identical until a refresh jumps them. The program collapses each trophy's reads to its distinct states (one point per refresh) and fits the trend on those.
- **Trend branch (the history moved).** Least-squares fit of logit(rate) against time over the distinct states, shrunk toward the cross-trophy median slope (strongly for trophies with few earners, whose rates are one-earner noise), then extrapolated to the target with the slope damped toward zero (launch dynamics decay; the effective window is 72 h × (1 − exp(−horizon / 72 h))). The spread combines the house random walk (0.5 logit per √day) with terms for how far the median moved and the slope's own standard error.
- **Flat branch (the history never moved — a stale page at launch, as in the practice cases).** No trend exists to fit, so a dilution prior applies: a refresh is coming and brings a flood of new players who have barely played, so every rate but a universal one falls. The median is the last rate × 0.20 with a 0.70-logit spread. A trophy at ~100% is universal (every new player earns it at once), so it stays at ~100% instead of diluting.

The key insight is dilution: the practice cases' flat histories hide a refresh that multiplied the tracked players ~6× while the earners barely moved, so the house baseline's no-change forecast (loss ~1.51) is punished and a forecast that expects rates to fall wins. On the final case the trend branch lets the data decide per trophy whether dilution or player progression dominates.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## License

MIT — see [LICENSE](LICENSE).
