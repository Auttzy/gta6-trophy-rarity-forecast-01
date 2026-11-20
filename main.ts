// GTA VI trophy-rarity forecaster (gta6-trophy-rarity-forecast-01).
//
// Node 22, standard library only, no network, deterministic. Reads every
// <id>.in in the input directory and writes <id>.out to the output directory:
// {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}.
//
// The model, per trophy, in the logit scale (the scale the contest scores):
//
//   * The history is a staircase: Exophase refreshes a game's counts on its
//     own schedule, hours apart, so consecutive reads are identical until a
//     refresh jumps them. We collapse each trophy's reads to its distinct
//     states (one point per refresh) and fit the trend on those.
//   * TREND branch (the history moved): fit, by least squares, the trophy's
//     logit(rate) against time over its distinct states, shrink that slope
//     toward the cross-trophy median slope (strongly for trophies with few
//     earners, whose rates are one-earner noise), then extrapolate to the
//     target with the slope damped toward zero (launch dynamics decay; the
//     effective window is H * (1 - exp(-horizon / H)) with H = 72 h, so a
//     4.5-day horizon extrapolates ~2.3 days of current slope, not 4.5).
//     The median is the damped trend read at the target. The spread is the
//     house random walk (0.5 logit per sqrt day) combined in quadrature with
//     a term for how far the median moved and a term for the slope's own
//     standard error, so a trophy whose trend is a guess gets a wide band.
//   * FLAT branch (the history never moved: a stale page at launch, as in
//     the practice cases): no trend exists to fit, so we use a dilution
//     prior. A refresh is coming, and it brings a flood of new players who
//     have barely played: the tracked count jumps several-fold while the
//     earners barely move, so every rate but a universal one falls. The
//     median is the last rate times F_MED (0.20, the median dilution
//     implied by the practice refresh: tracked 638 -> 3,868 while the
//     earners of the non-universal trophies rose only 305 -> 408), with a
//     0.70-logit spread because the size of the jump is uncertain. A trophy
//     at ~100% is universal (every new player earns it at once), so it
//     stays at ~100% with a narrow band instead of diluting.
//   * In both branches the median is floored at the scorer's 0.01% clamp and
//     the quantiles are monotone by construction (z_tau increasing).
//
// What the live page taught us (read 2026-11-20 ~09:08 UTC, used only to
// shape this model's priors, never hardcoded): by ~33 h after launch most
// trophies had collapsed to 0.01-0.02% while a handful of early/easy ones
// (Chapter 1, fishing, Breakfast Run) had risen several-fold. Dilution
// dominates the many; progression dominates the few. The trend branch is
// what lets the final case tell those apart; the flat branch is what keeps
// the practice cases from copying the house baseline's no-change forecast,
// which the refresh shock punishes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** The house baseline's random-walk spread, logit per square-root day. */
const SIGMA_RW = 0.5;
/** Baseline uncertainty every forecast carries, logit. */
const SIGMA_BASE = 0.35;
/** Flat branch: the median's dilution factor (last rate * F_MED). */
const F_MED = 0.2;
/** Flat branch: the spread, logit (a refresh jump is large and uncertain). */
const SIGMA_FLAT = 0.7;
/** Trend damping timescale, hours: slopes decay toward zero over ~H. */
const H_DAMP = 72;
/** Cap on the predicted logit move, either direction. */
const MAX_DLOGIT = 3.5;
/** Earners at which a trophy's own slope counts half (shrinkage constant). */
const SHRINK_K = 20;
/** A rate at/above this is universal: new players earn it at once. */
const UNIVERSAL = 99.5;

interface Case {
  levels: number[];
  horizon_minutes: number;
  trophies: Array<{ id: number }>;
  history: Array<{ at: string; players_tracked: number; rates: Array<number | null> }>;
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

interface State {
  /** Hours since the case's first read. */
  t: number;
  players: number;
  rate: number;
}

/** Least-squares slope and its standard error for y against x. */
function linfit(xs: number[], ys: number[]): { slope: number; se: number; resid: number } {
  const n = xs.length;
  if (n < 2) return { slope: 0, se: Infinity, resid: 0 };
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx <= 0) return { slope: 0, se: Infinity, resid: 0 };
  const slope = sxy / sxx;
  let sse = 0;
  for (let i = 0; i < n; i++) {
    const e = ys[i] - (my + slope * (xs[i] - mx));
    sse += e * e;
  }
  const dof = n - 2;
  const resid = dof > 0 ? Math.sqrt(sse / dof) : 0;
  const se = Math.sqrt((dof > 0 ? sse / dof : 0) / sxx);
  return { slope, se, resid };
}

/** A trophy's distinct states: one point per refresh, at first-seen time. */
function distinctStates(c: Case, idx: number): State[] {
  const t0 = Date.parse(c.history[0].at);
  const states: State[] = [];
  for (const h of c.history) {
    const r = h.rates[idx];
    if (r === null || r === undefined) continue;
    const st: State = { t: (Date.parse(h.at) - t0) / 3_600_000, players: h.players_tracked, rate: r };
    const prev = states[states.length - 1];
    if (prev && prev.players === st.players && prev.rate === st.rate) continue;
    states.push(st);
  }
  return states;
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`case asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error('case has no history');
  const horizonH = c.horizon_minutes / 60;
  const horizonD = c.horizon_minutes / 1440;
  // Effective extrapolation window under exponential decay of the slope.
  const tauEff = H_DAMP * (1 - Math.exp(-horizonH / H_DAMP));
  const last = c.history[c.history.length - 1];

  // First pass: each trophy's own logit slope, and the cross-trophy median.
  const slopes: number[] = [];
  const per: Array<{ states: State[]; slope: number; se: number; earners: number }> = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const states = distinctStates(c, i);
    const earners = Math.max(0, (last.rates[i] ?? 0) * last.players_tracked / 100);
    if (states.length >= 2) {
      const f = linfit(states.map((s) => s.t), states.map((s) => logit(s.rate)));
      slopes.push(f.slope);
      per.push({ states, slope: f.slope, se: f.se, earners });
    } else {
      per.push({ states, slope: 0, se: Infinity, earners });
    }
  }
  const sortedSlopes = [...slopes].sort((a, b) => a - b);
  const medianSlope = sortedSlopes.length > 0 ? sortedSlopes[Math.floor(sortedSlopes.length / 2)] : 0;

  return c.trophies.map((t, i) => {
    const rLast = last.rates[i] ?? 0;
    const info = per[i];
    // Universal trophies stay universal.
    if (rLast >= UNIVERSAL) {
      const med = logit(99.9);
      const s = 0.3;
      return { id: t.id, q: Z.map((z) => round4(clampPct(invLogit(med + z * s)))) };
    }
    const moved = info.states.length >= 2;
    if (!moved) {
      // Flat branch: stale page, a refresh is coming, rates dilute.
      const med = logit(Math.max(rLast * F_MED, 0.01));
      return { id: t.id, q: Z.map((z) => round4(clampPct(invLogit(med + z * SIGMA_FLAT)))) };
    }
    // Trend branch: shrink the slope toward the median by earner count.
    const w = info.earners / (info.earners + SHRINK_K);
    const slope = w * info.slope + (1 - w) * medianSlope;
    const dLogit = Math.min(MAX_DLOGIT, Math.max(-MAX_DLOGIT, slope * tauEff));
    const med = logit(rLast) + dLogit;
    const seTerm = Number.isFinite(info.se) ? info.se * tauEff : 0.5;
    const sigma = Math.sqrt(
      SIGMA_RW * SIGMA_RW * horizonD + 0.35 * 0.35 * dLogit * dLogit + SIGMA_BASE * SIGMA_BASE + seTerm * seTerm,
    );
    return { id: t.id, q: Z.map((z) => round4(clampPct(invLogit(med + z * sigma)))) };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
