#!/usr/bin/env python3
"""
M2 -- What-if analysis with Covasim
Data Driven Modeling and Simulation, WS 26/27

Runs a set of what-if scenarios (type, strictness, timing and combination of
interventions), each with N stochastic replications, and produces
  * mean trajectories with 90 % confidence bands (figures),
  * a summary table (CSV + LaTeX fragment),
  * run-time measurements (per run and total).

Usage examples
--------------
  python m2_whatif.py                                   # 50k agents, 20 runs, serial
  python m2_whatif.py --parallel --cpus 20              # on the compute server
  python m2_whatif.py --pop-size 20000 --n-runs 20      # quick test on a laptop
  python m2_whatif.py --plot-only                       # re-plot from saved results

Install:  pip install covasim scipy matplotlib pandas
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import covasim as cv
from scipy import stats

# ----------------------------------------------------------------------------
# Scenario definitions
# ----------------------------------------------------------------------------

START = 30  # default start day (days after simulation start) of the interventions


def distancing(reduction, start):
    """Distancing, masks and hygiene: reduce the per-contact transmission
    probability beta by `reduction` (e.g. 0.4 = 40 % less) from day `start`."""
    return cv.change_beta(days=start, changes=1.0 - reduction)


def school_closure(start):
    """Remove all contacts in the school layer from day `start`."""
    return cv.clip_edges(days=start, changes=0.0, layers='s')


def test_and_trace(start):
    """Test symptomatic people (10 % per day, 1 day delay) and trace their contacts."""
    return [
        cv.test_prob(symp_prob=0.10, asymp_prob=0.001, symp_quar_prob=0.10,
                     test_delay=1, start_day=start),
        cv.contact_tracing(trace_probs=dict(h=1.0, s=0.8, w=0.6, c=0.1),
                           trace_time=dict(h=0, s=2, w=2, c=2),
                           start_day=start),
    ]


def flat(*items):
    """Flatten a mix of interventions and lists of interventions."""
    out = []
    for it in items:
        out.extend(it if isinstance(it, list) else [it])
    return out


# name -> (label for plots, factory returning a FRESH list of interventions)
# (a factory is used because intervention objects hold state and must not be shared)
SCENARIOS = {
    'baseline':      ('No intervention',                       lambda: []),
    'beta20':        ('Distancing -20 %',                      lambda: flat(distancing(0.2, START))),
    'beta40':        ('Distancing -40 %',                      lambda: flat(distancing(0.4, START))),
    'beta60':        ('Distancing -60 %',                      lambda: flat(distancing(0.6, START))),
    'beta40_d15':    ('Distancing -40 %, start day 15',        lambda: flat(distancing(0.4, 15))),
    'beta40_d45':    ('Distancing -40 %, start day 45',        lambda: flat(distancing(0.4, 45))),
    'schools':       ('School closure',                        lambda: flat(school_closure(START))),
    'ttq':           ('Test + trace',                          lambda: flat(test_and_trace(START))),
    'dist_schools':  ('Distancing -40 % + school closure',     lambda: flat(distancing(0.4, START), school_closure(START))),
    'dist_ttq':      ('Distancing -40 % + test + trace',       lambda: flat(distancing(0.4, START), test_and_trace(START))),
    'all':           ('Distancing + schools + test + trace',   lambda: flat(distancing(0.4, START), school_closure(START), test_and_trace(START))),
}

# experiment name -> (title, scenario names, interventions start day for the vertical line)
EXPERIMENTS = {
    'E1_type':        ('Type of intervention',   ['baseline', 'beta40', 'schools', 'ttq'], START),
    'E2_strictness':  ('Strictness',             ['baseline', 'beta20', 'beta40', 'beta60'], START),
    'E3_timing':      ('Timing',                 ['baseline', 'beta40_d15', 'beta40', 'beta40_d45'], None),
    'E4_combination': ('Combination',            ['baseline', 'beta40', 'schools', 'ttq', 'dist_schools', 'dist_ttq', 'all'], START),
}

CONFIDENCE = 0.90


# ----------------------------------------------------------------------------
# Running the experiments
# ----------------------------------------------------------------------------

def run_scenario(name, args):
    """Run one scenario n_runs times. Returns dict with arrays and timings."""
    label, make_interventions = SCENARIOS[name]
    pars = dict(
        pop_size=args.pop_size,
        pop_infected=args.pop_infected,
        pop_type='hybrid',
        start_day='2020-03-01',
        n_days=args.n_days,
        rand_seed=args.seed,   # same base seed in all scenarios -> common random numbers
        verbose=0,
    )
    interventions = make_interventions()  # fresh objects for every scenario
    sim = cv.Sim(pars, interventions=interventions, label=label)
    msim = cv.MultiSim(sim)

    t0 = time.perf_counter()
    msim.run(n_runs=args.n_runs, parallel=args.parallel, n_cpus=args.cpus,
             keep_people=False, verbose=0)
    wall = time.perf_counter() - t0

    sims = msim.sims
    out = dict(
        new_infections=np.array([s.results['new_infections'].values for s in sims]),
        cum_infections=np.array([s.results['cum_infections'].values for s in sims]),
        cum_deaths=np.array([s.results['cum_deaths'].values for s in sims]),
        run_times=np.array([s.elapsed for s in sims]),
        wall=wall,
    )
    return out


def run_all(args):
    needed = []
    for _, (_, names, _) in EXPERIMENTS.items():
        for n in names:
            if n not in needed:
                needed.append(n)

    results = {}
    t_start = time.perf_counter()
    for i, name in enumerate(needed, 1):
        print(f'[{i}/{len(needed)}] {name}: {args.n_runs} runs ...', flush=True)
        results[name] = run_scenario(name, args)
        r = results[name]
        print(f'    wall-clock {r["wall"]:.1f} s, mean run time {r["run_times"].mean():.2f} s', flush=True)
    total = time.perf_counter() - t_start
    print(f'Total time for the whole simulation experiment: {total:.1f} s')

    # save raw results so that plots can be redone without rerunning
    os.makedirs(args.outdir, exist_ok=True)
    flat_dict = {'total_time': np.array(total)}
    for name, r in results.items():
        for key, val in r.items():
            flat_dict[f'{name}__{key}'] = np.asarray(val)
    np.savez_compressed(os.path.join(args.outdir, 'results_raw.npz'), **flat_dict)
    return results, total


def load_results(args):
    data = np.load(os.path.join(args.outdir, 'results_raw.npz'))
    results = {}
    for key in data.files:
        if key == 'total_time':
            continue
        name, field = key.split('__')
        results.setdefault(name, {})[field] = data[key]
    return results, float(data['total_time'])


# ----------------------------------------------------------------------------
# Statistics and plotting
# ----------------------------------------------------------------------------

def mean_ci(arr, conf=CONFIDENCE):
    """Mean and t-based confidence interval of the mean over axis 0 (the runs)."""
    n = arr.shape[0]
    mean = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * sem
    return mean, mean - h, mean + h


def plot_experiment(exp_key, results, outdir, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    title, names, start = EXPERIMENTS[exp_key]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [('new_infections', 'New infections per day'),
              ('cum_infections', 'Cumulative infections'),
              ('cum_deaths', 'Cumulative deaths')]
    colors = plt.cm.tab10.colors

    for ax, (key, ylabel) in zip(axes, panels):
        for i, name in enumerate(names):
            arr = results[name][key]
            mean, lo, hi = mean_ci(arr)
            days = np.arange(arr.shape[1])
            ax.plot(days, mean, color=colors[i], lw=1.6, label=SCENARIOS[name][0])
            ax.fill_between(days, lo, hi, color=colors[i], alpha=0.25, lw=0)
        if start is not None:
            ax.axvline(start, color='k', ls='--', lw=0.8)
        ax.set_xlabel('Day')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc='upper right')
    n = results[names[0]]['new_infections'].shape[0]
    fig.suptitle(f'{title} -- mean of {n} runs with {int(CONFIDENCE*100)} % confidence band '
                 f'({args.pop_size:,.0f} agents)', fontsize=10)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(outdir, f'{exp_key}.{ext}'), dpi=200)
    plt.close(fig)


def summary_table(results, total, outdir):
    rows = []
    for name, r in results.items():
        n = r['new_infections'].shape[0]
        ci = lambda x: (x.mean(), stats.t.ppf(0.5 + CONFIDENCE / 2, n - 1) * x.std(ddof=1) / np.sqrt(n))
        inf_m, inf_h = ci(r['cum_infections'][:, -1])
        dth_m, dth_h = ci(r['cum_deaths'][:, -1])
        pk_m, pk_h = ci(r['new_infections'].max(axis=1))
        rows.append(dict(
            scenario=name, label=SCENARIO_LABEL(name), runs=n,
            infections_mean=inf_m, infections_ci=inf_h,
            deaths_mean=dth_m, deaths_ci=dth_h,
            peak_mean=pk_m, peak_ci=pk_h,
            run_time_mean_s=r['run_times'].mean(), run_time_sd_s=r['run_times'].std(ddof=1),
            scenario_wall_s=float(r['wall']),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, 'summary.csv'), index=False)

    # LaTeX fragment (paste into the report)
    lines = [r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
             r'Scenario & Infections & Deaths & Peak/day & Run time (s)\\', r'\midrule']
    for _, row in df.iterrows():
        lines.append(f"{row['label']} & {row['infections_mean']:.0f}$\\pm${row['infections_ci']:.0f} & "
                     f"{row['deaths_mean']:.1f}$\\pm${row['deaths_ci']:.1f} & "
                     f"{row['peak_mean']:.0f}$\\pm${row['peak_ci']:.0f} & "
                     f"{row['run_time_mean_s']:.2f}\\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    with open(os.path.join(outdir, 'summary_table.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    with open(os.path.join(outdir, 'timing.txt'), 'w') as f:
        f.write(f'Total time of the simulation experiment: {total:.1f} s\n')
        f.write(f'Number of runs: {int(df["runs"].sum())}\n')
        f.write(f'Mean run time: {np.concatenate([r["run_times"] for r in results.values()]).mean():.2f} s\n')
    return df


def SCENARIO_LABEL(name):
    return SCENARIOS[name][0]


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pop-size', type=int, default=50000, help='number of agents')
    ap.add_argument('--pop-infected', type=int, default=50, help='initially infected agents')
    ap.add_argument('--n-days', type=int, default=150, help='simulation length in days')
    ap.add_argument('--n-runs', type=int, default=20, help='replications per scenario (>= 20 required)')
    ap.add_argument('--seed', type=int, default=1, help='base random seed')
    ap.add_argument('--parallel', action='store_true', help='run replications in parallel')
    ap.add_argument('--cpus', type=int, default=None, help='number of CPU cores to use when --parallel')
    ap.add_argument('--outdir', default='m2_results', help='output directory')
    ap.add_argument('--plot-only', action='store_true', help='skip simulations, re-plot saved results')
    args = ap.parse_args()

    print(f'Covasim {cv.__version__}; {args.pop_size:,} agents, {args.n_runs} runs per scenario, '
          f'{args.n_days} days, parallel={args.parallel}')
    if args.plot_only:
        results, total = load_results(args)
    else:
        results, total = run_all(args)

    for exp_key in EXPERIMENTS:
        plot_experiment(exp_key, results, args.outdir, args)
    df = summary_table(results, total, args.outdir)
    pd.set_option('display.width', 200)
    print(df[['label', 'infections_mean', 'deaths_mean', 'peak_mean', 'run_time_mean_s']].round(1).to_string(index=False))
    print(f'Outputs written to {args.outdir}/')


if __name__ == '__main__':
    main()
