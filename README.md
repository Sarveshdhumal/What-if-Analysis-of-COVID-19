# What-if Analysis of COVID-19 Interventions with Covasim

Code for milestone **M2 (What-if Analysis)** of the course *Data Driven Modeling and Simulation*
(University of Rostock, winter semester 2026/27, Group 4).

The script `m2_whatif.py` uses the agent-based model [Covasim](https://github.com/InstituteforDiseaseModeling/covasim)
to compare how the **type**, **strictness**, **timing**, and **combination** of interventions change the
dynamics of an epidemic. Every scenario is simulated with 20 stochastic replications. The script plots the
mean trajectories with 90 % confidence bands and measures the run time of every run and of the whole experiment.

All experiments are what-if analyses on a synthetic population with Covasim's default parameters. They are
not forecasts of a real outbreak.

## What the script does

1. Defines 11 scenarios from four building blocks: distancing, school closure, testing, and contact tracing.
2. Runs every scenario with 20 replications (random seeds 1 to 20; the same seeds in all scenarios, so the
   replications are paired across scenarios).
3. Computes, for each day, the mean over the replications and the 90 % confidence interval of the mean
   (t-distribution, 19 degrees of freedom).
4. Writes four figures (one per experiment), a summary table, and the run-time measurements.

### Simulation setup

- 50,000 agents, "hybrid" contact network (households, schools, workplaces, community)
- Start on 1 March 2020, 150 days, 50 initially infected agents
- All other parameters are Covasim defaults
- Cumulative infections can exceed the population size, because Covasim also counts reinfections

### Interventions

- **Distancing, masks, hygiene:** the transmission probability beta is reduced by 20, 40, or 60 % in all
  layers (`cv.change_beta`).
- **School closure:** all contacts in the school layer are removed (`cv.clip_edges`).
- **Testing:** symptomatic agents are tested with probability 0.1 per day, asymptomatic agents with 0.001,
  results arrive after 1 day (`cv.test_prob`).
- **Contact tracing and quarantine:** contacts are traced with probability 1.0 (household), 0.8 (school),
  0.6 (work), and 0.1 (community) (`cv.contact_tracing`).

Interventions start on day 30 unless stated otherwise.

### Experiments

| Experiment | Question | Scenarios (plus baseline without intervention) |
|---|---|---|
| E1 `E1_type` | Which type of intervention? | distancing -40 %, school closure, test + trace |
| E2 `E2_strictness` | How strict? | distancing -20 %, -40 %, -60 % |
| E3 `E3_timing` | How early? | distancing -40 % starting on day 15, 30, 45 |
| E4 `E4_combination` | Which combination? | distancing, school closure, test + trace, and their combinations |

## Requirements

- Python 3 (tested with Covasim 4.0.2; other versions were not tested)
- Packages: `covasim`, `scipy`, `matplotlib`, `pandas`

```
pip install covasim==4.0.2 scipy matplotlib pandas
```

A virtual environment is recommended:

```
python3 -m venv m2env
source m2env/bin/activate
pip install covasim==4.0.2 scipy matplotlib pandas
```

## Usage

Quick test (smaller population and fewer runs, under a minute on one core):

```
python m2_whatif.py --pop-size 5000 --n-runs 4 --n-days 100 --outdir test
```

Full experiment as used in the report (parallel, 16 cores):

```
python m2_whatif.py --parallel --cpus 16 --outdir m2_results
```

Re-create the figures and tables from saved results without rerunning the simulations:

```
python m2_whatif.py --plot-only --outdir m2_results
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `--pop-size` | 50000 | number of agents |
| `--pop-infected` | 50 | initially infected agents |
| `--n-days` | 150 | simulation length in days |
| `--n-runs` | 20 | replications per scenario (at least 20 are required for the course) |
| `--seed` | 1 | base random seed (run i uses seed + i) |
| `--parallel` | off | run the replications of a scenario in parallel |
| `--cpus` | all | number of CPU cores to use with `--parallel` |
| `--outdir` | `m2_results` | output directory |
| `--plot-only` | off | skip the simulations and use the saved raw results |

## Output files

All files are written to the output directory:

- `E1_type`, `E2_strictness`, `E3_timing`, `E4_combination` (`.png` and `.pdf`): mean trajectories with 90 %
  confidence bands for new infections per day, cumulative infections, and cumulative deaths
- `summary.csv`: final infections, deaths, peak of daily infections (mean and confidence half-width), and run
  times for every scenario
- `summary_table.tex`: the same table as a LaTeX fragment for the report
- `timing.txt`: total time of the experiment and mean run time
- `results_raw.npz`: all raw trajectories and run times (needed for `--plot-only`)

## Run times

The script measures the run time of every single simulation run, the wall-clock time of every scenario, and the
total time of the whole experiment. In our run with 50,000 agents and 20 runs for each of the 11 scenarios
(220 runs) on a server with 16 cores, the whole experiment took about 90 seconds in parallel mode, and a single
run took about 4 to 5 seconds. Run times depend on the hardware, so repeat the measurement on your own machine.

## Reproducibility

The random seeds are fixed (1 to 20), so a rerun with the same Covasim version produces the same numbers.

## Notes on the results

- The peak of the daily infections is the maximum over the whole simulation. For strong interventions, this
  maximum is reached before the interventions start on day 30, which is why several scenarios show the same value.
- The 90 % bands are confidence intervals of the mean over 20 replications, not prediction intervals of single runs.

## Adding your own scenario

Scenarios are defined in the dictionary `SCENARIOS` in `m2_whatif.py` as a label and a function that returns a
fresh list of interventions. For example, a 50 % distancing scenario starting on day 20:

```python
'beta50_d20': ('Distancing -50 %, start day 20', lambda: flat(distancing(0.5, 20))),
```

Then add the scenario name to one of the experiments in the dictionary `EXPERIMENTS` and run the script again.

## Authors

Group 4: Aayush Mishra, Sarvesh Dhumal, Fabian Darrell Widyadhana Reswara, and Sarthak Dhasmana.

## Acknowledgments and license

This project uses Covasim: Kerr, C.C., Stuart, R.M., Mistry, D., et al., *Covasim: An agent-based model of
COVID-19 dynamics and interventions*, PLOS Computational Biology 17(7), e1009149 (2021),
https://doi.org/10.1371/journal.pcbi.1009149.

Add a license file if you want others to reuse the code, for example the MIT license.
