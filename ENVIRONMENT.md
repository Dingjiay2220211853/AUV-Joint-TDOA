# Environment and dependencies

## 1. Tested environment

The Windows reference environment used to check the experiment code is listed below.

| Component | Version / requirement |
|---|---|
| Operating system | Windows 11, 64-bit |
| Python | 3.10.20 |
| GTSAM | conda-forge 4.2.0, Windows build `py310h1939499_14` |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| pandas | 2.3.3 |
| Matplotlib | 3.10.8 |
| tqdm | 4.67.3 |

The published experiments ran on Windows. Use the module invocations below
so the multiprocessing workers can import their entry points.

## 2. Recommended Conda installation

From the repository root:

```powershell
conda env create -f environment.yml
conda activate joint-tdoa
conda list gtsam
```

To update an existing copy of this environment:

```powershell
conda env update -n joint-tdoa -f environment.yml --prune
```

The conda-forge package supplies the compiled Python bindings. On Windows the
bindings in this build report `4.2a9` as Python package metadata, while
`conda list gtsam` reports `4.2.0`. The PyPI `4.2a9` release has no Windows
wheel. See the [conda-forge package](https://anaconda.org/conda-forge/gtsam)
and the [PyPI release files](https://pypi.org/project/gtsam/4.2a9/#files).

## 3. pip installation

If you already have Python 3.10 and compatible GTSAM 4.2 bindings that can
be imported with NumPy 2.2, the remaining packages can be installed with:

```powershell
python -m pip install -r requirements.txt
```

Matplotlib is used by plotting commands. A simulation-only run uses GTSAM, NumPy, SciPy, pandas and tqdm.
The directional-schedule processor can also run as a script in a separate
environment with NumPy, pandas and Matplotlib. Use `--no-plots` to generate
tables and the report without importing Matplotlib.

## 4. Verification

Check the bindings and numeric arrays:

```powershell
python -c "import gtsam, numpy, scipy, pandas, matplotlib, tqdm; assert gtsam.Pose2().retract(numpy.array([1., 0., 0.])).x() == 1.; print('GTSAM and NumPy OK')"
```

From `GTSAM_v2/`, run one complete seed:

```powershell
python -u -m experiments.master_follower_paper.formal_experiment_6_directional_joint_schedules `
  --mode smoke --start-seed 1 --seed-count 1 --workers 1 `
  --output experiments/master_follower_paper/results/smoke_directional_joint
```

## 5. Runtime and storage

- Formal execution defaults to 12 worker processes; use fewer workers on
  machines with limited RAM.
- The reference 1000-seed, five-method run took about 3 h 38 min with 12
  workers. Its checkpoint was approximately 1.2 GB. Reserve at least 3 GB
  for one formal run and its analysis exports.
- Results use SQLite transactional checkpoints.  Reusing the same output path
  resumes completed seeds only when the frozen configuration fingerprint is
  identical.
- Generated simulation results and Python caches are excluded from Git.
