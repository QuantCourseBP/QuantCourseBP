# Homework 2 (Lecture 4): Implement calibrator for Tree

Carlos Chipantiza (Neptun EACN90, course id 54932fw) - Financial Processes (BMETE95MM14), BME, Fall 2026.

Part 1: Background (submitted, unchanged). Part 2: Task, the `calibrate` method. Consolidated report: both parts in one PDF.

## Reports

| File | Content |
|---|---|
| `2_Homework_Lesson04_Consolidated.pdf` (`.tex`) | Part 1 (as submitted) + Part 2 in one document; Part 2 cites the results of Part 1 by number |
| `2_Homework_Lesson04_Part2.pdf` (`.tex`) | Part 2 alone: plan, flowcharts, results used, Q1-Q3 with code, outputs and plots, Q4 pictures of the calibrated trees |
| `2_Homework_Lesson04.pdf` (`.tex`) | Part 1 alone, as submitted |

All `.tex` files are self-contained (code and plots inside); compile with `pdflatex` (twice).

## Notebooks

| File | Content |
|---|---|
| `2_HW_EXE_1.ipynb` | Part 1: all the code of the Part 1 write-up, same order |
| `2_HW_EXE_2.ipynb` | Part 2, Q1: professor's code + test, Homework 1 pricer, Part 1 code, NEW CODE `calibrate` + test and plot |
| `2_HW_EXE_3.ipynb` | Part 2, Q2: Demo A, same contract with N = 1, ..., 100 steps (scaled DF) + plots |
| `2_HW_EXE_4.ipynb` | Part 2, Q3: Demo B, one-step tree for 4 expiries x 5 strikes (flat rate) + plots |
| `2_HW_EXE_5.ipynb` | Part 2, Q4: pictures of the calibrated trees (spot tree + price tree), N = 2, 5, 50 |

## Code

| File | Content |
|---|---|
| `prof_tree.py` | professor's lecture code (`Binomial_tree_coding.ipynb`, cells 1, 3, 5, 8, 13), unchanged |
| `bg_functions.py` | Part 1 code (`balanced_tree_call_price`, `up_step_from_tree_vol`); one change for Part 2: `# change # 1` (imports, so the file can be imported) |
| `hw1_call.py` | Homework 1 (HW 2/2, NEW VERSION): Black-Scholes call pricer `black_scholes_eur_call` |
| `plotter.py` | Homework 1 (HW 2/1): `my_plotter` |
| `calibrator.py` | NEW (Part 2): `calibrate` (Steps 1-5), `price_range` (Step 2), `tree_vol` |
| `tree_plot.py` | NEW (Part 2): `calibrated_trees`, `important_nodes`, `plot_tree` |
| `q1_calibrate_test.py`, `demo_A.py`, `demo_B.py`, `trees_demo.py` | runnable code of Q1, Q2, Q3, Q4 (same code as the notebooks) |
| `q1_calibrate_test.out`, `demo_A.out`, `demo_B.out`, `trees_demo.out` | their printed outputs (as in the reports) |
| `run_all.py` | extra numbers for the report: writes `results.json`; its printout is `outputs.txt` |
| `figures/` | plots saved by the scripts (PDF), including `tree_N2.pdf`, `tree_N5.pdf`, `tree_N50.pdf` |

## Reading order

1. `2_Homework_Lesson04_Consolidated.pdf` (or Part 1 and Part 2 separately)
2. `2_HW_EXE_1.ipynb` (Part 1)
3. `2_HW_EXE_2.ipynb` to `2_HW_EXE_5.ipynb` (Part 2: Q1, Q2, Q3, Q4)

## How to run

- **VS Code**: open this folder, select the interpreter `venv_quant_course_v312`, open a notebook, pick that kernel, Run All.
- **Terminal** (from this folder):
  ```bash
  source ~/proyectos/QuantCourseBP/venv_quant_course_v312/bin/activate
  python q1_calibrate_test.py     # prints q1_calibrate_test.out
  python demo_A.py                # prints demo_A.out
  python demo_B.py                # prints demo_B.out
  python trees_demo.py            # prints trees_demo.out, saves figures/tree_N*.pdf
  ```
  Requirements: numpy, scipy, matplotlib.
- **Colab**: copy the `.py` files to the Drive folder `.../HOMEWORKS/2_HW_LECTURE_4`; the first cell of each notebook mounts Drive and moves there (locally it does nothing).
