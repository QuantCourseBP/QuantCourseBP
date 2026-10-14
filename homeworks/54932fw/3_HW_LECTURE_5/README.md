# Homework 3 (Lecture 5): Digital options

Carlos Chipantiza (Neptun EACN90, course id 54932fw) - Financial Processes (BMETE95MM14), BME, Fall 2026.

Started from the lecture's `solution` folder (as asked in the assignment); the NEW CODE is at the end of
`contract.py`, `pricer.py` and `test_pricer.py`, marked with `# change # k` / `#Delete.` / `# replace by` / `# end change # k`.
Status: Q1 and Q2 done.

| File | Content |
|---|---|
| `3_Homework_Lesson05.pdf` (`.tex`) | report: results used (with proofs and references), Q1 and Q2 code, flowcharts, block diagrams, outputs, figure |
| `3_HW_EXE_1.ipynb` | Q1: professor's code + test, NEW VERSION of the contract and of the pricer + tests, unit tests; block diagrams after each step |
| `3_HW_EXE_2.ipynb` | Q2: bull spreads with the lecture pricer, digital with the Q1 pricer, table and figure |
| `contract.py` | lecture library + NEW `EuropeanDigitalContract` (payoff `direction * 1{S_T > K}` or `1{S_T < K}`) |
| `pricer.py` | lecture library + NEW `EuropeanDigitalAnalyticPricer` (fair value `DF N(d2)`, `DF N(-d2)` and 5 Greeks) |
| `test_pricer.py` | professor's `TestPutCallParity` + NEW tests of the digital (fair value, payoff, type checks, Greeks) |
| `market_data.py`, `model.py`, `GOOG.csv` | lecture library, unchanged |
| `q1_digital_demo.py` / `.out` | Q1 run: payoffs, fair values, parity, bull-spread bounds, Greeks vs bumps |
| `test_pricer.out` | output of the unit tests: `127 passed` |
| `q2_bull_spread.py` / `.out` | Q2 run: table R_h <= C_d <= L_h and the figure |
| `figures/q2_digital_vs_bull_spreads.pdf` (`.png`) | Q2 figure: (a) payoffs, (b) prices against the spot, (c) error against h |
| `plotter.py` | Homework 1 (HW 2/1): `my_plotter`, used for the Q2 figure |

## How to run

The library imports `src.enums` from the repository, so run from the repository folder:

```bash
cd ~/proyectos/QuantCourseBP
source venv_quant_course_v312/bin/activate
python -m pytest -q homeworks/54932fw/3_HW_LECTURE_5/test_pricer.py      # 127 passed
python homeworks/54932fw/3_HW_LECTURE_5/q1_digital_demo.py               # = q1_digital_demo.out
python homeworks/54932fw/3_HW_LECTURE_5/q2_bull_spread.py                # = q2_bull_spread.out, figures/
```

Notebooks: open `3_HW_EXE_1.ipynb` / `3_HW_EXE_2.ipynb` in VS Code with the kernel `venv_quant_course_v312`, Run All
(the first cell puts this folder and the repository folder on the path).
Colab: also copy `src/enums.py` and an empty `src/__init__.py` into the Drive folder `.../HOMEWORKS/3_HW_LECTURE_5`.
