# Homework 4 (Lecture 6): Numerical Greeks

Carlos Chipantiza, course id 54932fw. Report: `4_Homework_Lesson06.pdf` (source `4_Homework_Lesson06.tex`, self-contained).

## Files
- Library of Lecture 6: `contract.py`, `enums.py` (lecture folder) and `market_data.py`, `model.py`, `pricer.py` (`solution` folder).
  - NEW CODE at the end of `enums.py`: `FDMethod` (FORWARD, BACKWARD, CENTRAL), `BumpType` (ABSOLUTE, RELATIVE).
  - NEW VERSION of `Pricer` in `pricer.py` (changes # 1 - # 4): class variables `fd_method`, `bump_type`, `absolute_bump_size`;
    `calc_spot_bump_size`, `calc_fair_value_bumped_spot`; all combinations in `calc_delta` and `calc_gamma`.
    Defaults = the lecture (CENTRAL, RELATIVE, 1%).
- `q1_fd_greeks_demo.py` (+ `.out`): the six combinations on the ATM 1Y call of BLUECHIP_BANK, checks.
- `q1_bump_study.py` (+ `.out`): error against the bump size, 72 European options, absolute vs relative, optimal bumps;
  writes `figures/` (png and the csv data of the report's plots).
- `4_HW_EXE_1.ipynb`: the same code as notebook cells (professor's code + test, NEW CODE, NEW VERSION + test, study).

## Run (course venv, from this folder)
    python q1_fd_greeks_demo.py
    python q1_bump_study.py
The scripts read the market data from `QuantCourseBP/mkt` (searched in this folder and its parents).
For Colab, copy `QuantCourseBP/mkt` into this folder on Drive.

## Result
Central differences with a relative bump: best eps = 3e-6 (delta), 1e-4 (gamma), the same for every stock;
one common value eps = 1e-4 gives a worst relative error 4.1e-8 on the 72 options (lecture's 1%: 3.2e-4).
An absolute bump needs a different h for each spot level.
