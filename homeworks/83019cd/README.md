# Financial Processes — Group 1

Student ID: **83019cd**

- [Assignment 1](assignment_1/assignment_1.ipynb): plot customization, call Greeks,
  put pricing and put–call parity.
- [Assignment 2](assignment_2/assignment_2.ipynb): calibration of a balanced
  binomial tree.
- [Assignment 3](assignment_3/assignment_3.ipynb): digital options and their
  approximation by bull call spreads.

Each folder contains the code, tests and an executed notebook with the results.
The notebooks identify the lecture material used as a starting point.

Run all homework tests from the repository root:

```sh
venv_quant_course_v312/bin/python -m pytest homeworks/83019cd -q
```

There are 244 passing tests: 61 for Assignment 1, 55 for Assignment 2 and 128
for Assignment 3 (including the 24 tests copied from the lecture solution).

The files `my_env.json`, `my_git.json` and `inv.csv` were generated with
`git_homework.py`, following `homeworks/git_homework_description.md`.
The revision in `my_git.json` is the source revision when that script ran.
The submission branch is `20261020_83019cd_homework`.
