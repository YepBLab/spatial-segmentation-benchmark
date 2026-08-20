# Contributing

1. Create a branch and keep changes limited to code, tests, documentation, and
   approved aggregate figures.
2. Run `python -m unittest discover -s tests -v` before opening a pull request.
3. Add a synthetic-mask regression test for every metric behavior change.
4. Do not commit private paths or data. Run the repository audit command from
   the README before pushing.
5. Describe any change to matching, exclusions, averaging, tolerances, or
   denominators prominently in the pull request.
