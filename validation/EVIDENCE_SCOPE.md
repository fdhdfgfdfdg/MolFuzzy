# Validation evidence scope

The current numerical record is validation/results.json, regenerated with MolFuzzy
2.0.10 on 3 November 2026, Python 3.12.14 and NumPy 2.3.5. environment.txt
and reproduce-current.txt belong to that run. It includes the complete fixture,
15 geometry/rate combinations, 100 perturbations, leave-one-expert-out results,
20 one-factor scenarios, 100 heuristic seeds and 30-sample timing measurements.

historical-2.0.7-results.json and historical-environment.txt retain the supplied
earlier numerical evidence. Other pre-existing logs retain their original scope:
source-210.txt is the supplied 2.0.10 source-test log; source-209.txt is older.
The supplied build/installed-wheel logs under validation/ are historical records,
not this run. The complete validation/ inventory is frozen into the rebuilt sdist.

Historical 2.0.10 rebuilt-wheel test and packaging logs are in release-validation/ in the
full ZIP, outside the sdist to avoid self-referential builds. Supplied post-build
logs remain in release-validation/historical/. These bundled local logs do not include Windows or macOS. Remote CI outcomes
are recorded separately in the public repository.
