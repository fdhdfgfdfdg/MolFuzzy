"""Optional TOPSIS adapter; a comparison, not validation of equivalent algorithms.
Install with pip install '.[interop]'. All linguistic columns here are benefits.
"""
import json
from pathlib import Path
import numpy as np
from molfuzzy import DecisionPipeline


def main():
    from pymcdm.methods import TOPSIS
    s=json.loads(Path(__file__).with_name('lean_energy_scenario.json').read_text())
    r=DecisionPipeline(s['criterion_labels'],s['alternative_labels']).run(s['criteria_matrices'],s['decision_matrices'])
    scores=TOPSIS()(r.ranking.utilities,r.criteria.weights,np.ones(len(s['criterion_labels'])))
    order=np.argsort(-scores)
    print('TOPSIS comparison:', ' > '.join(s['alternative_labels'][i] for i in order))
    print('MolFuzzy preference:', ' > '.join(r.ranking.ranking_labels))

if __name__=='__main__': main()
