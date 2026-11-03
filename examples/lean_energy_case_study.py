"""Run after installing the package: python examples/lean_energy_case_study.py."""
import json
from pathlib import Path
from molfuzzy import DecisionPipeline


def main():
    s=json.loads(Path(__file__).with_name('lean_energy_scenario.json').read_text())
    p=DecisionPipeline(s['criterion_labels'],s['alternative_labels'],include_sequence=True)
    print(p.run(s['criteria_matrices'],s['decision_matrices']).summary())

if __name__=='__main__': main()
