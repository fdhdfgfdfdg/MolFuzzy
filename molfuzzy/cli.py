"""JSON CLI with validation errors on stderr and machine-readable stdout."""
import argparse
from dataclasses import asdict
import json
import sys
import os
import tempfile
from pathlib import Path
from .pipeline import DecisionPipeline,jsonable
from .sensitivity import run_sensitivity_analysis,rankings_are_stable,perturbation_analysis


def _load_scenario(path):
    def reject_constant(value):
        raise ValueError(f'nonfinite JSON constant {value} is not allowed')
    def pairs(items):
        out={}
        for k,v in items:
            if k in out: raise ValueError(f'duplicate JSON key: {k}')
            out[k]=v
        return out
    with open(path,encoding='utf-8') as f:
        data=json.load(f,parse_constant=reject_constant,object_pairs_hook=pairs)
    required={'criterion_labels','alternative_labels','criteria_matrices','decision_matrices'}
    allowed=required|{'criterion_types','expert_weights','criterion_weights'}
    if not isinstance(data,dict) or required-set(data) or set(data)-allowed:
        raise ValueError('scenario must contain the four required label/matrix fields and only supported optional fields')
    return data


def _write_output(source, destination, text, force=False):
    """Publish complete output; protect input aliases and existing files."""
    src, dst = Path(source), Path(destination)
    def check():
        if src.resolve() == dst.resolve() or (dst.exists() and os.path.samefile(src, dst)):
            raise ValueError('output must not refer to the input scenario, even with --force')
        if dst.is_symlink():
            raise ValueError('output must not be a symbolic link')
        if dst.exists() and not force:
            raise ValueError('output already exists; use --force to replace it')
    check()
    fd, temporary = tempfile.mkstemp(prefix='.molfuzzy-', suffix='.tmp', dir=dst.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        check()
        if force:
            os.replace(temporary, dst)
        else:
            # Atomic no-clobber publication on the same filesystem.
            os.link(temporary, dst)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_parser():
    p=argparse.ArgumentParser(prog='molfuzzy')
    sub=p.add_subparsers(dest='command',required=True)
    for cmd in ('run','sensitivity','robustness'):
        s=sub.add_parser(cmd)
        s.add_argument('scenario')
        s.add_argument('--json',action='store_true',help='emit only JSON to stdout')
        s.add_argument('--output',help='write the complete JSON result to this path')
        s.add_argument('--force',action='store_true',help='allow replacing an existing output file, never the input')
        s.add_argument('--seed',type=int,default=42)
        if cmd=='run':
            s.add_argument('--geometry',default='linear')
            s.add_argument('--learning-rate',type=float,default=.1)
            s.add_argument('--sequence',action='store_true',help='also analyze angular traversal, not preference')
            s.add_argument('--aco-heuristic',action='store_true',help='disable exact traversal enumeration')
        elif cmd=='robustness':
            s.add_argument('--trials',type=int,default=100)
        else:
            s.add_argument('--sequence',action='store_true')
    return p


def main(argv=None):
    args=build_parser().parse_args(argv)
    try:
        s=_load_scenario(args.scenario)
        common={k:s[k] for k in ('criterion_types','expert_weights','criterion_weights') if k in s}
        if args.command=='run':
            r=DecisionPipeline(s['criterion_labels'],s['alternative_labels'],geometry=args.geometry,
                learning_rate=args.learning_rate,aco_seed=args.seed,include_sequence=args.sequence,
                aco_options={'exact_threshold':0} if args.aco_heuristic else None,**common).run(s['criteria_matrices'],s['decision_matrices'])
            payload=r.to_dict(); summary=r.summary()
        elif args.command=='sensitivity':
            rows=run_sensitivity_analysis(s['criterion_labels'],s['alternative_labels'],s['criteria_matrices'],s['decision_matrices'],seeds=(args.seed,),include_sequence=args.sequence,**common)
            payload={'rows':[asdict(r) for r in rows],
                'alternative_rankings_stable':rankings_are_stable(rows,'alternatives'),
                'criterion_orders_stable':rankings_are_stable(rows,'criteria'),
                'interpretation':'Geometry invariance is algebraic; learning rate changes relaxation speed, not the analytic consensus.'}
            summary=json.dumps(payload,indent=2)
        else:
            payload=perturbation_analysis(s['criterion_labels'],s['alternative_labels'],s['criteria_matrices'],s['decision_matrices'],trials=args.trials,seed=args.seed,**common)
            summary=json.dumps(payload,indent=2)
        encoded=json.dumps(jsonable(payload),indent=2,allow_nan=False)
        if args.output:
            _write_output(args.scenario,args.output,encoded+'\n',args.force)
        print(encoded if args.json else summary)
        return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:
        print(f'molfuzzy: {exc}',file=sys.stderr)
        return 2

if __name__=='__main__':
    sys.exit(main())
