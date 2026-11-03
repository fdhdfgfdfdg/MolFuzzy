"""MolFuzzy 2: validated fuzzy preference scoring and traversal diagnostics."""
__version__='2.0.11'
from .mfv import MFV,WeightedTriple,GEOMETRIES,register_geometry
from .pipeline import DecisionPipeline,PipelineResult
from .ranking import rank_alternatives,RankingResult
from .balancing import balance_experts,BalanceResult
from .aco import aco_sequence,ACOResult
__all__=['MFV','WeightedTriple','GEOMETRIES','register_geometry','DecisionPipeline',
    'PipelineResult','rank_alternatives','RankingResult','balance_experts','BalanceResult',
    'aco_sequence','ACOResult']
