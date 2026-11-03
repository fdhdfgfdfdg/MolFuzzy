"""Deprecated import bridge. The implemented algorithm is convex relaxation."""
import warnings
from .balancing import BalanceResult as QLearningResult
from .balancing import balance_experts as _balance


def balance_experts(matrices,weights,benchmark=None,**kwargs):
    if benchmark is not None:
        raise ValueError('v2 balances to weighted consensus; benchmark selection was removed')
    warnings.warn('molfuzzy.qlearning is deprecated; use molfuzzy.balancing',DeprecationWarning,stacklevel=2)
    return _balance(matrices,weights,**kwargs)
