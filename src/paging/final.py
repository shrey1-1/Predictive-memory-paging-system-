"""
The final system, in one place (after Phases 9-16):
  readahead (confidence = run/(run+1)) + delta-Markov hybrid predictor
  -> fault-delta decider fed by a shadow directory (page IDs only), frozen controller
  -> LRU (or any policy) underneath.
"""
from .config import CONTROLLER_FROZEN
from .deciders import CostAwareDirectory
from .engine import LRU
from .predictors import make_predictor, make_readahead, StreamMarkovPredictor
from .simulate import run_prefetch_sim_dir


def make_final_predictor(n_frames):
    return StreamMarkovPredictor(n_frames, markov=make_predictor('delta-hybrid'),
                                 readahead=make_readahead(n_frames, conf_k=1))


def make_final_decider(sample=1.0):
    return CostAwareDirectory(sample=sample, **CONTROLLER_FROZEN)


def run_final(trace, n_frames, policy_cls=LRU, log=None, sample=1.0):
    return run_prefetch_sim_dir(trace, n_frames, policy_cls, make_final_predictor(n_frames),
                                make_final_decider(sample), log=log)
