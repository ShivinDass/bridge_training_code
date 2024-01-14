from .continuous.bc import BCAgent
from .continuous.gc_bc import GCBCAgent
from .continuous.gc_ddpm_bc import GCDDPMBCAgent
from .continuous.gc_iql import GCIQLAgent
from .continuous.iql import IQLAgent
from .continuous.lc_bc import LCBCAgent
from .continuous.stable_contrastive_rl import StableContrastiveRLAgent
from .continuous.wrapped_bc import WrappedBCAgent

agents = {
    "gc_bc": GCBCAgent,
    "lc_bc": LCBCAgent,
    "gc_iql": GCIQLAgent,
    "gc_ddpm_bc": GCDDPMBCAgent,
    "bc": BCAgent,
    "iql": IQLAgent,
    "stable_contrastive_rl": StableContrastiveRLAgent,
    "flow_bc": WrappedBCAgent,
}
