from .gcg import GCG
from .gcq import GCQ
from .beast import BEAST
from .autodan_zhu import AutoDANZhu
from .pair import PAIR
from .deepinception import DeepInception


__attacker_zoo__ = {
    "gcg"           : GCG,
    "gcq"           : GCQ,
    "beast"         : BEAST,
    "autodan-zhu"   : AutoDANZhu,
    "pair"          : PAIR,
    "deepinception" : DeepInception,
}

def build_attacker(name: str, **kwargs):
    return __attacker_zoo__[name](**kwargs)
