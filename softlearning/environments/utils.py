#from .adapters.gym_adapter import (
#    GYM_ENVIRONMENTS,
#    GymAdapter,
#)

from .adapters.fetch_adapter import (
    FETCH_ENVIRONMENTS,
    FetchAdapter,
)
ENVIRONMENTS = {
    'fetch': FETCH_ENVIRONMENTS,
}

ADAPTERS = {
    'fetch': FetchAdapter,
}


def get_environment(universe, domain, task, env_params):
    return ADAPTERS[universe](domain, task, **env_params)


def get_environment_from_variant(variant):
    universe = variant['universe']
    task = variant['task']
    domain = variant['domain']
    env_params = variant['env_params']

    return get_environment(universe, domain, task, env_params)
