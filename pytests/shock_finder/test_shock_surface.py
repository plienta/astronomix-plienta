import numpy as np
import jax.numpy as jnp

from astronomix._physics_modules._shock_finder._shock_surface import (
    identify_shock_surface,
)
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    SimulationConfig,
)
from astronomix.variable_registry.registered_variables import RegisteredVariables

def test_identify_shock_surface_selects_one_surface_cell_per_zone():
    config = SimulationConfig(
        geometry=CARTESIAN,
        dimensionality=1,
        num_cells=10,
        grid_spacing=1.0,
    )

    registered_variables = RegisteredVariables()

    density = jnp.ones(10)
    pressure = jnp.ones(10)

    # Two separate shock-zone regions:
    # region 1: indices 2, 3, 4
    # region 2: indices 7, 8
    #
    # The function should choose one shock-surface cell
    # from each connected shock-zone region.
    velocity = jnp.array([
        0.0, 0.0, -1.0, -4.0, -2.0,
        0.0, 0.0, -1.0, -5.0, -2.0,
    ])

    primitive_state = jnp.stack([density, velocity, pressure])

    shock_zones = jnp.array([
        False, False, True, True, True,
        False, False, True, True, False,
    ])

    shock_direction = jnp.ones(10)

    shock_surface = identify_shock_surface(
        primitive_state=primitive_state,
        shock_zones=shock_zones,
        shock_direction=shock_direction,
        config=config,
        registered_variables=registered_variables,
    )

    assert shock_surface.shape == shock_zones.shape
    assert shock_surface.dtype == jnp.bool_

    # One selected surface cell in each connected shock-zone region.
    assert int(jnp.sum(shock_surface[2:5])) == 1
    assert int(jnp.sum(shock_surface[7:9])) == 1

    # No selected surface cells outside the shock zones.
    assert not bool(jnp.any(shock_surface[:2]))
    assert not bool(jnp.any(shock_surface[5:7]))
    assert not bool(jnp.any(shock_surface[9:]))
    
def test_identify_shock_surface_empty_zones_returns_all_false():
    config = SimulationConfig(
        geometry=CARTESIAN,
        dimensionality=1,
        num_cells=7,
        grid_spacing=1.0,
    )

    registered_variables = RegisteredVariables()

    density = jnp.ones(7)
    velocity = jnp.zeros(7)
    pressure = jnp.ones(7)

    primitive_state = jnp.stack([density, velocity, pressure])

    shock_zones = jnp.array([
        False, False, False, False, False, False, False
    ])

    shock_direction = jnp.ones(7)

    shock_surface = identify_shock_surface(
        primitive_state=primitive_state,
        shock_zones=shock_zones,
        shock_direction=shock_direction,
        config=config,
        registered_variables=registered_variables,
    )

    expected = np.array([
        False, False, False, False, False, False, False
    ])

    np.testing.assert_array_equal(
        np.asarray(shock_surface),
        expected,
    )