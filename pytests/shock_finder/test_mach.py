import numpy as np
import jax.numpy as jnp

from astronomix._physics_modules._shock_finder._mach import (
    _calculate_mach_at_surface,
)
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    SimulationConfig,
)
from astronomix.variable_registry.registered_variables import RegisteredVariables


def test_calculate_mach_at_surface_zero_outside_surface():
    config = SimulationConfig(
        geometry=CARTESIAN,
        dimensionality=1,
        num_cells=7,
        grid_spacing=1.0,
    )

    registered_variables = RegisteredVariables()

    density = jnp.ones(7)
    velocity = jnp.zeros(7)

    # Pressure jump around index 3.
    # For positive shock direction:
    # post-shock value is on the left, pre-shock value is on the right.
    pressure = jnp.array([16.0, 16.0, 16.0, 8.0, 4.0, 4.0, 4.0])

    primitive_state = jnp.stack([density, velocity, pressure])

    shock_surface = jnp.array([
        False, False, False, True, False, False, False
    ])

    shock_direction = jnp.ones(7)

    mach = _calculate_mach_at_surface(
        primitive_state=primitive_state,
        shock_surface=shock_surface,
        shock_direction=shock_direction,
        config=config,
        registered_variables=registered_variables,
    )

    assert mach.shape == pressure.shape
    assert mach[3] > 1.0

    expected_zero_mask = np.array([
        True, True, True, False, True, True, True
    ])

    np.testing.assert_allclose(
        np.asarray(mach)[expected_zero_mask],
        0.0,
        rtol=1e-6,
    )
    
def test_calculate_mach_at_surface_matches_pressure_jump_formula():
    config = SimulationConfig(
        geometry=CARTESIAN,
        dimensionality=1,
        num_cells=7,
        grid_spacing=1.0,
    )

    registered_variables = RegisteredVariables()

    density = jnp.ones(7)
    velocity = jnp.zeros(7)

    # At surface index 3, with positive shock direction:
    # post-shock pressure = left neighbor = 16
    # pre-shock pressure = right neighbor = 4
    pressure = jnp.array([16.0, 16.0, 16.0, 8.0, 4.0, 4.0, 4.0])

    primitive_state = jnp.stack([density, velocity, pressure])

    shock_surface = jnp.array([
        False, False, False, True, False, False, False
    ])

    shock_direction = jnp.ones(7)

    mach = _calculate_mach_at_surface(
        primitive_state=primitive_state,
        shock_surface=shock_surface,
        shock_direction=shock_direction,
        config=config,
        registered_variables=registered_variables,
    )

    gamma = 5.0 / 3.0
    pressure_ratio = 16.0 / 4.0

    expected_mach = np.sqrt(
        (pressure_ratio * (gamma + 1.0) + (gamma - 1.0))
        / (2.0 * gamma)
    )

    np.testing.assert_allclose(
        float(mach[3]),
        expected_mach,
        rtol=1e-6,
    )