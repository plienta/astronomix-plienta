import numpy as np
import jax.numpy as jnp

from astronomix._physics_modules._shock_finder._shock_direction import (
    _calculate_shock_direction,
)
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    SimulationConfig,
)

def test_calculate_shock_direction_hot_to_cold():
    config = SimulationConfig(
        geometry=CARTESIAN,
        dimensionality=1,
        num_cells=7,
        grid_spacing=1.0,
    )

    density = jnp.ones(7)

    # temperature = pressure / density
    # hot on the left, cold on the right
    pressure = jnp.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0])

    shock_direction = _calculate_shock_direction(
        pressure=pressure,
        density=density,
        config=config,
    )

    np.testing.assert_allclose(
        np.asarray(shock_direction[1:-1]),
        np.ones(5),
        rtol=1e-6,
    )
    
def test_calculate_shock_direction_cold_to_hot():
    config = SimulationConfig(
        geometry=CARTESIAN,
        dimensionality=1,
        num_cells=7,
        grid_spacing=1.0,
    )

    density = jnp.ones(7)

    # temperature = pressure / density
    # cold on the left, hot on the right
    pressure = jnp.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    shock_direction = _calculate_shock_direction(
        pressure=pressure,
        density=density,
        config=config,
    )

    np.testing.assert_allclose(
        np.asarray(shock_direction[1:-1]),
        -np.ones(5),
        rtol=1e-6,
    )