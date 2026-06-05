import numpy as np
import jax.numpy as jnp

from astronomix._physics_modules._shock_finder._shock_zones import (
    get_post_pre_shock_values,
    identify_shock_zones,
)
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    SimulationConfig,
)
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.data_classes.simulation_helper_data import HelperData


def test_get_post_pre_shock_values_positive_direction():
    shock_direction = jnp.array([0.0, 1.0, 1.0, 1.0, 0.0])

    pressure = jnp.array([10.0, 9.0, 8.0, 4.0, 3.0])
    temperature = pressure

    p_post, p_pre, T_post, T_pre = get_post_pre_shock_values(
        shock_direction=shock_direction,
        pressure=pressure,
        temperature=temperature,
    )

    np.testing.assert_allclose(
        np.asarray(p_post),
        np.array([10.0, 9.0, 8.0]),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        np.asarray(p_pre),
        np.array([8.0, 4.0, 3.0]),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        np.asarray(T_post),
        np.array([10.0, 9.0, 8.0]),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        np.asarray(T_pre),
        np.array([8.0, 4.0, 3.0]),
        rtol=1e-6,
    )
    
def test_get_post_pre_shock_values_negative_direction():
    shock_direction = jnp.array([0.0, -1.0, -1.0, -1.0, 0.0])

    pressure = jnp.array([3.0, 4.0, 8.0, 9.0, 10.0])
    temperature = pressure

    p_post, p_pre, T_post, T_pre = get_post_pre_shock_values(
        shock_direction=shock_direction,
        pressure=pressure,
        temperature=temperature,
    )

    np.testing.assert_allclose(
        np.asarray(p_post),
        np.array([8.0, 9.0, 10.0]),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        np.asarray(p_pre),
        np.array([3.0, 4.0, 8.0]),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        np.asarray(T_post),
        np.array([8.0, 9.0, 10.0]),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        np.asarray(T_pre),
        np.array([3.0, 4.0, 8.0]),
        rtol=1e-6,
    )
def test_identify_shock_zones_detects_converging_shock():
    config = SimulationConfig(
        geometry=CARTESIAN,
        dimensionality=1,
        num_cells=7,
        grid_spacing=1.0,
    )

    registered_variables = RegisteredVariables()
    helper_data = HelperData(geometric_centers=None)

    density = jnp.array([4.0, 4.0, 4.0, 2.0, 1.0, 1.0, 1.0])
    pressure = jnp.array([20.0, 20.0, 20.0, 5.0, 1.0, 1.0, 1.0])

    # Velocity decreases from left to right.
    # This creates converging/compressing flow.
    velocity = jnp.array([2.0, 1.0, 0.0, -1.0, -2.0, -2.0, -2.0])

    primitive_state = jnp.stack([density, velocity, pressure])
    shock_direction = jnp.ones(7)

    shock_zones = identify_shock_zones(
        primitive_state=primitive_state,
        config=config,
        registered_variables=registered_variables,
        helper_data=helper_data,
        shock_direction=shock_direction,
        mach_min=1.3,
    )

    assert shock_zones.shape == pressure.shape
    assert shock_zones.dtype == jnp.bool_
    assert bool(jnp.any(shock_zones))
       
def test_identify_shock_zones_returns_false_when_flow_not_converging():
    config = SimulationConfig(
        geometry=CARTESIAN,
        dimensionality=1,
        num_cells=7,
        grid_spacing=1.0,
    )

    registered_variables = RegisteredVariables()
    helper_data = HelperData(geometric_centers=None)

    density = jnp.array([4.0, 4.0, 4.0, 2.0, 1.0, 1.0, 1.0])
    pressure = jnp.array([20.0, 20.0, 20.0, 5.0, 1.0, 1.0, 1.0])

    # Velocity increases from left to right.
    # This is not converging flow, so no shock zone should be detected.
    velocity = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0, 2.0, 2.0])

    primitive_state = jnp.stack([density, velocity, pressure])
    shock_direction = jnp.ones(7)

    shock_zones = identify_shock_zones(
        primitive_state=primitive_state,
        config=config,
        registered_variables=registered_variables,
        helper_data=helper_data,
        shock_direction=shock_direction,
        mach_min=1.3,
    )

    assert shock_zones.shape == pressure.shape
    assert shock_zones.dtype == jnp.bool_
    assert not bool(jnp.any(shock_zones))