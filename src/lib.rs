mod dfo;
mod fo;
mod fo_grouping;
mod chrg_event;
mod disaggregation;
mod uncontrolled;
mod virtual_battery;

use pyo3::prelude::*;

#[pymodule]
fn ev_agg_bench_rs(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<chrg_event::ChrgEvent>()?;
    m.add_function(wrap_pyfunction!(disaggregation::disaggregate, m)?)?;
    m.add_function(wrap_pyfunction!(uncontrolled::run_unctr, m)?)?;
    register_dfo(py, m)?;
    register_fo(py, m)?;
    register_vb(py, m)?;
    Ok(())
}

fn register_dfo(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "dfo")?;
    m.add_class::<dfo::DFO>()?;
    m.add_class::<dfo::FlexModel>()?;
    m.add_function(wrap_pyfunction!(dfo::aggregate_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(dfo::disaggregate_and_add, m)?)?;
    parent_module.add_submodule(m)?;
    Ok(())
}

fn register_fo(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "fo")?;
    m.add_function(wrap_pyfunction!(fo::pipeline, m)?)?;
    parent_module.add_submodule(m)?;
    Ok(())
}

fn register_vb(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "vb")?;
    m.add_class::<virtual_battery::VBSlice>()?;
    m.add_function(wrap_pyfunction!(virtual_battery::get_vb_params, m)?)?;
    m.add_function(wrap_pyfunction!(virtual_battery::group, m)?)?;
    parent_module.add_submodule(m)?;
    Ok(())
}