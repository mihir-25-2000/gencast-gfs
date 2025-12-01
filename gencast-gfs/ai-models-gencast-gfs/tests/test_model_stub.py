import pytest

pytest.importorskip("ai_models_gfs.model")
from ai_models_gencast_gfs import model as gencast_mod


@pytest.mark.skipif(not gencast_mod.HAS_DEPS, reason="GenCast/JAX stack not installed")
def test_gencast_model_instantiation():
    m = gencast_mod.GenCast0p25deg()
    assert m.expver in {"genc", "ge10"}
    assert "lsm" in m.constant_fields
