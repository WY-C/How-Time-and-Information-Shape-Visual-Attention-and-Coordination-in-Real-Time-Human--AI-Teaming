import pandas as pd
import numpy as np
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("gaze", HERE/"01_gaze_reanalysis.py")
gaze=importlib.util.module_from_spec(spec); spec.loader.exec_module(gaze)

def test_h1_h4_exactly_partitions_information():
    overall=pd.DataFrame([{
        "pid":"P01","trial_id":"T1","Time":200,"Horizon":4,"Layout":"A",
        "total_fixation_ms":1000.0,"information_ms":200.0,
        "h1_ms":80.0,"h2_ms":60.0,"h3_ms":40.0,"h4_ms":20.0,
        "h1_prop_total":.08,"h2_prop_total":.06,"h3_prop_total":.04,"h4_prop_total":.02,
    }])
    raw,_=gaze.future_step_inputs(overall)
    assert abs(raw.dwell_ms.sum()-200.0)<1e-12
    assert abs(raw.relative_allocation.sum()-1.0)<1e-12
    assert np.allclose(raw.dwell_prop_total.values,[.08,.06,.04,.02])

def test_near_far_exactly_partitions_information():
    overall=pd.DataFrame([{
        "pid":"P01","trial_id":"T1","Time":200,"Horizon":4,"Layout":"A",
        "total_fixation_ms":1000.0,"information_ms":200.0,
        "h1_ms":80.0,"h2_ms":60.0,"h3_ms":40.0,"h4_ms":20.0,
        "h1_prop_total":.08,"h2_prop_total":.06,"h3_prop_total":.04,"h4_prop_total":.02,
    }])
    raw,_=gaze.future_step_inputs(overall)
    nf,_=gaze.near_far_inputs(raw)
    assert abs(nf.dwell_ms.sum()-200.0)<1e-12
    assert abs(nf.relative_allocation.sum()-1.0)<1e-12
