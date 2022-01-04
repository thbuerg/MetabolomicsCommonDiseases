import numpy as np
import pandas as pd
from lifelines import CRCSplineFitter


def get_observed_probability(F_t, events, durations, t0: float):
    def ccl(p): return np.log(-np.log(1 - p))

    T = "time"
    E = "event"

    predictions_at_t0 = np.clip(F_t, 1e-10, 1 - 1e-10)
    prediction_df = pd.DataFrame({f"ccl_at_{t0}": ccl(predictions_at_t0), T: durations, E: events})

    if any(x <= 1 for x in events):
        pass
    else:
        prediction_df["event"] = [0 if v > 1 else v for v in prediction_df["event"].to_list()]

    index_old = prediction_df.index
    prediction_df = prediction_df.dropna()
    index_new = prediction_df.index
    diff = index_old.difference(index_new)

    knots = 3
    regressors = {"beta_": [f"ccl_at_{t0}"], **{f"gamma{i}_": "1" for i in range(knots)}}

    crc = CRCSplineFitter(knots, penalizer=0.001).fit(prediction_df, T, E, regressors=regressors, show_progress=False)

    risk_obs = (1 - crc.predict_survival_function(prediction_df, times=[t0])).T.squeeze()

    return risk_obs, diff.to_list()
