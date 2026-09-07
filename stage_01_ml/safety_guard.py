import numpy as np
import pandas as pd


class GuardRailedPredictor:
    """Deterministic safety guard rail applied on top of the ML pipeline.

    Layered on top of the RandomForest pipeline so the deployed model can never
    under-warn below the regulatory thresholds set by the data engineering stage.
    The ML model handles generalisation within risk regions; the guard rail
    guarantees that the exact boundary rules are always respected.
    """
    def __init__(self, base_model):
        self.base_model = base_model

    def __getattr__(self, name):
        # Avoid recursion during unpickling – base_model is set via __setstate__
        if name == "base_model":
            raise AttributeError(name)
        try:
            base = object.__getattribute__(self, "base_model")
        except AttributeError:
            raise AttributeError(name)
        return getattr(base, name)

    def predict(self, X):
        pred = self.base_model.predict(X)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.base_model.feature_names_in_)
        riv   = X.reindex(columns=['river_level']).iloc[:, 0].fillna(0.0).to_numpy()
        rain  = X.reindex(columns=['rainfall_rolling_72h_sum']).iloc[:, 0].fillna(0.0).to_numpy()
        calls = X.reindex(columns=['emergency_call_volume']).iloc[:, 0].fillna(0.0).to_numpy()
        out = []
        for p, r, ra, c in zip(pred, riv, rain, calls):
            if r >= 4.5 or (ra >= 150.0 and r >= 3.5):
                out.append('SEVERE')
            elif r >= 3.0 or ra >= 80.0 or c >= 100.0:
                out.append('SEVERE' if p == 'SEVERE' else 'MODERATE')
            else:
                out.append(p)
        return np.asarray(out, dtype=object)
