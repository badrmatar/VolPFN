"""Local-only adapters. No online model fitting on evaluation targets."""
from pathlib import Path
import numpy as np
from .metrics import QUANTILES
from .data import covariates


def checked(point,q,semantics):
    q=np.asarray(q,float).reshape(-1)
    if q.shape!=(9,) or not np.isfinite(q).all() or not np.isfinite(point):
        raise ValueError('Invalid native forecast shape/values')
    crossing=bool(np.any(np.diff(q)<0))
    # Monotone rearrangement is explicit and logged, not silently ignored.
    return dict(point=float(point),quantiles=np.sort(q),point_semantics=semantics,
                rearranged=crossing,native_density_available=False)


class Chronos:
    def __init__(self,path):
        from chronos import Chronos2Pipeline
        self.model=Chronos2Pipeline.from_pretrained(str(path),device_map='cpu',local_files_only=True)
        self.model.model.eval()

    def predict(self,past,track):
        import torch
        y=np.asarray(past,float)
        if track=='squared': y=y*y
        elif track!='raw': raise ValueError('Unknown track')
        q,point=self.model.predict_quantiles([torch.tensor(y,dtype=torch.float32)[None]],
            prediction_length=1,quantile_levels=QUANTILES.tolist(),batch_size=1,
            context_length=len(y),cross_learning=False)
        return checked(float(point[0].reshape(-1)[0]),q[0].reshape(-1).numpy(),
            'median: Chronos2 predict_quantiles mean field is implemented as q0.5')


class Moirai:
    def __init__(self,path):
        from uni2ts.model.moirai2 import Moirai2Module
        self.module=Moirai2Module.from_pretrained(str(path),local_files_only=True)

    def predict(self,past,track):
        import pandas as pd
        from gluonts.dataset.common import ListDataset
        from uni2ts.model.moirai2 import Moirai2Forecast
        y=np.asarray(past,float)
        if track=='squared': y=y*y
        elif track!='raw': raise ValueError('Unknown track')
        model=Moirai2Forecast(module=self.module,prediction_length=1,context_length=len(y),
            target_dim=1,feat_dynamic_real_dim=0,past_feat_dynamic_real_dim=0)
        predictor=model.create_predictor(batch_size=1,device='cpu')
        # Synthetic index labels encode observed sessions; no calendar features enter.
        ds=ListDataset([{'start':pd.Period('2000-01-01',freq='D'),'target':y.astype(np.float32)}],freq='D')
        f=next(iter(predictor.predict(ds)))
        q=np.array([np.asarray(f.quantile(float(p))).reshape(-1)[0] for p in QUANTILES])
        return checked(q[4],q,'median q0.5; no claim of native expectation')


class TabPFN:
    def __init__(self,path):
        if not Path(path).is_file():
            raise FileNotFoundError('User-authorized local TabPFN checkpoint required; no cloud fallback')
        from tabpfn_time_series import TabPFNTSPipeline,TabPFNMode
        from tabpfn_time_series.features import RunningIndexFeature
        self.model=TabPFNTSPipeline(max_context_length=1024,
            temporal_features=[RunningIndexFeature()],tabpfn_mode=TabPFNMode.LOCAL,
            tabpfn_output_selection='mean',tabpfn_model_config={
                'model_path':str(path),'device':'cpu','n_estimators':8,'random_state':9182026,
                'n_preprocessing_jobs':1})

    def predict(self,past,track):
        import pandas as pd
        past=np.asarray(past,float)
        x,xnext=covariates(past)
        y=past[1:].copy()
        if track=='squared': y=y*y
        elif track!='raw': raise ValueError('Unknown track')
        frame=pd.DataFrame({'item_id':'series','timestamp':pd.date_range('2000-01-02',periods=len(y),freq='D'),'target':y})
        future=pd.DataFrame({'item_id':['series'],'timestamp':[frame.timestamp.iloc[-1]+pd.Timedelta(days=1)]})
        for j in range(8):
            frame[f'causal_{j}']=x[:,j]
            future[f'causal_{j}']=xnext[:,j]
        pred=self.model.predict_df(frame,future_df=future,quantiles=QUANTILES.tolist())
        point_column='target' if 'target' in pred else 'predictions' if 'predictions' in pred else 'mean'
        q=[float(pred[float(p)].iloc[0]) for p in QUANTILES]
        return checked(float(pred[point_column].iloc[0]),q,'TabPFN-TS explicit mean output')


def make(name,path):
    return {'chronos2':Chronos,'moirai2':Moirai,'tabpfn_ts':TabPFN}[name](path)
