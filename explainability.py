# -*- coding: utf-8 -*-
import numpy as np
from feature_extractor import get_feature_names


def get_explanation_shap(model, X_sample, feature_names=None):
    try:
        import shap
    except ImportError:
        return {'available': False, 'reason': 'SHAP not installed'}

    if feature_names is None:
        feature_names = get_feature_names()

    try:
        if hasattr(model, 'predict_proba'):
            # LightGBM / classifier
            explainer = shap.TreeExplainer(model, X_sample[:100] if len(X_sample) > 100 else X_sample)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif hasattr(model, 'decision_function'):
            # Isolation Forest
            explainer = shap.KernelExplainer(
                lambda x: model.decision_function(x),
                X_sample[:50] if len(X_sample) > 50 else X_sample
            )
            shap_values = explainer.shap_values(X_sample[:1], nsamples=50)
        else:
            return {'available': False, 'reason': 'Model type not supported for SHAP'}

        if len(shap_values.shape) > 1:
            vals = shap_values[0]
        else:
            vals = shap_values

        contributions = []
        for i, name in enumerate(feature_names):
            if i < len(vals):
                contributions.append({
                    'feature': name,
                    'value': float(X_sample[0, i]) if X_sample.ndim > 1 else float(X_sample[i]),
                    'contribution': float(vals[i])
                })
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        return {
            'available': True,
            'method': 'SHAP',
            'top_contributors': contributions[:8],
            'all_contributions': contributions
        }
    except Exception as e:
        return {'available': False, 'reason': str(e)}


def get_explanation_lime(model, X_sample, feature_names=None, predict_fn=None):
    try:
        import lime
        import lime.lime_tabular
    except ImportError:
        return {'available': False, 'reason': 'LIME not installed'}

    if feature_names is None:
        feature_names = get_feature_names()

    try:
        if predict_fn is None:
            if hasattr(model, 'predict_proba'):
                predict_fn = lambda x: model.predict_proba(x)[:, 1]  # anomaly prob
            elif hasattr(model, 'decision_function'):
                predict_fn = lambda x: -model.decision_function(x)  # higher = more anomalous
            else:
                return {'available': False, 'reason': 'No predict function'}

        X = np.array(X_sample)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        explainer = lime.lime_tabular.LimeTabularExplainer(
            X,
            feature_names=feature_names,
            mode='regression',
            verbose=False
        )
        exp = explainer.explain_instance(
            X[0],
            predict_fn,
            num_features=min(8, len(feature_names))
        )
        contributions = []
        for feat, weight in exp.as_list():
            contributions.append({
                'feature': feat,
                'contribution': float(weight)
            })
        return {
            'available': True,
            'method': 'LIME',
            'top_contributors': contributions,
            'all_contributions': contributions
        }
    except Exception as e:
        return {'available': False, 'reason': str(e)}


def explain_anomaly(ml_ensemble, X_sample, use_shap=True, use_lime=True):
    result = {
        'shap': None,
        'lime': None,
        'summary': []
    }

    # Prefer LightGBM for explainability (tree-based, works well with SHAP/LIME)
    model = ml_ensemble.lgb_model if ml_ensemble.lgb_model is not None else ml_ensemble.if_model
    if model is None:
        return {'summary': ['ML models not yet trained. Run training first.']}

    X = np.array(X_sample, dtype=np.float64)
    if len(X.shape) == 1:
        X = X.reshape(1, -1)

    feature_names = get_feature_names()

    if use_shap:
        result['shap'] = get_explanation_shap(model, X, feature_names)

    if use_lime:
        predict_fn = None
        if ml_ensemble.lgb_model is not None:
            predict_fn = lambda x: ml_ensemble.lgb_model.predict_proba(x)[:, 1]
        elif ml_ensemble.if_model is not None:
            predict_fn = lambda x: -ml_ensemble.if_model.decision_function(x)
        result['lime'] = get_explanation_lime(model, X, feature_names, predict_fn)

    # Build human-readable summary
    if result.get('shap', {}).get('available'):
        for c in result['shap'].get('top_contributors', [])[:5]:
            s = c['contribution']
            direction = 'increased' if s > 0 else 'decreased'
            result['summary'].append(
                f"{c['feature']}: {direction} anomaly score (contribution: {s:.2f})"
            )
    if result.get('lime', {}).get('available'):
        for c in result['lime'].get('top_contributors', [])[:3]:
            result['summary'].append(
                f"LIME: {c['feature']} contributed {c['contribution']:.2f}"
            )

    if not result['summary']:
        result['summary'] = ['Anomaly detected by ML ensemble. Run training for detailed explanations.']

    return result
