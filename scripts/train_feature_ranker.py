#!/usr/bin/env python3
"""
train_feature_ranker.py — Treino do Modelo de Ranking Tabular (Learning-to-Rank)
================================================================================

Treina um modelo (Random Forest) utilizando as features extraídas pelo motor físico.
O objetivo é classificar/pontuar as imagens com base em preferências pairwise
(Qual é a melhor imagem de um par).

Features esperadas: kd_b02, water_trans, contrast, signal_strength, cleanliness

Gera:
- artifacts/training_features.csv
- artifacts/pairwise_labels.csv
- models/feature_ranker_model.pkl
- models/feature_ranker_metadata.json
"""

import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.inspection import permutation_importance

log = logging.getLogger(__name__)

# Diretórios
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, 'artifacts')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Nomes padronizados das features que o motor físico (Fase 2) vai extrair
FEATURE_COLS = [
    'kd_b02', 
    'water_trans', 
    'signal_strength', 
    'cleanliness'
]

# Features temporarily disabled from model training (non-discriminative in production)
DISABLED_FEATURES = {
    'contrast': 'Constant proxy (0.8) — does not discriminate images. Will reactivate when real benthic contrast is available.'
}

def load_real_pairwise_data():
    """
    Tenta carregar pares semi-reais de data/real_pairwise_*.csv.
    Retorna (df_features, df_labels) ou (None, None) se não disponível.
    """
    real_features_path = os.path.join(PROJECT_DIR, 'data', 'real_pairwise_features.csv')
    real_labels_path = os.path.join(PROJECT_DIR, 'data', 'real_pairwise_labels.csv')
    
    if not os.path.exists(real_features_path) or not os.path.exists(real_labels_path):
        log.info("Real pairwise data not found, will use synthetic.")
        return None, None
    
    try:
        df_features = pd.read_csv(real_features_path)
        df_labels = pd.read_csv(real_labels_path)
        
        # Validate required columns
        required_cols = ['image_id'] + FEATURE_COLS
        missing = [c for c in required_cols if c not in df_features.columns]
        if missing:
            log.warning(f"Real data missing columns: {missing}, falling back to synthetic.")
            return None, None
        
        log.info(f"Loaded real pairwise data: {len(df_labels)} pairs from {len(df_features)} records")
        return df_features, df_labels
    except Exception as e:
        log.warning(f"Failed to load real data: {e}, falling back to synthetic.")
        return None, None


def generate_dummy_pairwise_data(num_pairs=50):
    """
    Gera dados sintéticos de pares de imagens para demonstração do pipeline.
    Usado como fallback quando dados reais não estão disponíveis.
    """
    np.random.seed(42)
    records = []
    labels = []
    
    for i in range(num_pairs):
        # CANONICAL UNITS (must match production output of analyse_band / reef_ml_predictor_acolite):
        #   kd_b02:          Kd490 [1/m]       — typical Algarve range [0.04, 0.12]
        #   water_trans:     ratio at depth     — Beer-Lambert two-way at ~16m, range [0.05, 0.50]
        #   contrast:        ratio [0, 1]       — (sand_btm - rock_btm)/sand_btm
        #   signal_strength: SNR linear         — sig_mean/noise_std at 16m window, range [5, 200]
        #   cleanliness:     FFT proxy score    — range [1000, 15000]
        # Imagem A (boa: água clara, alto SNR, alta limpeza)
        a_kd = np.random.uniform(0.040, 0.060)
        a_trans = np.random.uniform(0.20, 0.45)
        a_contrast = np.random.uniform(0.60, 0.90)
        a_signal = np.random.uniform(60, 180)
        a_clean = np.random.uniform(8000, 15000)

        # Imagem B (má: água turva, baixo SNR, ruído)
        b_kd = np.random.uniform(0.070, 0.120)
        b_trans = np.random.uniform(0.05, 0.18)
        b_contrast = np.random.uniform(0.10, 0.45)
        b_signal = np.random.uniform(5, 40)
        b_clean = np.random.uniform(1000, 4000)
        
        # Ocasionalmente inverter para o modelo aprender ambas as direções
        if np.random.rand() > 0.5:
            # A é melhor
            records.append({'image_id': f'img_good_{i}', 'kd_b02': a_kd, 'water_trans': a_trans, 'contrast': a_contrast, 'signal_strength': a_signal, 'cleanliness': a_clean})
            records.append({'image_id': f'img_bad_{i}', 'kd_b02': b_kd, 'water_trans': b_trans, 'contrast': b_contrast, 'signal_strength': b_signal, 'cleanliness': b_clean})
            labels.append({'pair_id': i, 'image_a': f'img_good_{i}', 'image_b': f'img_bad_{i}', 'winner': f'img_good_{i}'})
        else:
            # B é melhor (porque colocamos os dados "bons" no slot B)
            records.append({'image_id': f'img_bad_{i}', 'kd_b02': b_kd, 'water_trans': b_trans, 'contrast': b_contrast, 'signal_strength': b_signal, 'cleanliness': b_clean})
            records.append({'image_id': f'img_good_{i}', 'kd_b02': a_kd, 'water_trans': a_trans, 'contrast': a_contrast, 'signal_strength': a_signal, 'cleanliness': a_clean})
            labels.append({'pair_id': i, 'image_a': f'img_bad_{i}', 'image_b': f'img_good_{i}', 'winner': f'img_good_{i}'})
            
    df_features = pd.DataFrame(records)
    df_labels = pd.DataFrame(labels)
    
    return df_features, df_labels

def prepare_pointwise_dataset(df_features, df_labels):
    """
    Converte as preferências pairwise num dataset pointwise para um Regressor.
    Winner recebe score 1.0, Loser recebe score 0.0.
    Isto permite inferir um score absoluto para qualquer imagem individualmente na Fase 3.
    """
    X = []
    y = []
    
    for _, row in df_labels.iterrows():
        feat_a = df_features[df_features['image_id'] == row['image_a']].iloc[0]
        feat_b = df_features[df_features['image_id'] == row['image_b']].iloc[0]
        
        if row['winner'] == row['image_a']:
            X.append(feat_a[FEATURE_COLS].values)
            y.append(1.0)
            X.append(feat_b[FEATURE_COLS].values)
            y.append(0.0)
        else:
            X.append(feat_b[FEATURE_COLS].values)
            y.append(1.0)
            X.append(feat_a[FEATURE_COLS].values)
            y.append(0.0)
            
    # Criar um DataFrame limpo
    df_X = pd.DataFrame(X, columns=FEATURE_COLS)
    return df_X, np.array(y)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log.info("=== Treino do Ranker de Features Físicas ===")
    
    # 1. Carregar/Gerar Dados
    log.info("Loading dataset...")
    
    # Try real data first
    df_features, df_labels = load_real_pairwise_data()
    data_source = "real"
    
    # Fall back to synthetic if needed
    if df_features is None:
        log.info("Generating synthetic pairwise data...")
        df_features, df_labels = generate_dummy_pairwise_data(num_pairs=100)
        data_source = "synthetic"
    
    log.info(f"Using {data_source} data: {len(df_labels)} pairs")
    
    # Exportar datasets para auditabilidade
    features_csv = os.path.join(ARTIFACTS_DIR, 'training_features.csv')
    labels_csv = os.path.join(ARTIFACTS_DIR, 'pairwise_labels.csv')
    df_features.to_csv(features_csv, index=False)
    df_labels.to_csv(labels_csv, index=False)
    log.info(f"Dataset exportado: {features_csv} e {labels_csv}")
    
    # 2. Preparar Dataset Pointwise e Split
    X, y = prepare_pointwise_dataset(df_features, df_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    log.info(f"Split de dados: {len(X_train)} amostras de treino, {len(X_test)} amostras de validação/teste.")
    
    # 3. Treinar Modelo
    log.info("A treinar RandomForestRegressor (Pointwise Ranker)...")
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Avaliar Modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Transformar em classificação binária (>= 0.5 é "melhor")
    y_pred_bin = (y_pred >= 0.5).astype(int)
    y_test_bin = (y_test >= 0.5).astype(int)
    acc = accuracy_score(y_test_bin, y_pred_bin)
    
    log.info(f"Resultados Validação -> MSE: {mse:.4f} | Precisão (Ranking Binário): {acc:.2%}")
    
    # 5. Importância das Features e Permutation Importance
    importance = model.feature_importances_
    feat_importance = sorted(zip(FEATURE_COLS, importance), key=lambda x: x[1], reverse=True)
    log.info("Importância Gini das Features:")
    for feat, imp in feat_importance:
        log.info(f"  - {feat}: {imp:.4f}")
        
    log.info("A calcular Permutation Importance no conjunto de teste...")
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    perm_importance_dict = {}
    log.info("Permutation Importance:")
    for i, col in enumerate(FEATURE_COLS):
        perm_importance_dict[col] = {
            "mean": float(perm_importance.importances_mean[i]),
            "std": float(perm_importance.importances_std[i])
        }
        log.info(f"  - {col}: {perm_importance.importances_mean[i]:.4f} +/- {perm_importance.importances_std[i]:.4f}")
        
    # Export Permutation Importance to artifacts
    perm_path = os.path.join(ARTIFACTS_DIR, 'permutation_importance.json')
    with open(perm_path, 'w') as f:
        json.dump(perm_importance_dict, f, indent=2)
    log.info(f"Permutation Importance guardada em: {perm_path}")
        
    # 6. Guardar Modelo e Metadados
    model_path = os.path.join(MODELS_DIR, 'feature_ranker_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    import datetime
    metadata = {
        "model_version": "1.2",
        "schema_version": "2.0",
        "canonical": True,
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "algorithm": "RandomForestRegressor",
        "features": FEATURE_COLS,
        "disabled_features": DISABLED_FEATURES,
        "deprecated_models": {
            "visibility_rf_bathy.pkl": "Legacy bathy-only predictor. Not used by production ranking flow."
        },
        "data_source": data_source,
        "metrics": {
            "validation_mse": mse,
            "validation_accuracy": acc
        },
        "feature_importance": {feat: imp for feat, imp in feat_importance},
        "dataset_size_pairs": len(df_labels)
    }
    
    meta_path = os.path.join(MODELS_DIR, 'feature_ranker_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    log.info(f"Modelo guardado em: {model_path}")
    log.info(f"Metadados guardados em: {meta_path}")
    log.info("=== Concluído ===")

if __name__ == "__main__":
    main()
