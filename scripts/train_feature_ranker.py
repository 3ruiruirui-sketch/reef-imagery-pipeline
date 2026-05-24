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
    'contrast', 
    'signal_strength', 
    'cleanliness'
]

def generate_dummy_pairwise_data(num_pairs=50):
    """
    Gera dados sintéticos de pares de imagens para demonstração do pipeline.
    Na prática, o utilizador deverá substituir isto pelo carregamento dos seus rótulos reais.
    """
    np.random.seed(42)
    records = []
    labels = []
    
    for i in range(num_pairs):
        # Imagem A (Simulação de uma boa imagem: baixo kd, alto contraste, alta limpeza)
        a_kd = np.random.uniform(0.04, 0.06)
        a_trans = np.random.uniform(0.7, 0.9)
        a_contrast = np.random.uniform(40, 80)
        a_signal = np.random.uniform(15, 25)
        a_clean = np.random.uniform(8000, 15000)
        
        # Imagem B (Simulação de uma má imagem: alto kd, baixo contraste, glint/ondas)
        b_kd = np.random.uniform(0.06, 0.12)
        b_trans = np.random.uniform(0.4, 0.6)
        b_contrast = np.random.uniform(10, 35)
        b_signal = np.random.uniform(5, 15)
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
    log.info("A gerar dataset pairwise...")
    df_features, df_labels = generate_dummy_pairwise_data(num_pairs=100)
    
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
        "model_version": "1.0",
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "algorithm": "RandomForestRegressor",
        "features": FEATURE_COLS,
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
