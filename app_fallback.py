# -*- coding: utf-8 -*-
"""
CrediFast — Fallback App (sem .joblib):
Se os arquivos preprocessor.joblib / best_model_XGBoost.joblib não estiverem presentes,
este app treina rapidamente um modelo com base no CSV enviado (necessita loan_status).

Uso:
- Faça upload do CSV com a coluna loan_status.
- O app treina XGBoost + preprocessor e segue com as abas.
- Se os .joblib estiverem presentes, ele apenas carrega.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score
from sklearn.neighbors import NearestNeighbors
import shap
import xgboost as xgb

st.set_page_config(page_title="CrediFast — Fallback App", layout="wide")
plt.style.use('seaborn-v0_8')
sns.set_context('talk')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

EXPECTED_COLS = ['person_age','person_income','person_home_ownership','person_emp_length','loan_intent','loan_grade','loan_amnt','loan_int_rate','loan_status','loan_percent_income','cb_person_default_on_file','cb_person_cred_hist_length']
NUM_COLS = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
CAT_COLS = ['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']

st.title("CrediFast — Dashboard (Fallback)")
st.caption("Funciona mesmo sem .joblib: treina com seu CSV e segue.")

# Try to load artifacts
preproc = None
model = None
if os.path.exists('preprocessor.joblib') and os.path.exists('best_model_XGBoost.joblib'):
    try:
        preproc = load('preprocessor.joblib')
        model = load('best_model_XGBoost.joblib')
        st.success("Artefatos carregados: usando modelo salvo.")
    except Exception as e:
        st.warning(f"Falha ao carregar artefatos: {e}. Será feito treino rápido com CSV.")

uploaded = st.sidebar.file_uploader("Upload do CSV (deve conter loan_status)", type=['csv'])
if uploaded is None and (preproc is None or model is None):
    st.info("Faça upload do CSV para que o app possa treinar e funcionar sem artefatos.")

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("**Amostra**:")
    st.dataframe(df.head(10))

# Train quickly if artifacts missing
if df is not None and (preproc is None or model is None):
    if 'loan_status' not in df.columns:
        st.error("CSV precisa conter a coluna 'loan_status' para treino.")
        st.stop()
    X = df.drop(columns=['loan_status'])
    y = df['loan_status'].astype(int)
    # Build preprocessor
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preproc = ColumnTransformer(transformers=[('num', numeric_transformer, NUM_COLS), ('cat', categorical_transformer, CAT_COLS)], remainder='drop')
    # Split + fit
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    X_train_prep = preproc.fit_transform(X_train)
    X_test_prep  = preproc.transform(X_test)
    # Train XGBoost
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE, n_estimators=200)
    model.fit(X_train_prep, y_train)
    # Quick validation
    val_score = model.predict_proba(X_test_prep)[:,1]
    st.info(f"Modelo treinado rapidamente. AUC de validação: {roc_auc_score(y_test, val_score):.3f}")
    # Save artifacts (optional)
    try:
        dump(preproc, 'preprocessor.joblib')
        dump(model, 'best_model_XGBoost.joblib')
        st.caption("Artefatos salvos na pasta (preprocessor.joblib, best_model_XGBoost.joblib).")
    except Exception:
        pass

# If still missing, stop
if preproc is None or model is None:
    st.stop()

# Helper funcs
def get_feature_names(preprocessor):
    num_features = NUM_COLS
    try:
        onehot = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = list(onehot.get_feature_names_out(CAT_COLS))
    except Exception:
        cat_features = []
    return num_features + cat_features

feature_names = get_feature_names(preproc)

tab1, tab2, tab3, tab4 = st.tabs(["Classificação","Explicabilidade","Clusters","Outliers"])

with tab1:
    st.subheader("Classificação de Risco")
    if df is None:
        st.info("Faça upload do CSV para classificar.")
    else:
        X_all = df.drop(columns=[c for c in ['loan_status'] if c in df.columns])
        X_prep = preproc.transform(X_all)
        scores = model.predict_proba(X_prep)[:,1]
        thr = st.slider("Limiar",0.0,1.0,0.5,0.01)
        preds = (scores>=thr).astype(int)
        fig,ax = plt.subplots(figsize=(6,3))
        sns.histplot(scores,bins=30,kde=True,ax=ax)
        ax.axvline(thr,color='red',linestyle='--')
        ax.set_title("Probabilidades (Bad=1)")
        st.pyplot(fig)
        out_df = df.copy(); out_df['score_bad']=scores; out_df['pred_bad']=preds
        st.download_button("Baixar previsões", out_df.to_csv(index=False).encode('utf-8'), "predicoes.csv")
        if 'loan_status' in df.columns:
            y_true = df['loan_status'].astype(int).values
            from sklearn.metrics import roc_curve
            fpr,tpr,_ = roc_curve(y_true,scores)
            auc = roc_auc_score(y_true,scores)
            cm = confusion_matrix(y_true,preds)
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("AUC", f"{auc:.3f}"); c2.metric("Accuracy", f"{accuracy_score(y_true,preds):.3f}")
            c3.metric("Precision", f"{precision_score(y_true,preds,zero_division=0):.3f}")
            c4.metric("Recall", f"{recall_score(y_true,preds,zero_division=0):.3f}")
            c5.metric("F1", f"{f1_score(y_true,preds,zero_division=0):.3f}")
            fig2,ax2=plt.subplots(figsize=(6,5))
            ax2.plot(fpr,tpr,label=f"AUC={auc:.3f}"); ax2.plot([0,1],[0,1],'k--'); ax2.legend(); ax2.set_title("ROC")
            st.pyplot(fig2)
            fig3,ax3=plt.subplots(); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax3); ax3.set_title('Matriz de Confusão'); st.pyplot(fig3)

with tab2:
    st.subheader("Explicabilidade (SHAP)")
    if df is None:
        st.info("Faça upload do CSV e gere previsões na aba Classificação.")
    else:
        X_all = df.drop(columns=[c for c in ['loan_status'] if c in df.columns])
        X_prep = preproc.transform(X_all)
        try:
            explainer = shap.TreeExplainer(model)
            sample_size = min(1000, X_prep.shape[0])
            idx = np.random.choice(np.arange(X_prep.shape[0]), size=sample_size, replace=False)
            X_shap = X_prep[idx]
            shap_values = explainer.shap_values(X_shap)
            shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
            fig = plt.figure(figsize=(10,6))
            shap.summary_plot(shap_vals, X_shap, feature_names=feature_names, show=False)
            st.pyplot(fig)
            i = st.number_input("Índice para análise local", min_value=0, max_value=int(X_prep.shape[0]-1), value=0)
            local_vals = explainer.shap_values(X_prep[[i]])
            local = local_vals[1] if isinstance(local_vals, list) else local_vals
            base_val = explainer.expected_value[1] if isinstance(explainer.expected_value,(list,np.ndarray)) else explainer.expected_value
            try:
                exp = shap.Explanation(values=local[0], base_values=base_val, data=X_prep[[i]][0], feature_names=feature_names)
                fig2 = shap.plots.waterfall(exp, show=False); st.pyplot(fig2)
            except Exception:
                st.warning("Waterfall indisponível. Exibindo barras das contribuições.")
                import pandas as pd
                contrib = pd.Series(local[0], index=feature_names).sort_values(key=np.abs, ascending=False)[:20]
                fig_alt, ax_alt = plt.subplots(figsize=(8,6))
                sns.barplot(y=contrib.index, x=contrib.values, ax=ax_alt)
                ax_alt.set_title("Top 20 contribuições SHAP (local)")
                st.pyplot(fig_alt)
        except Exception as e:
            st.error(f"Falha no SHAP: {e}")

with tab3:
    st.subheader("Clusters (KMeans + PCA)")
    if df is None:
        st.info("Faça upload do CSV.")
    else:
        X_all = df.drop(columns=[c for c in ['loan_status'] if c in df.columns])
        X_all_prep = preproc.transform(X_all)
        scaler = RobustScaler(); X_scaled = scaler.fit_transform(X_all_prep)
        pca = PCA(n_components=2, random_state=RANDOM_STATE); X_pca = pca.fit_transform(X_scaled)
        k = st.slider("Número de clusters (k)", 2, 8, 4)
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        clusters = kmeans.fit_predict(X_pca)
        sil = silhouette_score(X_pca, clusters)
        st.caption(f"Silhouette: {sil:.3f}")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette='tab10', s=20, ax=ax)
        ax.set_title(f"KMeans (k={k}) em PCA(2)"); st.pyplot(fig)
        if 'loan_status' in df.columns:
            import pandas as pd
            tmp = pd.DataFrame({'cluster': clusters, 'loan_status': df['loan_status'].astype(int)})
            risk = tmp.groupby('cluster')['loan_status'].agg(['count','sum']); risk['bad_rate']=risk['sum']/risk['count']
            st.dataframe(risk.sort_values('bad_rate', ascending=False))

with tab4:
    st.subheader("Outliers (DBSCAN)")
    if df is None:
        st.info("Faça upload do CSV.")
    else:
        X_all = df.drop(columns=[c for c in ['loan_status'] if c in df.columns])
        X_all_prep = preproc.transform(X_all)
        scaler = RobustScaler(); X_scaled = scaler.fit_transform(X_all_prep)
        neigh = NearestNeighbors(n_neighbors=5); distances,_ = neigh.fit(X_scaled).kneighbors(X_scaled)
        eps_auto = float(np.percentile(np.sort(distances[:,-1]),95))
        c1,c2 = st.columns(2)
        with c1: eps = st.number_input("eps", value=float(f"{eps_auto:.4f}"))
        with c2: min_samples = st.number_input("min_samples", min_value=2, value=5, step=1)
        db = DBSCAN(eps=eps, min_samples=int(min_samples))
        labels = db.fit_predict(X_scaled)
        out_mask = labels==-1
        st.write(f"Outliers: {out_mask.sum()} de {len(labels)}")
        pca = PCA(n_components=2, random_state=RANDOM_STATE); X_pca = pca.fit_transform(X_scaled)
        fig, ax = plt.subplots(figsize=(8,6))
        sns.scatterplot(x=X_pca[~out_mask,0], y=X_pca[~out_mask,1], s=15, color='tab:blue', label='Inliers', ax=ax)
        sns.scatterplot(x=X_pca[out_mask,0], y=X_pca[out_mask,1], s=20, color='tab:red', label='Outliers', ax=ax)
        ax.set_title("DBSCAN: Inliers vs Outliers (PCA 2D)"); st.pyplot(fig)
