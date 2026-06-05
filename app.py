import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use("Agg")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="PredAorta", layout="centered")

@st.cache_resource
def carregar_modelos():
    modelos = {
        "obito"      : joblib.load(os.path.join(BASE_DIR, "modelo_obito_30d.pkl")),
        "complicacao": joblib.load(os.path.join(BASE_DIR, "modelo_complicacao_grave.pkl")),
    }
    imputers = {
        "obito"      : joblib.load(os.path.join(BASE_DIR, "imputer_obito_30d.pkl")),
        "complicacao": joblib.load(os.path.join(BASE_DIR, "imputer_complicacao_grave.pkl")),
    }
    with open(os.path.join(BASE_DIR, "features.json"))      as f: features      = json.load(f)
    with open(os.path.join(BASE_DIR, "score_clinico.json")) as f: score_clinico = json.load(f)
    with open(os.path.join(BASE_DIR, "metricas.json"))      as f: metricas      = json.load(f)
    return modelos, imputers, features, score_clinico, metricas

modelos, imputers, features, score_clinico, metricas = carregar_modelos()

st.markdown("#PredAorta")
st.markdown("### Preditor de Risco em Aneurisma de Aorta")
st.markdown("Modelo baseado em machine learning desenvolvido com dados do **MIMIC-IV**. Preencha os dados do paciente e clique em **Calcular Risco**.")
st.divider()
st.warning("**Uso exclusivamente para pesquisa.** Este modelo não substitui o julgamento clínico. Desenvolvido com dados do MIMIC-IV (2008–2019, Boston, EUA).")

st.markdown("## Dados do Paciente")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Demográficos**")
    idade  = st.number_input("Idade (anos)", min_value=18, max_value=90, value=70, step=1)
    sexo   = st.radio("Sexo", ["Masculino", "Feminino"], horizontal=True)
    gender = 1 if sexo == "Masculino" else 0
with col2:
    st.markdown("**Intervenção**")
    evar          = st.checkbox("EVAR realizado")
    reparo_aberto = st.checkbox("Reparo aberto realizado")

st.divider()
st.markdown("## Exames Laboratoriais (primeiras 24h)")
st.caption("Deixe em branco se o exame não foi coletado.")

col3, col4, col5 = st.columns(3)
with col3:
    albumina   = st.number_input("Albumina (g/dL)",    min_value=0.0, max_value=6.0,    value=None, placeholder="ex: 3.2",  format="%.1f")
    ureia      = st.number_input("Ureia (mg/dL)",      min_value=0.0, max_value=300.0,  value=None, placeholder="ex: 25.0", format="%.1f")
    lactato    = st.number_input("Lactato (mmol/L)",   min_value=0.0, max_value=20.0,   value=None, placeholder="ex: 1.8",  format="%.1f")
    creatinina = st.number_input("Creatinina (mg/dL)", min_value=0.0, max_value=20.0,   value=None, placeholder="ex: 1.1",  format="%.1f")
with col4:
    hemoglobina = st.number_input("Hemoglobina (g/dL)", min_value=0.0, max_value=20.0,   value=None, placeholder="ex: 12.5", format="%.1f")
    hematocrito = st.number_input("Hematócrito (%)",    min_value=0.0, max_value=60.0,   value=None, placeholder="ex: 38.0", format="%.1f")
    plaquetas   = st.number_input("Plaquetas (10³/µL)", min_value=0.0, max_value=1000.0, value=None, placeholder="ex: 180",  format="%.0f")
    inr         = st.number_input("INR",                min_value=0.0, max_value=10.0,   value=None, placeholder="ex: 1.2",  format="%.1f")
with col5:
    leucocitos = st.number_input("Leucócitos (10³/µL)", min_value=0.0, max_value=100.0,  value=None, placeholder="ex: 9.5",  format="%.1f")
    sodio      = st.number_input("Sódio (mEq/L)",       min_value=100.0, max_value=170.0, value=None, placeholder="ex: 138", format="%.1f")

st.divider()
st.markdown("## Comorbidades")
col6, col7 = st.columns(2)
with col6:
    htn  = st.checkbox("Hipertensão (HAS)")
    dm   = st.checkbox("Diabetes mellitus")
    dpoc = st.checkbox("DPOC")
    irc  = st.checkbox("Insuf. renal crônica")
with col7:
    dac = st.checkbox("Doença arterial coronariana")
    ic  = st.checkbox("Insuficiência cardíaca")

st.divider()

if st.button("Calcular Risco", type="primary", use_container_width=True):

    entrada = {f: np.nan for f in features}
    entrada.update({
        "idade": idade, "gender": gender,
        "evar": int(evar), "reparo_aberto": int(reparo_aberto),
        "htn": int(htn), "dm": int(dm), "dpoc": int(dpoc),
        "irc": int(irc), "dac": int(dac), "ic": int(ic),
    })
    for nome_var, valor in [
        ("albumina", albumina), ("ureia", ureia), ("lactato", lactato),
        ("creatinina", creatinina), ("hemoglobina", hemoglobina),
        ("hematocrito", hematocrito), ("plaquetas", plaquetas),
        ("inr", inr), ("leucocitos", leucocitos), ("sodio", sodio),
    ]:
        if nome_var in entrada and valor is not None:
            entrada[nome_var] = valor

    X_input = pd.DataFrame([entrada])[features]

    score = 0
    criterios_atingidos = []
    for c in score_clinico:
        val = entrada.get(c["exame"])
        if val is not None and not np.isnan(float(val)):
            atingido = (val < c["corte"]) if c["sinal"] == "<" else (val > c["corte"])
            if atingido:
                score += 1
                criterios_atingidos.append(c["label"])

    resultados_pred = {}
    for chave, label in [("obito","Óbito 30 dias"),("complicacao","Complicação grave")]:
        X_imp = imputers[chave].transform(X_input)
        prob  = modelos[chave].predict_proba(X_imp)[0][1]
        resultados_pred[label] = prob

    st.markdown("---")
    st.markdown("## Resultado")

    def classificar(prob):
        if prob < 0.10: return "🟢 BAIXO",        "success"
        if prob < 0.25: return "🟡 INTERMEDIÁRIO", "warning"
        return               "🔴 ALTO",           "error"

    col_r1, col_r2 = st.columns(2)
    for col_r, (label, prob) in zip([col_r1, col_r2], resultados_pred.items()):
        emoji_risco, tipo = classificar(prob)
        with col_r:
            st.metric(label=label, value=f"{prob*100:.1f}%")
            if tipo == "success": st.success(emoji_risco)
            elif tipo == "warning": st.warning(emoji_risco)
            else: st.error(emoji_risco)

    st.divider()
    st.markdown("### 📋 Score Clínico Simplificado")
    if score == 0:   nivel = "🟢 Baixo risco (~2,6% mortalidade histórica)"
    elif score <= 2: nivel = "🟡 Risco intermediário (~7,9%)"
    else:            nivel = "🔴 Alto risco (~39,2%)"
    st.markdown(f"**{score}/4 critérios → {nivel}**")
    if criterios_atingidos:
        for c in criterios_atingidos:
            st.markdown(f"  - ⚠️ {c}")
    else:
        st.markdown("Nenhum critério de alto risco atingido.")

    st.divider()
    st.markdown("### Desempenho do Modelo")
    cols_met = st.columns(len(metricas))
    for col_m, (nome_m, met) in zip(cols_met, metricas.items()):
        with col_m:
            st.markdown(f"**{nome_m}**")
            st.markdown(f"AUROC: **{met['auroc']}** (IC 95%: {met['ic_inf']}–{met['ic_sup']})")
            st.markdown(f"Brier: **{met['brier']}**")

    st.divider()
    st.caption("Modelo desenvolvido com dados do MIMIC-IV v2.2 · Random Forest · Validação temporal · Bootstrap IC 95% · Apenas para pesquisa.")
