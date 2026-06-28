"""Dashboard Streamlit — cliente visual da API de previsão de churn.

Este app NÃO reimplementa nenhuma lógica de ML: é um cliente HTTP puro da
API construída na Etapa 3 (endpoints /infer e /predict/batch). Toda a
inferência, validação e regra de negócio continuam centralizadas na API —
este app só oferece uma interface amigável sobre ela, para um usuário de
negócio (time de Retenção/CRM) sem conhecimento técnico de APIs.

Uso local:
    streamlit run app/streamlit_app.py

A URL da API é configurável via variável de ambiente CHURN_API_URL (ou
st.secrets, em deploy), com fallback para localhost — ver README.md.
"""

from __future__ import annotations

import io
import os

import pandas as pd
import requests
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 30
MAX_BATCH_SIZE = 500

# Domínios de cada campo categórico, espelhando exatamente
# churn_prediction.schemas.ChurnPredictionRequest (Etapa 3) — qualquer
# mudança no schema da API deve ser refletida aqui também.
FIELD_OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

REQUIRED_CSV_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

RISK_COLORS = {"low": "🟢", "medium": "🟡", "high": "🔴"}


def get_api_url() -> str:
    """Resolve a URL base da API: secrets (deploy) > env var > fallback local."""
    try:
        if "CHURN_API_URL" in st.secrets:
            return str(st.secrets["CHURN_API_URL"]).rstrip("/")
    except Exception:
        pass  # st.secrets pode não existir em execução local sem secrets.toml
    return os.environ.get("CHURN_API_URL", DEFAULT_API_URL).rstrip("/")


def call_api(method: str, path: str, **kwargs):
    """Chama a API e retorna (sucesso, dados_ou_mensagem_de_erro)."""
    url = f"{get_api_url()}{path}"
    try:
        response = requests.request(method, url, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
    except requests.exceptions.ConnectionError:
        return False, f"Não foi possível conectar à API em {url}. Ela está no ar?"
    except requests.exceptions.Timeout:
        return False, f"A API em {url} não respondeu a tempo ({REQUEST_TIMEOUT_SECONDS}s)."

    if response.status_code == 200:
        return True, response.json()

    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return False, f"Erro {response.status_code}: {detail}"


def render_risk_badge(risk_level: str) -> str:
    return f"{RISK_COLORS.get(risk_level, '⚪')} {risk_level.upper()}"


def render_sidebar() -> dict | None:
    """Renderiza a barra lateral com o status da API e retorna model_info, se disponível."""
    st.sidebar.title("Status da API")
    st.sidebar.caption(f"Endpoint: `{get_api_url()}`")

    ready_ok, ready_data = call_api("GET", "/ready")
    if ready_ok and ready_data.get("model_loaded"):
        st.sidebar.success("✅ Modelo carregado e pronto")
    elif ready_ok:
        st.sidebar.warning("⚠️ API no ar, mas o modelo não está carregado")
    else:
        st.sidebar.error(f"❌ {ready_data}")
        return None

    meta_ok, meta_data = call_api("GET", "/metadata")
    if not meta_ok:
        return None

    info = meta_data.get("model_info", {})
    st.sidebar.divider()
    st.sidebar.subheader("Modelo em produção")
    st.sidebar.metric("Versão", info.get("model_version", "—"))
    st.sidebar.metric("AUC-ROC (teste)", f"{info.get('test_roc_auc', 0):.3f}")
    st.sidebar.metric("Recall (teste)", f"{info.get('test_recall', 0):.3f}")
    net_cost = info.get("business_net_cost")
    if net_cost is not None:
        label = "Ganho líquido estimado" if net_cost < 0 else "Custo líquido estimado"
        st.sidebar.metric(label, f"R$ {abs(net_cost):,.2f}")

    st.sidebar.divider()
    st.sidebar.caption(
        "Este painel é um cliente da API de inferência (Tech Challenge 1). "
        "Nenhuma lógica de ML roda aqui — toda predição é feita pela API."
    )
    return info


def render_single_customer_tab() -> None:
    st.header("Avaliar um cliente")
    st.caption("Preencha os dados do cliente para estimar o risco de churn.")

    with st.form("single_customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Perfil")
            gender = st.selectbox("Gênero", FIELD_OPTIONS["gender"])
            senior_citizen = st.selectbox(
                "Idoso (Senior Citizen)", FIELD_OPTIONS["SeniorCitizen"], format_func=lambda x: "Sim" if x else "Não"
            )
            partner = st.selectbox("Possui parceiro(a)", FIELD_OPTIONS["Partner"])
            dependents = st.selectbox("Possui dependentes", FIELD_OPTIONS["Dependents"])
            tenure = st.number_input(
                "Tempo de relacionamento (meses)", min_value=0, max_value=120, value=12
            )

        with col2:
            st.subheader("Serviços")
            phone_service = st.selectbox("Serviço telefônico", FIELD_OPTIONS["PhoneService"])
            multiple_lines = st.selectbox("Múltiplas linhas", FIELD_OPTIONS["MultipleLines"])
            internet_service = st.selectbox("Serviço de internet", FIELD_OPTIONS["InternetService"])
            online_security = st.selectbox("Segurança online", FIELD_OPTIONS["OnlineSecurity"])
            online_backup = st.selectbox("Backup online", FIELD_OPTIONS["OnlineBackup"])
            device_protection = st.selectbox("Proteção de dispositivo", FIELD_OPTIONS["DeviceProtection"])
            tech_support = st.selectbox("Suporte técnico", FIELD_OPTIONS["TechSupport"])
            streaming_tv = st.selectbox("Streaming de TV", FIELD_OPTIONS["StreamingTV"])
            streaming_movies = st.selectbox("Streaming de filmes", FIELD_OPTIONS["StreamingMovies"])

        with col3:
            st.subheader("Contrato e cobrança")
            contract = st.selectbox("Tipo de contrato", FIELD_OPTIONS["Contract"])
            paperless_billing = st.selectbox("Fatura sem papel", FIELD_OPTIONS["PaperlessBilling"])
            payment_method = st.selectbox("Forma de pagamento", FIELD_OPTIONS["PaymentMethod"])
            monthly_charges = st.number_input(
                "Cobrança mensal (R$)", min_value=0.0, max_value=1000.0, value=70.0, step=0.5
            )
            total_charges = st.number_input(
                "Cobrança total acumulada (R$)", min_value=0.0, max_value=100_000.0, value=840.0, step=10.0
            )

        submitted = st.form_submit_button("Avaliar risco de churn", type="primary", use_container_width=True)

    if not submitted:
        return

    payload = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    with st.spinner("Consultando a API..."):
        ok, result = call_api("POST", "/infer", json=payload)

    if not ok:
        st.error(result)
        return

    st.divider()
    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("Probabilidade de churn", f"{result['churn_probability']:.1%}")
    result_col2.metric("Nível de risco", render_risk_badge(result["risk_level"]))
    result_col3.metric(
        "Predição", "Vai cancelar" if result["churn_prediction"] else "Não vai cancelar"
    )
    st.caption(f"Modelo: versão {result['model_version']}")


def render_batch_tab() -> None:
    st.header("Avaliar uma carteira de clientes (CSV)")
    st.caption(
        f"Envie um CSV com até {MAX_BATCH_SIZE} clientes, com as colunas no mesmo "
        "formato do dataset de treino, para pontuar todos de uma vez."
    )

    with st.expander("Ver colunas esperadas no CSV"):
        st.code(", ".join(REQUIRED_CSV_COLUMNS))

    uploaded_file = st.file_uploader("Arquivo CSV", type=["csv"])

    if uploaded_file is None:
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:  # noqa: BLE001 - erro de parsing de CSV do usuário, mostrado diretamente
        st.error(f"Não foi possível ler o CSV: {exc}")
        return

    missing_columns = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing_columns:
        st.error(f"Colunas faltando no CSV: {', '.join(missing_columns)}")
        return

    if len(df) > MAX_BATCH_SIZE:
        st.error(f"O CSV tem {len(df)} linhas; o máximo suportado por chamada é {MAX_BATCH_SIZE}.")
        return

    st.write(f"**{len(df)} clientes** encontrados no arquivo.")
    st.dataframe(df.head(5), use_container_width=True)

    if not st.button("Avaliar carteira", type="primary"):
        return

    customers = df[REQUIRED_CSV_COLUMNS].to_dict(orient="records")

    with st.spinner(f"Avaliando {len(customers)} clientes..."):
        ok, result = call_api("POST", "/predict/batch", json={"customers": customers})

    if not ok:
        st.error(result)
        return

    predictions_df = pd.DataFrame(result["predictions"])
    result_df = pd.concat([df.reset_index(drop=True), predictions_df], axis=1)

    st.divider()
    st.subheader("Resultado")

    risk_counts = result_df["risk_level"].value_counts()
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    risk_col1.metric("🔴 Alto risco", int(risk_counts.get("high", 0)))
    risk_col2.metric("🟡 Médio risco", int(risk_counts.get("medium", 0)))
    risk_col3.metric("🟢 Baixo risco", int(risk_counts.get("low", 0)))

    sorted_df = result_df.sort_values("churn_probability", ascending=False)
    st.dataframe(sorted_df, use_container_width=True)

    csv_buffer = io.StringIO()
    sorted_df.to_csv(csv_buffer, index=False)
    st.download_button(
        "Baixar resultado (CSV)",
        data=csv_buffer.getvalue(),
        file_name="churn_predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Previsão de Churn", page_icon="📉", layout="wide")
    st.title("📉 Painel de Previsão de Churn")
    st.caption(
        "Tech Challenge 1 (FIAP MLE10) — cliente visual da API de inferência "
        "de churn (MLP em PyTorch). Toda a predição é feita pela API; este "
        "painel apenas envia os dados e exibe o resultado."
    )

    render_sidebar()

    tab_single, tab_batch = st.tabs(["👤 Cliente único", "📋 Carteira (CSV)"])
    with tab_single:
        render_single_customer_tab()
    with tab_batch:
        render_batch_tab()


if __name__ == "__main__":
    main()
