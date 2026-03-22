import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px


# KONFIGURASI

st.set_page_config(
    page_title="Jaya Jaya Institut - Student Risk Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CUSTOM CSS

st.markdown(
    """
<style>
 .main-header {
 font-size: 2.2rem;
 font-weight: 700;
 color: #406093;
 text-align: center;
 padding: 1rem 0;
 border-bottom: 3px solid #4C8CE4;
 margin-bottom: 1.5rem;
 }
 .sub-header {
 font-size: 1.1rem;
 color: #406093;
 text-align: center;
 margin-bottom: 2rem;
 }
 .metric-card {
 background: #4C8CE4;
 padding: 1.2rem;
 border-radius: 12px;
 color: white;
 text-align: center;
 margin-bottom: 1rem;
 }
 .risk-high {
 background: #406093;
 padding: 1.5rem;
 border-radius: 12px;
 color: white;
 text-align: center;
 font-size: 1.3rem;
 font-weight: bold;
 }
 .risk-medium {
 background: #FFF799; color: #406093;
 padding: 1.5rem;
 border-radius: 12px;
 color: white;
 text-align: center;
 font-size: 1.3rem;
 font-weight: bold;
 }
 .risk-low {
 background: #91D06C;
 padding: 1.5rem;
 border-radius: 12px;
 color: white;
 text-align: center;
 font-size: 1.3rem;
 font-weight: bold;
 }
 .stTabs [data-baseweb="tab-list"] {
 gap: 8px;
 }
 .stTabs [data-baseweb="tab"] {
 padding: 10px 24px;
 font-weight: 600;
 }
</style>
""",
    unsafe_allow_html=True,
)


# LOAD MODEL


@st.cache_resource
def load_model():
    try:
        model_data = joblib.load("model/model.joblib")
        return model_data
    except FileNotFoundError:
        st.error(
            " Model tidak ditemukan! Jalankan notebook.ipynb terlebih dahulu untuk melatih model."
        )
        st.stop()


model_data = load_model()
model = model_data["model"]
feature_names = model_data["feature_names"]
le = model_data["label_encoder"]
best_model_name = model_data["best_model_name"]


# FEATURE ENGINEERING FUNCTION


def engineer_features(df):
    """Apply same feature engineering as notebook."""
    df = df.copy()
    df["GPA"] = (
        df["Curricular_units_1st_sem_grade"] + df["Curricular_units_2nd_sem_grade"]
    ) / 2
    df["Total_approved"] = (
        df["Curricular_units_1st_sem_approved"]
        + df["Curricular_units_2nd_sem_approved"]
    )
    df["Total_enrolled"] = (
        df["Curricular_units_1st_sem_enrolled"]
        + df["Curricular_units_2nd_sem_enrolled"]
    )
    df["Approval_rate"] = df["Total_approved"] / (df["Total_enrolled"] + 1e-9)
    df["Approval_rate"] = df["Approval_rate"].clip(0, 1)
    df["Total_evaluations"] = (
        df["Curricular_units_1st_sem_evaluations"]
        + df["Curricular_units_2nd_sem_evaluations"]
    )
    df["Absences"] = (
        df["Curricular_units_1st_sem_without_evaluations"]
        + df["Curricular_units_2nd_sem_without_evaluations"]
    )
    df["Failures"] = (df["Total_enrolled"] - df["Total_approved"]).clip(lower=0)
    df["Risk_score"] = df["Failures"] * (df["Absences"] + 1) / (df["GPA"] + 1)
    df["Grade_diff"] = (
        df["Curricular_units_2nd_sem_grade"] - df["Curricular_units_1st_sem_grade"]
    )
    return df


def create_gauge(prob, label, color):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": label, "font": {"size": 16}},
            number={"suffix": "%", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 2,
                "steps": [
                    {"range": [0, 33], "color": "#f8f9fa"},
                    {"range": [33, 66], "color": "#f1f3f5"},
                    {"range": [66, 100], "color": "#e9ecef"},
                ],
            },
        )
    )
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
    return fig


# HEADER

st.markdown(
    '<div class="main-header"> \U0001f3eb Jaya Jaya Institut</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Student Risk Prediction System | Early Warning untuk Identifikasi Risiko Dropout</div>',
    unsafe_allow_html=True,
)


# SIDEBAR - INPUT PARAMETERS

st.sidebar.header(" \U0001f4dd Parameter Mahasiswa")
st.sidebar.markdown("---")

st.sidebar.subheader(" \U0001f464 Demografi")
gender = st.sidebar.selectbox(
    "Gender", [1, 0], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan"
)
age = st.sidebar.number_input(
    "Usia Saat Pendaftaran", min_value=17, max_value=70, value=20
)
marital = st.sidebar.selectbox(
    "Status Pernikahan",
    [1, 2, 3, 4, 5, 6],
    format_func=lambda x: {
        1: "Single",
        2: "Married",
        3: "Widower",
        4: "Divorced",
        5: "Facto Union",
        6: "Separated",
    }[x],
)

st.sidebar.subheader(" \U0001f393 Akademik")
prev_qual_grade = st.sidebar.slider(
    "Nilai Kualifikasi Sebelumnya", 0.0, 200.0, 130.0, 1.0
)
admission_grade = st.sidebar.slider("Nilai Penerimaan", 0.0, 200.0, 125.0, 1.0)
course = st.sidebar.selectbox(
    "Program Studi",
    [
        33,
        171,
        8014,
        9003,
        9070,
        9085,
        9119,
        9130,
        9147,
        9238,
        9254,
        9500,
        9556,
        9670,
        9773,
        9853,
        9991,
    ],
    format_func=lambda x: {
        33: "Biofuel Tech",
        171: "Animation",
        8014: "Social Service (Eve)",
        9003: "Agronomy",
        9070: "Comm Design",
        9085: "Vet Nursing",
        9119: "Informatics",
        9130: "Equinculture",
        9147: "Management",
        9238: "Social Service",
        9254: "Tourism",
        9500: "Nursing",
        9556: "Oral Hygiene",
        9670: "Advertising",
        9773: "Journalism",
        9853: "Basic Education",
        9991: "Management (Eve)",
    }.get(x, str(x)),
)

st.sidebar.subheader(" \U0001f4da Semester 1")
enrolled_1 = st.sidebar.number_input("Unit Terdaftar Sem 1", 0, 30, 6)
approved_1 = st.sidebar.number_input("Unit Lulus Sem 1", 0, 30, 5)
grade_1 = st.sidebar.slider("Nilai Rata-rata Sem 1", 0.0, 20.0, 12.0, 0.1)
eval_1 = st.sidebar.number_input("Jumlah Evaluasi Sem 1", 0, 50, 6)
without_eval_1 = st.sidebar.number_input("Tanpa Evaluasi Sem 1", 0, 20, 0)
credited_1 = st.sidebar.number_input("Unit Dikreditkan Sem 1", 0, 20, 0)

st.sidebar.subheader(" \U0001f4d6 Semester 2")
enrolled_2 = st.sidebar.number_input("Unit Terdaftar Sem 2", 0, 30, 6)
approved_2 = st.sidebar.number_input("Unit Lulus Sem 2", 0, 30, 5)
grade_2 = st.sidebar.slider("Nilai Rata-rata Sem 2", 0.0, 20.0, 12.0, 0.1)
eval_2 = st.sidebar.number_input("Jumlah Evaluasi Sem 2", 0, 50, 6)
without_eval_2 = st.sidebar.number_input("Tanpa Evaluasi Sem 2", 0, 20, 0)
credited_2 = st.sidebar.number_input("Unit Dikreditkan Sem 2", 0, 20, 0)

st.sidebar.subheader(" \U0001f4b0 Finansial & Lainnya")
tuition = st.sidebar.selectbox(
    "Tuition Up to Date", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak"
)
debtor = st.sidebar.selectbox(
    "Debtor", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak"
)
scholarship = st.sidebar.selectbox(
    "Penerima Beasiswa", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak"
)
displaced = st.sidebar.selectbox(
    "Displaced", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak"
)


# TABS

tab1, tab2, tab3 = st.tabs(
    [
        "\U0001f464 Prediksi Individu",
        "\U0001f4c2 Batch Prediction",
        "\U0001f4ca Dashboard Insight",
    ]
)


# TAB 1: PREDIKSI INDIVIDU

with tab1:
    st.subheader(" \U0001f3af Prediksi Risiko Dropout - Individu")
    st.markdown(
        "Klik tombol di bawah untuk memprediksi risiko dropout berdasarkan parameter di sidebar."
    )

    if st.button(
        " \U0001f50d Prediksi Sekarang", type="primary", use_container_width=True
    ):
        with st.spinner("Memproses prediksi..."):
            try:
                input_data = {
                    "Marital_status": marital,
                    "Application_mode": 1,
                    "Application_order": 1,
                    "Course": course,
                    "Daytime_evening_attendance": 1,
                    "Previous_qualification": 1,
                    "Previous_qualification_grade": prev_qual_grade,
                    "Nacionality": 1,
                    "Mothers_qualification": 1,
                    "Fathers_qualification": 1,
                    "Mothers_occupation": 5,
                    "Fathers_occupation": 5,
                    "Admission_grade": admission_grade,
                    "Displaced": displaced,
                    "Educational_special_needs": 0,
                    "Debtor": debtor,
                    "Tuition_fees_up_to_date": tuition,
                    "Gender": gender,
                    "Scholarship_holder": scholarship,
                    "Age_at_enrollment": age,
                    "International": 0,
                    "Curricular_units_1st_sem_credited": credited_1,
                    "Curricular_units_1st_sem_enrolled": enrolled_1,
                    "Curricular_units_1st_sem_evaluations": eval_1,
                    "Curricular_units_1st_sem_approved": approved_1,
                    "Curricular_units_1st_sem_grade": grade_1,
                    "Curricular_units_1st_sem_without_evaluations": without_eval_1,
                    "Curricular_units_2nd_sem_credited": credited_2,
                    "Curricular_units_2nd_sem_enrolled": enrolled_2,
                    "Curricular_units_2nd_sem_evaluations": eval_2,
                    "Curricular_units_2nd_sem_approved": approved_2,
                    "Curricular_units_2nd_sem_grade": grade_2,
                    "Curricular_units_2nd_sem_without_evaluations": without_eval_2,
                    "Unemployment_rate": 10.8,
                    "Inflation_rate": 1.4,
                    "GDP": 1.74,
                }

                input_df = pd.DataFrame([input_data])
                input_df = engineer_features(input_df)
                input_df = input_df[feature_names]

                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                predicted_label = le.inverse_transform([prediction])[0]

                # Display result
                if predicted_label == "Dropout":
                    st.markdown(
                        '<div class="risk-high">\u26a0\ufe0f RISIKO TINGGI - Prediksi: DROPOUT</div>',
                        unsafe_allow_html=True,
                    )
                elif predicted_label == "Enrolled":
                    st.markdown(
                        '<div class="risk-medium">\u26a0\ufe0f RISIKO SEDANG - Prediksi: ENROLLED</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="risk-low">\u2714\ufe0f RISIKO RENDAH - Prediksi: GRADUATE</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("---")
                st.subheader(" \U0001f4ca Probabilitas per Kelas")

                cols = st.columns(3)
                colors_gauge = ["#406093", "#FFF799", "#91D06C"]
                for i, cls_name in enumerate(le.classes_):
                    with cols[i]:
                        fig = create_gauge(probabilities[i], cls_name, colors_gauge[i])
                        st.plotly_chart(fig, use_container_width=True)

                if hasattr(model.named_steps["clf"], "feature_importances_"):
                    st.subheader(" \U0001f527 Faktor Utama yang Mempengaruhi")
                    importances = model.named_steps["clf"].feature_importances_
                    feat_imp = (
                        pd.Series(importances, index=feature_names)
                        .sort_values(ascending=False)
                        .head(10)
                    )
                    fig = px.bar(
                        x=feat_imp.values,
                        y=feat_imp.index,
                        orientation="h",
                        labels={"x": "Importance", "y": "Feature"},
                        title="Top 10 Feature Importance",
                        color=feat_imp.values,
                        color_discrete_sequence=["#4C8CE4"],
                    )
                    fig.update_layout(
                        yaxis={"categoryorder": "total ascending"}, height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f" Error dalam prediksi: {str(e)}")


# TAB 2: BATCH PREDICTION

with tab2:
    st.subheader(" \U0001f4c2 Batch Prediction - Upload CSV")
    st.markdown(
        "Upload file CSV dengan kolom sesuai dataset asli untuk prediksi massal."
    )

    uploaded_file = st.file_uploader(" Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        with st.spinner("Membaca dan memproses file..."):
            try:
                batch_df = pd.read_csv(uploaded_file, sep=";")
                st.success(
                    f" File berhasil dibaca: {batch_df.shape[0]} baris, {batch_df.shape[1]} kolom"
                )

                if "Status" in batch_df.columns:
                    batch_df = batch_df.drop("Status", axis=1)
                if "Status_encoded" in batch_df.columns:
                    batch_df = batch_df.drop("Status_encoded", axis=1)

                batch_eng = engineer_features(batch_df)
                missing_cols = [c for c in feature_names if c not in batch_eng.columns]
                if missing_cols:
                    st.warning(f"Kolom hilang: {missing_cols}. Diisi dengan 0.")
                    for c in missing_cols:
                        batch_eng[c] = 0
                batch_input = batch_eng[feature_names]

                predictions = model.predict(batch_input)
                probas = model.predict_proba(batch_input)
                pred_labels = le.inverse_transform(predictions)

                results = batch_df.copy()
                results["Prediksi"] = pred_labels
                for i, cls in enumerate(le.classes_):
                    results[f"Prob_{cls}"] = probas[:, i].round(3)

                # Color
                st.subheader(" \U0001f4ca Hasil Prediksi")
                dropout_count = (pred_labels == "Dropout").sum()
                enrolled_count = (pred_labels == "Enrolled").sum()
                graduate_count = (pred_labels == "Graduate").sum()

                c1, c2, c3 = st.columns(3)
                c1.metric(
                    " \U0001f6a8 Dropout",
                    dropout_count,
                    f"{dropout_count / len(pred_labels) * 100:.1f}%",
                )
                c2.metric(
                    " \u26a0\ufe0f Enrolled",
                    enrolled_count,
                    f"{enrolled_count / len(pred_labels) * 100:.1f}%",
                )
                c3.metric(
                    " \U0001f393 Graduate",
                    graduate_count,
                    f"{graduate_count / len(pred_labels) * 100:.1f}%",
                )

                def highlight_risk(row):
                    if row["Prediksi"] == "Dropout":
                        return ["background-color: #406093; color: white;"] * len(row)
                    elif row["Prediksi"] == "Graduate":
                        return ["background-color: #91D06C; color: white;"] * len(row)
                    return [""] * len(row)

                display_cols = [
                    "Prediksi",
                    "Prob_Dropout",
                    "Prob_Enrolled",
                    "Prob_Graduate",
                ]
                extra = [
                    c
                    for c in ["Age_at_enrollment", "Gender", "Course"]
                    if c in results.columns
                ]
                st.dataframe(
                    results[extra + display_cols].style.apply(highlight_risk, axis=1),
                    use_container_width=True,
                    height=400,
                )

                # Download button
                csv_output = results.to_csv(index=False, sep=";")
                st.download_button(
                    " \U0001f4e5 Download Hasil Prediksi (CSV)",
                    csv_output,
                    "prediction_results.csv",
                    "text/csv",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.info(
            " \U0001f4c1 Upload file CSV untuk memulai batch prediction. Gunakan format yang sama dengan dataset asli (separator: ;)"
        )


# TAB 3: DASHBOARD INSIGHT

with tab3:
    st.subheader(" \U0001f4ca Dashboard Insight - Exploratory Data Analysis")

    try:
        df_viz = pd.read_csv("dataset/data.csv", sep=";")
        df_viz["GPA"] = (
            df_viz["Curricular_units_1st_sem_grade"]
            + df_viz["Curricular_units_2nd_sem_grade"]
        ) / 2

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        total = len(df_viz)
        dropout_n = (df_viz["Status"] == "Dropout").sum()
        c1.metric("Total Mahasiswa", f"{total:,}")
        c2.metric("Dropout", f"{dropout_n}", f"{dropout_n / total * 100:.1f}%")
        c3.metric("Graduate", f"{(df_viz['Status'] == 'Graduate').sum()}")
        c4.metric("Avg GPA", f"{df_viz['GPA'].mean():.2f}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Distribusi Status Mahasiswa")
            fig = px.pie(
                df_viz,
                names="Status",
                color="Status",
                color_discrete_map={
                    "Dropout": "#406093",
                    "Enrolled": "#FFF799",
                    "Graduate": "#91D06C",
                },
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Distribusi GPA per Status")
            fig = px.box(
                df_viz,
                x="Status",
                y="GPA",
                color="Status",
                color_discrete_map={
                    "Dropout": "#406093",
                    "Enrolled": "#FFF799",
                    "Graduate": "#91D06C",
                },
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### Dropout Rate by Tuition Status")
            tuit_data = (
                df_viz.groupby(["Tuition_fees_up_to_date", "Status"])
                .size()
                .reset_index(name="Count")
            )
            fig = px.bar(
                tuit_data,
                x="Tuition_fees_up_to_date",
                y="Count",
                color="Status",
                barmode="group",
                color_discrete_map={
                    "Dropout": "#406093",
                    "Enrolled": "#FFF799",
                    "Graduate": "#91D06C",
                },
            )
            fig.update_layout(
                height=350, xaxis_title="Tuition Up to Date (0=No, 1=Yes)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown("#### Distribusi Usia per Status")
            fig = px.histogram(
                df_viz,
                x="Age_at_enrollment",
                color="Status",
                barmode="overlay",
                opacity=0.7,
                color_discrete_map={
                    "Dropout": "#406093",
                    "Enrolled": "#FFF799",
                    "Graduate": "#91D06C",
                },
            )
            fig.update_layout(height=350, xaxis_title="Usia Saat Pendaftaran")
            st.plotly_chart(fig, use_container_width=True)

        if hasattr(model.named_steps["clf"], "feature_importances_"):
            st.markdown("#### Top Risk Factors (Feature Importance)")
            importances = model.named_steps["clf"].feature_importances_
            feat_imp = (
                pd.Series(importances, index=feature_names)
                .sort_values(ascending=False)
                .head(15)
            )
            fig = px.bar(
                x=feat_imp.values,
                y=feat_imp.index,
                orientation="h",
                color=feat_imp.values,
                color_discrete_sequence=["#4C8CE4"],
                labels={"x": "Importance", "y": "Feature"},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error memuat data: {str(e)}")


# FOOTER
st.markdown("---")
st.markdown(
    f"""<div style='text-align: center; color: #888; font-size: 0.85rem;'>
        Jaya Jaya Institut - Student Risk Prediction System |
        Model: {best_model_name}
    </div>""",
    unsafe_allow_html=True,
)
