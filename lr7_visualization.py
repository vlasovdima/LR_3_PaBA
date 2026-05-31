# -*- coding: utf-8 -*-
"""
ЛР 7: Визуализация данных с использованием Seaborn и Plotly
Датасет: ментальное здоровье подростков (~600 записей, генерация на основе
         реального паттерна Kaggle dataset "algozee/teenager-menthal-healy")
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#                          
# 0. Пути
#                          
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(OUT_DIR, "teen_mental_health.csv")
FIG_DIR = os.path.join(OUT_DIR, "figures")
HTML_DIR = os.path.join(OUT_DIR, "html")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)

#                          
# 1. Генерация датасета (т.к. Kaggle API недоступен — создаём реалистичный
#    синтетический датасет на основе реальных паттернов)
#                          
def generate_dataset(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(13, 20, size=n)
    gender = rng.choice(["Male", "Female", "Non-binary"], size=n, p=[0.45, 0.45, 0.10])

    # Уровень стресса зависит от возраста, пола, сна
    stress_base = 5.0 + (age - 13) * 0.3
    stress_noise = rng.normal(0, 1.5, n)
    stress_level = np.clip(np.round(stress_base + stress_noise, 1), 1, 10)

    sleep_hours = np.clip(np.round(rng.normal(7.0, 1.5, n) - (stress_level - 5) * 0.1, 1), 3, 12)

    # Физическая активность (дни в неделю)
    physical_activity = rng.integers(0, 8, size=n)

    # Социальные сети (часы в день)
    social_media_hours = np.clip(np.round(rng.exponential(3.0, n), 1), 0, 12)

    # Отношения в семье (1-10)
    family_relationship = np.clip(np.round(rng.normal(6.5, 1.8, n), 1), 1, 10)

    # Успеваемость (GPA, 0-4)
    gpa = np.clip(np.round(
        3.0
        - (stress_level - 5) * 0.05
        + (sleep_hours - 7) * 0.05
        + rng.normal(0, 0.4, n)
    , 2), 0.5, 4.0)

    # Депрессивные симптомы (шкала PHQ-0..27, упрощённо 0..20)
    depression_score = np.clip(np.round(
        (stress_level - 3) * 1.2
        + (10 - family_relationship) * 0.5
        + social_media_hours * 0.3
        + rng.normal(0, 1.5, n)
    , 1), 0, 20)

    # Тревожность (0..20)
    anxiety_score = np.clip(np.round(
        (stress_level - 2) * 1.0
        + (sleep_hours < 6) * 2.0
        - physical_activity * 0.2
        + rng.normal(0, 1.5, n)
    , 1), 0, 20)

    # Категория ментального здоровья
    def classify(row):
        if row["depression_score"] <= 5 and row["anxiety_score"] <= 5 and row["stress_level"] <= 4:
            return "Good"
        elif row["depression_score"] <= 10 and row["anxiety_score"] <= 10 and row["stress_level"] <= 6:
            return "Moderate"
        else:
            return "Poor"

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "stress_level": stress_level,
        "sleep_hours": sleep_hours,
        "physical_activity": physical_activity,
        "social_media_hours": social_media_hours,
        "family_relationship": family_relationship,
        "gpa": gpa,
        "depression_score": depression_score,
        "anxiety_score": anxiety_score,
    })
    df["mental_health_category"] = df.apply(classify, axis=1)

    # Добавляем пропуски (~5% случайно)
    for col in ["sleep_hours", "social_media_hours", "family_relationship", "gpa"]:
        mask = rng.random(n) < 0.05
        df.loc[mask, col] = np.nan

    return df


#                          
# 2. Предобработка
#                          
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    report = []

    # Пропуски до заполнения
    missing_before = df.isnull().sum()
    report.append("=== Пропуски до заполнения ===")
    report.append(missing_before[missing_before > 0].to_string())

    # Заполняем числовые медианой
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)
            report.append(f"  {col}: заполнено медианой {med}")

    # Приведение типов
    df["age"] = df["age"].astype(int)
    df["physical_activity"] = df["physical_activity"].astype(int)
    df["mental_health_category"] = pd.Categorical(
        df["mental_health_category"], categories=["Good", "Moderate", "Poor"], ordered=True
    )

    missing_after = df.isnull().sum().sum()
    report.append(f"\nПропуски после заполнения: {missing_after}")

    report.append(f"\nРазмер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
    report.append(f"\nТипы данных:\n{df.dtypes.to_string()}")

    return df, "\n".join(report)


#                          
# 3. Seaborn визуализации
#                          
def seaborn_plots(df: pd.DataFrame) -> list[str]:
    saved = []
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    # 3.1 Распределение уровня стресса (histplot + KDE)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["stress_level"], bins=20, kde=True, color="steelblue", ax=ax)
    ax.set_title("Распределение уровня стресса у подростков", fontsize=14)
    ax.set_xlabel("Уровень стресса (1-10)")
    ax.set_ylabel("Количество учеников")
    path = os.path.join(FIG_DIR, "seaborn_1_stress_distribution.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # 3.2 Scatterplot: сон vs депрессия, цвет — категория ментального здоровья
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=df, x="sleep_hours", y="depression_score",
        hue="mental_health_category", palette="Set1",
        alpha=0.7, s=50, ax=ax
    )
    ax.set_title("Сон и уровень депрессии", fontsize=14)
    ax.set_xlabel("Часы сна")
    ax.set_ylabel("Баллы депрессии")
    path = os.path.join(FIG_DIR, "seabern_2_scatter_sleep_depression.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # 3.3 Boxplot: тревожность по полу
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(
        data=df, x="gender", y="anxiety_score",
        hue="gender", palette="pastel", ax=ax, legend=False
    )
    ax.set_title("Уровень тревожности по полу", fontsize=14)
    ax.set_xlabel("Пол")
    ax.set_ylabel("Баллы тревожности")
    path = os.path.join(FIG_DIR, "seaborn_3_boxplot_anxiety_gender.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # 3.4 Heatmap корреляций (числовые признаки)
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, linewidths=0.5, ax=ax
    )
    ax.set_title("Корреляционная матрица числовых признаков", fontsize=14)
    path = os.path.join(FIG_DIR, "seaborn_4_correlation_heatmap.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    return saved


#                          
# 4. Plotly визуализации
#                          
def plotly_plots(df: pd.DataFrame) -> list[str]:
    saved = []

    # 4.1 Scatterplot
    fig = px.scatter(
        df, x="social_media_hours", y="anxiety_score",
        color="mental_health_category",
        size="stress_level",
        hover_data=["age", "gpa", "sleep_hours"],
        title="Соцсети vs тревожность (размер точки = стресс)",
        labels={
            "social_media_hours": "Соцсети (ч/день)",
            "anxiety_score": "Баллы тревожности",
            "mental_health_category": "Ментальное здоровье",
        },
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig.update_layout(width=900, height=600)
    path = os.path.join(HTML_DIR, "plotly_1_scatter.html")
    fig.write_html(path)
    saved.append(path)

    # 4.2 Line plot: средний стресс по возрасту
    age_stress = df.groupby("age").agg(
        stress_mean=("stress_level", "mean"),
        depression_mean=("depression_score", "mean"),
        count=("age", "size"),
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=age_stress["age"], y=age_stress["stress_mean"],
            mode="lines+markers", name="Средний стресс",
            line=dict(color="tomato", width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=age_stress["age"], y=age_stress["depression_mean"],
            mode="lines+markers", name="Средняя депрессия",
            line=dict(color="steelblue", width=2),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Средний уровень стресса и депрессии по возрасту",
        width=900, height=500,
    )
    fig.update_xaxes(title_text="Возраст")
    fig.update_yaxes(title_text="Средний стресс", secondary_y=False)
    fig.update_yaxes(title_text="Средняя депрессия", secondary_y=True)
    path = os.path.join(HTML_DIR, "plotly_2_line_age.html")
    fig.write_html(path)
    saved.append(path)

    # 4.3 Bar chart: средний GPA по категориям ментального здоровья и полу
    gpa_gender = df.groupby(["mental_health_category", "gender"])["gpa"].mean().reset_index()

    fig = px.bar(
        gpa_gender, x="mental_health_category", y="gpa",
        color="gender", barmode="group",
        title="Средний балл успеваемости (GPA) по категориям ментального здоровья",
        labels={
            "mental_health_category": "Категория ментального здоровья",
            "gpa": "Средний GPA",
            "gender": "Пол",
        },
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(width=850, height=500)
    path = os.path.join(HTML_DIR, "plotly_3_bar_gpa.html")
    fig.write_html(path)
    saved.append(path)

    # 4.4 Heatmap (интерактивный)
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr().round(2)

    fig = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r",
        title="Интерактивная корреляционная матрица",
        labels=dict(color="Корреляция"),
        zmin=-1, zmax=1,
    )
    fig.update_layout(width=850, height=700)
    path = os.path.join(HTML_DIR, "plotly_4_heatmap.html")
    fig.write_html(path)
    saved.append(path)

    # 4.5 Scatterplot + dropdown / slider (интерактивный элемент)
    ages = sorted(df["age"].unique())
    dfs_list = []

    for a in ages:
        subset = df[df["age"] == a].copy()
        subset["age_filter"] = str(a)
        dfs_list.append(subset)
    df_all = pd.concat(dfs_list)

    fig = px.scatter(
        df_all, x="sleep_hours", y="depression_score",
        color="mental_health_category",
        animation_frame="age_filter",
        hover_data=["gpa", "social_media_hours"],
        title="Сон vs депрессия (фильтр по возрасту — слайдер)",
        labels={
            "sleep_hours": "Часы сна",
            "depression_score": "Баллы депрессии",
            "age_filter": "Возраст",
        },
        color_discrete_sequence=px.colors.qualitative.Set1,
        range_x=[3, 12], range_y=[0, 20],
    )

    # Dropdown для выбора пола
    buttons = [
        dict(
            label="Все",
            method="update",
            args=[{"visible": [True] * 3}],
        ),
    ]
    for g in df["gender"].unique():
        buttons.append(
            dict(
                label=g,
                method="update",
                args=[{"visible": [g]}],
            )
        )
    fig.update_layout(
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.02, y=1.15,
            xanchor="left", yanchor="top",
        )],
        width=950, height=600,
    )

    path = os.path.join(HTML_DIR, "plotly_5_animated_dropdown.html")
    fig.write_html(path)
    saved.append(path)

    return saved


#                          
# Main
#                          
def main():
    print("Генерация датасета...")
    df_raw = generate_dataset(n=600)
    df_raw.to_csv(DATA_CSV, index=False)
    print(f"  Сохранено: {DATA_CSV}  ({df_raw.shape[0]} строк)")

    print("\nПредобработка...")
    df, prep_report = preprocess(df_raw)
    print(prep_report)

    print(f"\nSeaborn визуализации:")
    for p in seaborn_plots(df):
        print(f"  saved: {p}")

    print(f"\nPlotly визуализации:")
    for p in plotly_plots(df):
        print(f"  saved: {p}")

    # Сохраняем отчёт о предобработке
    report_path = os.path.join(OUT_DIR, "preprocessing_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(prep_report)
    print(f"\nОтчёт предобработки: {report_path}")

    # Выводим описательную статистику
    print("\n=== Описательная статистика ===")
    print(df.describe().round(2).to_string())
    print(f"\nРаспределение категорий ментального здоровья:")
    print(df["mental_health_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
