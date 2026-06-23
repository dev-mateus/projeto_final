from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
N_SPLITS = 5
DEFAULT_DATASET_NAME = "creditcard.csv"
FAST_MODE_DEFAULT_MAX_ROWS = 120_000


def get_project_dir() -> Path:
    return Path.cwd()

def get_default_dataset_path() -> Path:
    return get_project_dir() / DEFAULT_DATASET_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline de deteccao de fraude com comparacao de modelos, "
            "tecnicas de balanceamento e metricas para desbalanceamento extremo."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=(
            "Caminho para creditcard.csv (dataset Kaggle ULB). "
            "Se omitido, usa projeto_final/creditcard.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=get_project_dir() / "resultados",
        help="Diretorio de saida para tabelas e figuras.",
    )
    parser.add_argument(
        "--include-rf",
        action="store_true",
        help="Inclui Random Forest na comparacao.",
    )
    parser.add_argument(
        "--no-svm",
        action="store_true",
        help="Remove SVM da comparacao (util para execucao rapida em datasets grandes).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=N_SPLITS,
        help="Numero de folds na validacao cruzada (padrao: 5).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limita quantidade de linhas carregadas do dataset.",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help=(
            "Modo rapido para iteracao local: reduz folds para 3, desativa SVM e limita "
            "linhas (se --max-rows nao for informado)."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Numero de jobs para paralelismo em validacao cruzada. -1 usa todos os nucleos.",
    )
    return parser.parse_args()


def resolve_data_path(path: Path | None) -> Path:
    if path is None:
        return get_default_dataset_path()

    if path.is_absolute():
        return path

    return Path.cwd() / path


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    tables = output_dir / "tabelas"
    figures = output_dir / "figuras"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return {"base": output_dir, "tables": tables, "figures": figures}


def load_data(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    df = pd.read_csv(path, nrows=max_rows)
    if "Class" not in df.columns:
        raise ValueError("Dataset deve conter a coluna alvo 'Class'.")

    float64_cols = df.select_dtypes(include=["float64"]).columns
    if len(float64_cols) > 0:
        df[float64_cols] = df[float64_cols].astype(np.float32)

    df["Class"] = df["Class"].astype(np.int8)

    return df


def generate_eda_artifacts(df: pd.DataFrame, out_tables: Path, out_figures: Path) -> None:
    class_counts = df["Class"].value_counts().sort_index()
    class_ratio = class_counts / class_counts.sum()

    eda_summary = {
        "n_linhas": int(df.shape[0]),
        "n_colunas": int(df.shape[1]),
        "n_fraudes": int(class_counts.get(1, 0)),
        "n_legitimas": int(class_counts.get(0, 0)),
        "fraude_ratio": float(class_ratio.get(1, 0.0)),
        "missing_total": int(df.isna().sum().sum()),
    }

    with (out_tables / "eda_resumo.json").open("w", encoding="utf-8") as fp:
        json.dump(eda_summary, fp, indent=2)

    desc = df[["Time", "Amount"]].describe().T
    desc.to_csv(out_tables / "eda_time_amount_describe.csv", index=True)

    plt.figure(figsize=(6, 4))
    sns.barplot(x=["Legitima", "Fraude"], y=[class_counts.get(0, 0), class_counts.get(1, 0)])
    plt.title("Distribuicao de Classes")
    plt.ylabel("Quantidade")
    plt.tight_layout()
    plt.savefig(out_figures / "eda_distribuicao_classes.png", dpi=130)
    plt.close()

    for col in ["Amount", "Time"]:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=60, kde=False)
        plt.title(f"Distribuicao de {col}")
        plt.tight_layout()
        plt.savefig(out_figures / f"eda_hist_{col.lower()}.png", dpi=130)
        plt.close()


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def gmean_from_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = safe_div(tp, tp + fn)
    tnr = safe_div(tn, tn + fp)
    return math.sqrt(max(tpr * tnr, 0.0))


def get_scores(model: Any, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x_test)[:, 1]
        return np.clip(prob, 0.0, 1.0)

    if hasattr(model, "decision_function"):
        score = model.decision_function(x_test)
        score = np.asarray(score)
        score_min = float(np.min(score))
        score_max = float(np.max(score))
        if score_max - score_min < 1e-12:
            return np.full_like(score, 0.5, dtype=float)
        return (score - score_min) / (score_max - score_min)

    pred = model.predict(x_test)
    return np.asarray(pred, dtype=float)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "gmean": gmean_from_confusion(y_true, y_pred),
        "fbeta_2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "brier": brier_score_loss(y_true, y_score),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
    except ValueError:
        metrics["pr_auc"] = float("nan")

    return metrics


def make_model_catalog(include_rf: bool, include_svm: bool = True) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {
        "dummy": {
            "model": DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
            "scale": False,
        },
        "logistic_regression": {
            "model": LogisticRegression(max_iter=400, solver="liblinear", random_state=RANDOM_STATE),
            "scale": True,
        },
    }

    if include_svm:
        catalog["svm_rbf"] = {
            "model": SVC(
                kernel="rbf",
                probability=True,
                class_weight=None,
                random_state=RANDOM_STATE,
                cache_size=700,
            ),
            "scale": True,
        }

    if include_rf:
        catalog["random_forest"] = {
            "model": RandomForestClassifier(
                n_estimators=220,
                max_depth=None,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "scale": False,
        }

    return catalog


def make_scenario_catalog() -> dict[str, Any]:
    return {
        "sem_balanceamento": None,
        "smote": SMOTE(random_state=RANDOM_STATE),
        "undersampling": RandomUnderSampler(random_state=RANDOM_STATE),
    }


def build_pipeline(model: Any, use_scaler: bool, sampler: Any) -> ImbPipeline:
    steps: list[tuple[str, Any]] = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("clf", model))
    return ImbPipeline(steps=steps)


def _process_fold(
    x_values: np.ndarray,
    y_values: np.ndarray,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    model: Any,
    scale: bool,
    sampler: Any,
    scenario_name: str,
    model_name: str,
    fold_id: int,
) -> dict[str, Any]:
    """Processa um único fold (executado em paralelo)."""
    x_train, x_test = x_values[idx_train], x_values[idx_test]
    y_train, y_test = y_values[idx_train], y_values[idx_test]

    sampler_instance = clone(sampler) if sampler is not None else None
    model_instance = clone(model)
    pipe = build_pipeline(
        model=model_instance,
        use_scaler=bool(scale),
        sampler=sampler_instance,
    )
    pipe.fit(x_train, y_train)

    y_pred = np.asarray(pipe.predict(x_test), dtype=int)
    y_score = get_scores(pipe, x_test)
    y_test_np = np.asarray(y_test, dtype=int)

    metrics = compute_metrics(y_test_np, y_pred, y_score)
    metrics.update(
        {
            "scenario": scenario_name,
            "model": model_name,
            "fold": fold_id,
        }
    )

    return {
        "metrics": metrics,
        "y_true": y_test_np,
        "y_pred": y_pred,
        "y_score": y_score,
    }


def run_cv_experiments(
    x: pd.DataFrame,
    y: pd.Series,
    model_catalog: dict[str, dict[str, Any]],
    scenario_catalog: dict[str, Any],
    n_splits: int,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    total_scenarios = len(scenario_catalog)
    total_models = len(model_catalog)
    x_values = x.to_numpy(copy=False)
    y_values = y.to_numpy(dtype=np.int8, copy=False)
    splits = list(skf.split(x_values, y_values))

    rows_fold: list[dict[str, Any]] = []
    oof_store: dict[str, dict[str, np.ndarray]] = {}

    for scenario_idx, (scenario_name, sampler) in enumerate(scenario_catalog.items(), start=1):
        print(
            f"[3/6] Cenario {scenario_idx}/{total_scenarios}: {scenario_name}",
            flush=True,
        )
        for model_idx, (model_name, cfg) in enumerate(model_catalog.items(), start=1):
            key = f"{scenario_name}__{model_name}"

            print(
                (
                    f"[3/6]   Modelo {model_idx}/{total_models}: {model_name} "
                    f"| cenario={scenario_name}"
                ),
                flush=True,
            )

            fold_tasks = [
                delayed(_process_fold)(
                    x_values,
                    y_values,
                    idx_train,
                    idx_test,
                    cfg["model"],
                    cfg["scale"],
                    sampler,
                    scenario_name,
                    model_name,
                    fold_id,
                )
                for fold_id, (idx_train, idx_test) in enumerate(splits, start=1)
            ]

            fold_results = Parallel(n_jobs=n_jobs, verbose=1)(
                fold_tasks
            )

            y_true_all: list[np.ndarray] = []
            y_score_all: list[np.ndarray] = []
            y_pred_all: list[np.ndarray] = []

            for fold_id, result in enumerate(fold_results, start=1):
                rows_fold.append(result["metrics"])
                y_true_all.append(result["y_true"])
                y_pred_all.append(result["y_pred"])
                y_score_all.append(result["y_score"])

                print(
                    (
                        f"[3/6]     Fold {fold_id}/{n_splits} concluido "
                        f"| cenario={scenario_name} | modelo={model_name} "
                        f"| recall={result['metrics']['recall']:.4f} | pr_auc={result['metrics']['pr_auc']:.4f}"
                    ),
                    flush=True,
                )

            oof_store[key] = {
                "y_true": np.concatenate(y_true_all),
                "y_pred": np.concatenate(y_pred_all),
                "y_score": np.concatenate(y_score_all),
            }

            model_metrics = compute_metrics(
                oof_store[key]["y_true"],
                oof_store[key]["y_pred"],
                oof_store[key]["y_score"],
            )
            print(
                (
                    f"[3/6]   Modelo concluido: {model_name} | cenario={scenario_name} "
                    f"| recall_oof={model_metrics['recall']:.4f} "
                    f"| pr_auc_oof={model_metrics['pr_auc']:.4f}"
                ),
                flush=True,
            )

    fold_df = pd.DataFrame(rows_fold)

    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "mcc",
        "balanced_accuracy",
        "gmean",
        "fbeta_2",
        "brier",
    ]

    summary = (
        fold_df.groupby(["scenario", "model"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "scenario",
        "model",
        *[
            f"{metric}_{stat}"
            for metric in metric_cols
            for stat in ["mean", "std"]
        ],
    ]

    return fold_df, summary, oof_store


def choose_best_config(summary_df: pd.DataFrame) -> tuple[str, str, str]:
    ordered = summary_df.sort_values(
        by=["pr_auc_mean", "recall_mean", "mcc_mean"],
        ascending=[False, False, False],
    )
    best = ordered.iloc[0]
    scenario = str(best["scenario"])
    model = str(best["model"])
    key = f"{scenario}__{model}"
    return scenario, model, key


def save_accuracy_vs_pr_table(summary_df: pd.DataFrame, out_tables: Path) -> None:
    table = summary_df[
        [
            "scenario",
            "model",
            "accuracy_mean",
            "pr_auc_mean",
            "recall_mean",
            "balanced_accuracy_mean",
            "mcc_mean",
        ]
    ].sort_values(by=["accuracy_mean"], ascending=False)
    table.to_csv(out_tables / "comparacao_acuracia_vs_metricas_imbalance.csv", index=False)


def plot_main_metric_bars(summary_df: pd.DataFrame, out_figures: Path) -> None:
    plot_cols = [
        "accuracy_mean",
        "recall_mean",
        "pr_auc_mean",
        "mcc_mean",
        "balanced_accuracy_mean",
        "gmean_mean",
        "fbeta_2_mean",
    ]

    data = summary_df.copy()
    data["config"] = data["scenario"] + " | " + data["model"]
    melted = data.melt(
        id_vars=["config"],
        value_vars=plot_cols,
        var_name="metric",
        value_name="value",
    )

    plt.figure(figsize=(14, 6))
    sns.barplot(data=melted, x="config", y="value", hue="metric")
    plt.xticks(rotation=35, ha="right")
    plt.title("Comparacao de Configuracoes por Metricas")
    plt.ylabel("Valor medio (CV)")
    plt.xlabel("Configuracao")
    plt.tight_layout()
    plt.savefig(out_figures / "comparacao_metricas_principais.png", dpi=130)
    plt.close()


def plot_curves_for_best(
    best_scenario: str,
    best_model: str,
    oof: dict[str, np.ndarray],
    out_figures: Path,
) -> None:
    y_true = oof["y_true"]
    y_pred = oof["y_pred"]
    y_score = oof["y_score"]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC-ROC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.7)
    plt.title(f"Curva ROC - Melhor Config\n{best_scenario} | {best_model}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_figures / "melhor_config_curva_roc.png", dpi=130)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"AUC-PR = {pr_auc:.4f}")
    base_rate = float(np.mean(y_true))
    plt.hlines(base_rate, xmin=0, xmax=1, linestyles="--", alpha=0.7, label="Baseline")
    plt.title(f"Curva Precision-Recall - Melhor Config\n{best_scenario} | {best_model}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_figures / "melhor_config_curva_pr.png", dpi=130)
    plt.close()

    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Modelo")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    plt.title(f"Curva de Calibracao - Melhor Config\n{best_scenario} | {best_model}")
    plt.xlabel("Probabilidade media predita")
    plt.ylabel("Frequencia observada")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_figures / "melhor_config_curva_calibracao.png", dpi=130)
    plt.close()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitima", "Fraude"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Matriz de Confusao - Melhor Config\n{best_scenario} | {best_model}")
    plt.tight_layout()
    plt.savefig(out_figures / "melhor_config_matriz_confusao.png", dpi=130)
    plt.close()


def save_best_config_report(
    best_scenario: str,
    best_model: str,
    best_oof: dict[str, np.ndarray],
    out_tables: Path,
) -> None:
    y_true = best_oof["y_true"]
    y_pred = best_oof["y_pred"]
    y_score = best_oof["y_score"]

    best_metrics = compute_metrics(y_true, y_pred, y_score)
    best_metrics["scenario"] = best_scenario
    best_metrics["model"] = best_model

    pd.DataFrame([best_metrics]).to_csv(out_tables / "melhor_config_metricas_oof.csv", index=False)


def main() -> None:
    args = parse_args() if "__file__" in globals() else argparse.Namespace(
        data=Path("/kaggle/input/datasets/mateuspereira1988/creditcard-fraud/creditcard.csv"),
        output_dir=Path("./resultados"),
        include_rf=False,
        no_svm=False,
        n_splits=5,
        max_rows=None,
        fast_mode=False,
    )
    if args.fast_mode:
        if not args.no_svm:
            print("[0/6] fast-mode: desativando SVM para reduzir tempo de execucao.", flush=True)
            args.no_svm = True
        if args.n_splits == N_SPLITS:
            args.n_splits = 3
            print("[0/6] fast-mode: usando 3 folds na validacao cruzada.", flush=True)
        if args.max_rows is None:
            args.max_rows = FAST_MODE_DEFAULT_MAX_ROWS
            print(
                f"[0/6] fast-mode: limitando carga para {FAST_MODE_DEFAULT_MAX_ROWS} linhas.",
                flush=True,
            )

    if args.n_splits < 2:
        raise ValueError("--n-splits deve ser >= 2.")

    data_path = resolve_data_path(args.data)
    paths = ensure_output_dirs(args.output_dir)

    print(f"[1/6] Carregando dados de: {data_path}", flush=True)
    df = load_data(data_path, max_rows=args.max_rows)

    print("[2/6] Gerando artefatos de EDA...", flush=True)
    generate_eda_artifacts(df, paths["tables"], paths["figures"])

    x = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    model_catalog = make_model_catalog(include_rf=args.include_rf, include_svm=not args.no_svm)
    scenario_catalog = make_scenario_catalog()

    print("[3/6] Rodando validacao cruzada (pode demorar)...", flush=True)
    fold_df, summary_df, oof_store = run_cv_experiments(
        x=x,
        y=y,
        model_catalog=model_catalog,
        scenario_catalog=scenario_catalog,
        n_splits=args.n_splits,
        n_jobs=args.n_jobs,
    )

    print("[4/6] Salvando tabelas de metricas...", flush=True)
    fold_df.to_csv(paths["tables"] / "metricas_por_fold.csv", index=False)
    summary_df.to_csv(paths["tables"] / "metricas_resumo_media_desvio.csv", index=False)
    save_accuracy_vs_pr_table(summary_df, paths["tables"])

    print("[5/6] Gerando graficos comparativos...", flush=True)
    plot_main_metric_bars(summary_df, paths["figures"])

    best_scenario, best_model, best_key = choose_best_config(summary_df)
    best_oof = oof_store[best_key]
    plot_curves_for_best(best_scenario, best_model, best_oof, paths["figures"])
    save_best_config_report(best_scenario, best_model, best_oof, paths["tables"])

    print("[6/6] Finalizado.", flush=True)
    print(f"Melhor configuracao por AUC-PR: {best_scenario} | {best_model}", flush=True)
    print(f"Resultados em: {paths['base']}", flush=True)


if __name__ == "__main__":
    main()