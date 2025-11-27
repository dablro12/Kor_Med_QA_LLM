import pandas as pd
from typing import Optional

class ClinicalQAEvaluator:
    def __init__(self, parquet_path: str):
        """
        parquet 파일을 읽어서 DataFrame으로 저장
        """
        self.df = pd.read_parquet(parquet_path)
        self._validate_columns()
        self._prepare_metrics()

    def _validate_columns(self):
        """
        데이터프레임 컬럼 확인
        """
        required_columns = [
            "question_id", "gt_answer", "pred_answer", "pred_explanation",
            "first_token_latency_s", "time_per_token_s", "vram_used_MB",
            "flops_this", "flops_per_token", "cost_per_token_s"
        ]
        missing_cols = [c for c in required_columns if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in input: {missing_cols}")

    def _prepare_metrics(self):
        """
        정답 여부 계산
        gt_answer, pred_answer가 문자나 숫자 등 다른 타입이어도 자동으로 올바르게 비교됩니다.
        """
        # gt_answer와 pred_answer가 각각 int/float/str일 수 있으므로, 이를 문자열로 비교
        self.df["is_correct"] = self.df["gt_answer"].astype(str) == self.df["pred_answer"].astype(str)

    def summary(self) -> pd.DataFrame:
        """
        모델 성능 요약표 반환
        """
        return pd.DataFrame({
            "total_samples":        [len(self.df)],
            "correct_predictions":  [self.df["is_correct"].sum()],
            "accuracy (%)":         [round(self.df["is_correct"].mean() * 100, 2)],
            "avg_first_token_latency (s)": [self.df["first_token_latency_s"].mean()],
            "avg_time_per_token (s)":     [self.df["time_per_token_s"].mean()],
            "avg_vram_usage (MB)":        [self.df["vram_used_MB"].mean()],
            "mean_flops":                 [self.df["flops_this"].mean()],
            "mean_flops_per_token":       [self.df["flops_per_token"].mean()],
            "mean_cost_per_token (s)":    [self.df["cost_per_token_s"].mean()]
        })

    def per_sample_table(self) -> pd.DataFrame:
        """
        원본 결과 + 정답 여부 포함 테이블 반환
        """
        return self.df[[
            "question_id", "gt_answer", "pred_answer", "pred_explanation",
            "is_correct", "first_token_latency_s", "time_per_token_s",
            "vram_used_MB", "flops_this", "flops_per_token", "cost_per_token_s"
        ]]

    def confusion_matrix(self) -> Optional[pd.DataFrame]:
        """
        혼동 행렬(optional)
        """
        # gt_answer, pred_answer가 숫자여도 정상적으로 동작 (자동 형변환 적용)
        try:
            return pd.crosstab(self.df["gt_answer"].astype(str), self.df["pred_answer"].astype(str))
        except Exception as e:
            print(f"Confusion matrix error: {e}")
            return None