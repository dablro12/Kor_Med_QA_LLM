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
        """
        # 일반적으로 gt_answer, pred_answer가 문자(예: 'C')로 같으면 정답
        self.df["is_correct"] = self.df["gt_answer"] == self.df["pred_answer"]

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
        # 분류값이 문자형으로 잘 들어있다면 정상적으로 ct가 작동함
        try:
            return pd.crosstab(self.df["gt_answer"], self.df["pred_answer"])
        except Exception as e:
            print(f"Confusion matrix error: {e}")
            return None
# End: 검토 코멘트
