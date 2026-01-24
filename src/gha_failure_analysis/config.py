import fnmatch
import logging
import os
from dataclasses import dataclass, field

from litellm import model_cost

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for the GitHub Actions failure analyzer."""

    # GitHub configuration (support both hyphen and underscore versions for local testing)
    github_token: str = field(
        default_factory=lambda: os.getenv("INPUT_GITHUB-TOKEN") or os.getenv("INPUT_GITHUB_TOKEN") or ""
    )
    repository: str = field(default_factory=lambda: os.getenv("GITHUB_REPOSITORY", ""))
    run_id: str = field(
        default_factory=lambda: os.getenv("INPUT_RUN-ID")
        or os.getenv("INPUT_RUN_ID")
        or os.getenv("GITHUB_RUN_ID")
        or ""
    )
    pr_number: str | None = field(
        default_factory=lambda: os.getenv("INPUT_PR-NUMBER") or os.getenv("INPUT_PR_NUMBER") or None
    )

    # LLM configuration (support both hyphen and underscore versions for local testing)
    llm_provider: str = field(
        default_factory=lambda: os.getenv("INPUT_LLM-PROVIDER") or os.getenv("INPUT_LLM_PROVIDER") or ""
    )
    llm_model: str = field(default_factory=lambda: os.getenv("INPUT_LLM-MODEL") or os.getenv("INPUT_LLM_MODEL") or "")
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("INPUT_LLM-API-KEY") or os.getenv("INPUT_LLM_API_KEY") or ""
    )
    llm_base_url: str | None = field(
        default_factory=lambda: os.getenv("INPUT_LLM-BASE-URL") or os.getenv("INPUT_LLM_BASE_URL") or None
    )

    # cordon configuration (support both hyphen and underscore versions for local testing)
    cordon_device: str = field(
        default_factory=lambda: os.getenv("INPUT_CORDON-DEVICE") or os.getenv("INPUT_CORDON_DEVICE") or "cpu"
    )
    cordon_backend: str = field(
        default_factory=lambda: os.getenv("INPUT_CORDON-BACKEND")
        or os.getenv("INPUT_CORDON_BACKEND")
        or "sentence-transformers"
    )
    cordon_model_name: str = field(
        default_factory=lambda: os.getenv("INPUT_CORDON-MODEL-NAME")
        or os.getenv("INPUT_CORDON_MODEL_NAME")
        or "all-MiniLM-L6-v2"
    )
    cordon_api_key: str | None = field(
        default_factory=lambda: os.getenv("INPUT_CORDON-API-KEY") or os.getenv("INPUT_CORDON_API_KEY") or None
    )
    cordon_endpoint: str | None = field(
        default_factory=lambda: os.getenv("INPUT_CORDON-ENDPOINT") or os.getenv("INPUT_CORDON_ENDPOINT") or None
    )
    cordon_batch_size: int = field(
        default_factory=lambda: int(
            os.getenv("INPUT_CORDON-BATCH-SIZE") or os.getenv("INPUT_CORDON_BATCH_SIZE") or "32"
        )
    )

    # GitHub Actions output configuration (support both hyphen and underscore versions for local testing)
    post_job_summary: bool = True
    post_pr_comment: bool = field(
        default_factory=lambda: (
            (os.getenv("INPUT_POST-PR-COMMENT") or os.getenv("INPUT_POST_PR_COMMENT") or "false").lower() == "true"
        )
    )

    # PR context analysis configuration (support both hyphen and underscore versions for local testing)
    analyze_pr_context: bool = field(
        default_factory=lambda: (
            (os.getenv("INPUT_ANALYZE-PR-CONTEXT") or os.getenv("INPUT_ANALYZE_PR_CONTEXT") or "true").lower() == "true"
        )
    )
    pr_context_token_budget_pct: int = field(
        default_factory=lambda: int(
            os.getenv("INPUT_PR-CONTEXT-TOKEN-BUDGET") or os.getenv("INPUT_PR_CONTEXT_TOKEN_BUDGET") or "20"
        )
    )

    # Filtering configuration (support both hyphen and underscore versions for local testing)
    ignored_jobs_patterns: list[str] = field(
        default_factory=lambda: (
            (os.getenv("INPUT_IGNORED-JOBS") or os.getenv("INPUT_IGNORED_JOBS") or "").split(",")
            if (os.getenv("INPUT_IGNORED-JOBS") or os.getenv("INPUT_IGNORED_JOBS"))
            else []
        )
    )
    ignored_steps_patterns: list[str] = field(
        default_factory=lambda: (
            (os.getenv("INPUT_IGNORED-STEPS") or os.getenv("INPUT_IGNORED_STEPS") or "").split(",")
            if (os.getenv("INPUT_IGNORED-STEPS") or os.getenv("INPUT_IGNORED_STEPS"))
            else []
        )
    )
    artifact_patterns: list[str] = field(
        default_factory=lambda: (
            (os.getenv("INPUT_ARTIFACT-PATTERNS") or os.getenv("INPUT_ARTIFACT_PATTERNS") or "").split(",")
            if (os.getenv("INPUT_ARTIFACT-PATTERNS") or os.getenv("INPUT_ARTIFACT_PATTERNS"))
            else []
        )
    )

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.github_token:
            errors.append("GITHUB_TOKEN is required")
        if not self.repository:
            errors.append("GITHUB_REPOSITORY is required")
        if not self.run_id:
            errors.append("RUN_ID is required")
        if not self.llm_provider:
            errors.append("LLM_PROVIDER is required")
        if not self.llm_model:
            errors.append("LLM_MODEL is required")
        if not self.llm_api_key:
            errors.append("LLM_API_KEY is required")

        return errors

    def detect_model_context_limit(self) -> int:
        """Query model's context window from LiteLLM database."""
        try:
            full_model = f"{self.llm_provider}/{self.llm_model}"

            if full_model in model_cost:
                max_input: int | None = model_cost[full_model].get("max_input_tokens")
                if max_input:
                    logger.info(f"Detected context for {full_model}: {max_input:,} tokens")
                    return max_input

            if self.llm_model in model_cost:
                max_input = model_cost[self.llm_model].get("max_input_tokens")
                if max_input:
                    logger.info(f"Detected context for {self.llm_model}: {max_input:,} tokens")
                    return max_input

            for model_key in model_cost.keys():
                if self.llm_model in model_key or model_key in self.llm_model:
                    max_input = model_cost[model_key].get("max_input_tokens")
                    if max_input:
                        logger.info(f"Detected context for {model_key}: {max_input:,} tokens")
                        return max_input

            logger.warning(f"Model {self.llm_model} not in database, using 128K default")
            return 128000

        except Exception as e:
            logger.warning(f"Error querying context limit: {e}, using 128K default")
            return 128000

    def calculate_token_budgets(
        self, num_failed_steps: int, num_failed_tests: int, num_artifacts: int
    ) -> tuple[int, int, int]:
        """Calculate dynamic token budgets based on number of failures.

        Returns:
            Tuple of (tokens_per_step, tokens_per_test, tokens_per_artifact_batch)
        """
        context_limit = self.detect_model_context_limit()
        available = context_limit - int(context_limit * 0.15)

        if num_failed_steps == 0 and num_failed_tests == 0 and num_artifacts == 0:
            return (int(context_limit * 0.20), int(context_limit * 0.08), int(context_limit * 0.08))

        # Weight: steps=2x, tests=1x, artifacts=1x total
        total_units = (num_failed_steps * 2) + num_failed_tests + 1
        tokens_per_unit = available // total_units

        tokens_per_step = max(10_000, min(200_000, tokens_per_unit * 2))
        tokens_per_test = max(10_000, min(80_000, tokens_per_unit))
        tokens_per_artifact_batch = max(20_000, min(150_000, tokens_per_unit))

        return (tokens_per_step, tokens_per_test, tokens_per_artifact_batch)

    def should_ignore_job(self, job_name: str) -> bool:
        """Check if job matches any ignore pattern."""
        return any(
            fnmatch.fnmatch(job_name, pattern.strip()) for pattern in self.ignored_jobs_patterns if pattern.strip()
        )

    def should_ignore_step(self, step_name: str) -> bool:
        """Check if step matches any ignore pattern."""
        return any(
            fnmatch.fnmatch(step_name, pattern.strip()) for pattern in self.ignored_steps_patterns if pattern.strip()
        )

    def should_include_artifact_path(self, artifact_path: str) -> bool:
        """Check if artifact path matches any include pattern."""
        return any(
            fnmatch.fnmatch(artifact_path, pattern.strip()) for pattern in self.artifact_patterns if pattern.strip()
        )
