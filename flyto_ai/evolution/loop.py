# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Evolution loop — the main orchestrator for prompt optimization.

Flow:
  1. Load eval cases + baseline prompt
  2. For each generation:
     a. Generate N candidate prompts (mutation/crossover)
     b. Round A: Quick screen (cheap model, subset)
     c. Round B: Full eval (target model, full set)
     d. Round C: Adversarial test (edge cases)
     e. Select top candidates + 1 elite + 1 novelty
     f. Check for regressions
  3. Output: best prompt + report

Safe by design:
  - Never auto-applies changes to production prompt
  - All results archived with diffs
  - Regression detection before recommending changes
"""
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from flyto_ai.evolution.blocks import get_baseline_candidate
from flyto_ai.evolution.models import (
    CandidateScore, EvolutionConfig, EvolutionReport,
    GenerationResult, PromptCandidate,
)
from flyto_ai.evolution.mutator import PromptMutator
from flyto_ai.evolution.runner import eval_candidate, format_score_report, load_eval_cases

logger = logging.getLogger(__name__)


class EvolutionLoop:
    """Orchestrates the prompt evolution process.

    Usage:
        loop = EvolutionLoop(config)
        report = await loop.run()
        print(report.best_candidate_id, report.best_score)
    """

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        provider=None,
        eval_cases_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        self._config = config or EvolutionConfig()
        self._provider = provider
        self._cases = load_eval_cases(eval_cases_path)
        self._mutator = PromptMutator(self._config)
        self._output_dir = Path(output_dir) if output_dir else Path("eval/results")
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Separate cases by category for multi-round eval
        self._main_cases = [c for c in self._cases if c.category != "adversarial"]
        self._adversarial_cases = [c for c in self._cases if c.category == "adversarial"]

    async def run(self, on_progress=None) -> EvolutionReport:
        """Run the full evolution loop.

        Parameters
        ----------
        on_progress : callable, optional
            Called with (generation, candidate_id, score) for progress tracking.

        Returns
        -------
        EvolutionReport
            Full report with all generations, scores, and best prompt.
        """
        t0 = time.time()
        report = EvolutionReport(config=self._config)

        # Start with baseline
        baseline = get_baseline_candidate(self._config.mode)
        parents = [baseline]

        best_score = 0.0
        no_improve_count = 0
        total_evals = 0

        for gen in range(self._config.generations):
            logger.info("=== Generation %d ===", gen)
            gen_result = GenerationResult(generation=gen)

            # Generate candidates
            candidates = self._mutator.generate_candidates(
                parents, generation=gen,
                count=self._config.population_size,
            )

            # --- Round A: Quick screen (subset) ---
            logger.info("Round A: Quick screening (%d cases)", self._config.quick_screen_subset)
            round_a_scores = []
            for cand in candidates:
                score = await eval_candidate(
                    cand, self._main_cases, self._config,
                    provider=self._provider,
                    subset=self._config.quick_screen_subset,
                )
                round_a_scores.append((cand, score))
                total_evals += score.eval_count
                if on_progress:
                    on_progress(gen, cand.id, score.weighted_total)

            # Sort by score, keep top 60%
            round_a_scores.sort(key=lambda x: x[1].weighted_total, reverse=True)
            cutoff = max(2, int(len(round_a_scores) * 0.6))
            survivors = round_a_scores[:cutoff]

            # --- Round B: Full eval (all main cases) ---
            logger.info("Round B: Full evaluation (%d candidates, %d cases)",
                        len(survivors), len(self._main_cases))
            round_b_scores = []
            for cand, _ in survivors:
                score = await eval_candidate(
                    cand, self._main_cases, self._config,
                    provider=self._provider,
                )
                round_b_scores.append((cand, score))
                total_evals += score.eval_count

            # --- Round C: Adversarial test (if available) ---
            if self._adversarial_cases:
                logger.info("Round C: Adversarial test (%d cases)", len(self._adversarial_cases))
                for i, (cand, main_score) in enumerate(round_b_scores):
                    adv_score = await eval_candidate(
                        cand, self._adversarial_cases, self._config,
                        provider=self._provider,
                    )
                    total_evals += adv_score.eval_count
                    # Combine: 70% main + 30% adversarial
                    combined_total = (
                        0.7 * main_score.weighted_total + 0.3 * adv_score.weighted_total
                    )
                    main_score.weighted_total = round(combined_total, 2)
                    main_score.stability = round(
                        min(main_score.stability, adv_score.stability), 2,
                    )

            # Sort by final score
            round_b_scores.sort(key=lambda x: x[1].weighted_total, reverse=True)

            # Record results
            for cand, score in round_b_scores:
                gen_result.candidates.append(score)
                logger.info(format_score_report(cand, score))

            # Best of this generation
            if round_b_scores:
                best_cand, best_gen_score = round_b_scores[0]
                gen_result.best_id = best_cand.id
                gen_result.best_score = best_gen_score.weighted_total
                gen_result.improvement = best_gen_score.weighted_total - best_score

                # Regression detection
                gen_result.regressions = self._detect_regressions(
                    round_b_scores, gen,
                )

                # Check improvement
                if best_gen_score.weighted_total > best_score:
                    best_score = best_gen_score.weighted_total
                    report.best_candidate_id = best_cand.id
                    report.best_score = best_score
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Select parents for next generation
                parents = [cand for cand, _ in round_b_scores[:3]]

            report.generations.append(gen_result)

            # Save intermediate results
            self._save_generation(gen, candidates, round_b_scores)

            # Stop criteria
            if no_improve_count >= self._config.stop_no_improve:
                logger.info("Stopping: no improvement for %d generations", no_improve_count)
                break
            if best_score >= self._config.target_score:
                logger.info("Stopping: target score %.1f reached", self._config.target_score)
                break

        report.total_eval_runs = total_evals
        report.total_duration_s = round(time.time() - t0, 1)

        # Collect all regression cases
        for gen_result in report.generations:
            report.regression_cases.extend(gen_result.regressions)

        # Save final report
        self._save_report(report)

        return report

    def _detect_regressions(
        self,
        scored_candidates: List[tuple],
        generation: int,
    ) -> List[str]:
        """Detect cases where the best candidate scored worse than baseline."""
        if not scored_candidates or generation == 0:
            return []

        # Compare best candidate's per-case scores against baseline
        best_cand, best_score = scored_candidates[0]
        baseline_cand = None
        baseline_score = None
        for cand, score in scored_candidates:
            if cand.mutation_type == "baseline" or "elite" in cand.id:
                baseline_cand = cand
                baseline_score = score
                break

        if not baseline_score:
            return []

        regressions = []
        baseline_by_case = {s.case_id: s for s in baseline_score.scores}
        for s in best_score.scores:
            baseline_s = baseline_by_case.get(s.case_id)
            if baseline_s and s.total_score < baseline_s.total_score - 5:
                regressions.append(s.case_id)

        return regressions

    def _save_generation(
        self,
        generation: int,
        candidates: List[PromptCandidate],
        scored: List[tuple],
    ):
        """Save generation results to disk."""
        gen_dir = self._output_dir / "gen_{}".format(generation)
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Save candidates
        for cand in candidates:
            cand_file = gen_dir / "{}.json".format(cand.id)
            with open(cand_file, "w", encoding="utf-8") as f:
                json.dump(cand.model_dump(), f, ensure_ascii=False, indent=2)

        # Save scores
        scores_data = []
        for cand, score in scored:
            scores_data.append({
                "candidate_id": cand.id,
                "mutation_type": cand.mutation_type,
                "mutation_desc": cand.mutation_desc,
                "weighted_total": score.weighted_total,
                "task_avg": score.task_avg,
                "compliance_avg": score.compliance_avg,
                "ux_avg": score.ux_avg,
                "penalty_total": score.penalty_total,
                "pass_rate": score.pass_rate,
                "stability": score.stability,
            })
        scores_file = gen_dir / "scores.json"
        with open(scores_file, "w", encoding="utf-8") as f:
            json.dump(scores_data, f, ensure_ascii=False, indent=2)

        # Save best prompt (rendered)
        if scored:
            best_cand = scored[0][0]
            prompt_file = gen_dir / "best_prompt.txt"
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(best_cand.render(module_count=self._config.module_count))

    def _save_report(self, report: EvolutionReport):
        """Save the final evolution report."""
        report_file = self._output_dir / "report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, ensure_ascii=False, indent=2)
        logger.info("Report saved to %s", report_file)


def format_evolution_report(report: EvolutionReport) -> str:
    """Format a human-readable evolution report."""
    lines = [
        "=" * 60,
        "Prompt Evolution Report",
        "=" * 60,
        "Generations: {}  Total evals: {}  Duration: {:.0f}s".format(
            len(report.generations), report.total_eval_runs, report.total_duration_s,
        ),
        "Best candidate: {}  Score: {:.1f}/100".format(
            report.best_candidate_id, report.best_score,
        ),
        "",
    ]

    for gen_result in report.generations:
        lines.append("--- Generation {} ---".format(gen_result.generation))
        lines.append("Best: {} ({:.1f})  Improvement: {:+.1f}".format(
            gen_result.best_id, gen_result.best_score, gen_result.improvement,
        ))
        if gen_result.regressions:
            lines.append("Regressions: {}".format(", ".join(gen_result.regressions)))
        for cs in gen_result.candidates[:3]:
            lines.append("  {} — {:.1f}/100 (pass {:.0%})".format(
                cs.candidate_id, cs.weighted_total, cs.pass_rate,
            ))
        lines.append("")

    if report.regression_cases:
        lines.append("Total regression cases: {}".format(
            ", ".join(set(report.regression_cases)),
        ))

    return "\n".join(lines)
