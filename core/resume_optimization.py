import json
import re
from collections import Counter
from typing import Any, Optional

from pydantic import BaseModel

from http_compat import HTTPException

from audit_logic import _audit_fallback, audit_resume_core, extract_jd_keywords, normalize_audit_result
from prompts import OPTIMIZE_SYSTEM_PROMPT, OPTIMIZE_WITH_AUDIT_SYSTEM_PROMPT, REVISION_SYSTEM_PROMPT
from resume_common import (
    _build_audit_issue_index,
    _build_change_reason,
    _clone_json,
    _collect_substantive_changes,
    _contains_any,
    _derive_non_bullet_changes,
    _derive_structured_changes,
    _extract_tech_anchor_tokens,
    _get_bullet_by_path,
    _has_metric,
    _is_trivial_wording_change,
    _list_bullet_locations,
    _looks_over_generic,
    _normalize_compare_text,
    _normalize_for_semantic_compare,
    _normalize_text_list,
    _remove_ai_phrases,
    _resolve_revision_target,
    _revert_trivial_rewrites,
    _set_bullet_by_path,
    chain_of_verification,
)
from resume_parsing import (
    _log_parse_text_debug,
    _log_resume_data_debug,
    _should_guard_resume_shrink,
    resume_data_to_text,
)
from schemas import OptimizeLLMOutput, OptimizeWithAuditLLMOutput, RevisionLLMOutput, RevisionTarget
from server_runtime import DETAIL_HINT_WORDS, RESPONSIBILITY_WORDS, SHRINK_GUARD_MIN_SOURCE_CHARS, call_llm_text, call_llm_typed, llm_enabled, logger, sanitize_user_text
from llm_gateway import parse_json_content


def _normalize_style(style: str) -> str:
    return style if style in {"conservative", "aggressive"} else "aggressive"


def _needs_forced_deep_rewrite(audit_result: dict[str, Any], changes: list[dict[str, Any]], style: str) -> bool:
    issues = audit_result.get("issues", []) if isinstance(audit_result, dict) else []
    if not isinstance(issues, list):
        issues = []
    issue_count = len(issues)
    high_count = sum(1 for item in issues if isinstance(item, dict) and str(item.get("severity") or "").lower() == "high")
    medium_count = sum(1 for item in issues if isinstance(item, dict) and str(item.get("severity") or "").lower() == "medium")
    substantive_changes = _collect_substantive_changes(changes)
    # In aggressive mode: always allow deep rewrite when issues are under-fixed
    if _normalize_style(style) == "aggressive":
        if high_count >= 1 and len(substantive_changes) < 2:
            return True
        if issue_count >= 3 and len(substantive_changes) < 2:
            return True
        # 2+ medium severity issues with insufficient fixes also triggers deep rewrite
        if medium_count >= 2 and len(substantive_changes) < 2:
            return True
        return False
    # In conservative mode: still allow deep rewrite for high-severity issues
    if high_count >= 2 and len(substantive_changes) < 1:
        return True
    return False


def _build_optimize_prompt(
    *,
    resume_data: dict[str, Any],
    audit_result: dict[str, Any],
    jd_profile: dict[str, Any],
    safe_jd: str,
    style: str,
    force_deep_rewrite: bool = False,
    source_text: Optional[str] = None,
) -> str:
    deep_rewrite_block = ""
    if _normalize_style(style) == "aggressive" or force_deep_rewrite:
        deep_rewrite_block = (
            "\u3010\u9996\u8f6e\u6df1\u6539\u8981\u6c42\u3011\n"
            "- \u9ed8\u8ba4\u5047\u8bbe\u8f93\u5165\u7b80\u5386\u8868\u8fbe\u504f\u5f31\uff0c\u5fc5\u987b\u4e3b\u52a8\u505a\u5b9e\u8d28\u6027\u91cd\u5199\uff0c\u800c\u4e0d\u662f\u53ea\u505a\u8bed\u75c5/\u8bed\u5e8f\u8c03\u6574\n"
            "- \u4f18\u5148\u91cd\u5199\u6700\u5f31\u7684 2-4 \u4e2a\u9879\u76ee\u8981\u70b9\u3001\u6458\u8981\u3001\u6807\u9898\u6216\u6280\u80fd\u5206\u7ec4\uff0c\u4f7f\u8868\u8fbe\u66f4\u5177\u4f53\u3001\u66f4\u53ef\u8ffd\u95ee\u3001\u66f4\u8d34\u8fd1 JD\n"
            "- \u82e5\u539f\u59cb bullet \u53ea\u6709\u7ed3\u679c\u6ca1\u6709\u8fc7\u7a0b\uff0c\u5fc5\u987b\u8865\u6210\u201c\u80cc\u666f/\u76ee\u6807 + \u65b9\u6848/\u5b9e\u73b0 + \u7ed3\u679c/\u9a8c\u8bc1\u201d\u7ed3\u6784\uff0c\u4f46\u4e0d\u80fd\u65b0\u589e\u4e8b\u5b9e\n"
            "- \u5982\u679c\u5ba1\u8ba1\u95ee\u9898\u8f83\u591a\uff0c\u4e0d\u5141\u8bb8\u53ea\u8fd4\u56de\u51e0\u5904\u5fae\u5c0f\u4fee\u6539\n\n"
        )
    source_context = ""
    if source_text and len(source_text) > 100:
        source_context = "\u3010\u539f\u59cb\u7b80\u5386\u7d20\u6750\uff08\u53ef\u53c2\u8003\u539f\u6587\u7ec6\u8282\uff0c\u4e0d\u5f97\u65b0\u589e\u539f\u6587\u6ca1\u6709\u7684\u4e8b\u5b9e\uff09\u3011\n"
        for exp in resume_data.get("experience", [])[:6]:
            if not isinstance(exp, dict):
                continue
            company = str(exp.get("company", "") or "").strip()
            if company and len(company) >= 3:
                idx = source_text.lower().find(company.lower())
                if idx < 0:
                    idx = source_text.find(company)
                if idx >= 0:
                    start = max(0, idx - 50)
                    end = min(len(source_text), idx + len(company) + 300)
                    snippet = source_text[start:end].strip()
                    snippet = re.sub(r"\s+", " ", snippet)[:400]
                    source_context += f"\n[{company} \u539f\u6587\u4e0a\u4e0b\u6587]\n{snippet}\n"
        source_context += "\n"

    return (
        "\u4f60\u73b0\u5728\u6267\u884c\u4e09\u9636\u6bb5\u4f18\u5316\u4e2d\u7684\u201c\u5b9a\u5411\u6539\u5199\u9636\u6bb5\u201d\uff1a\n"
        "\u76ee\u6807\u662f\u57fa\u4e8e JD \u753b\u50cf + \u5ba1\u8ba1\u95ee\u9898\u8fdb\u884c\u4e8b\u5b9e\u4fdd\u771f\u7684\u9ad8\u8d28\u91cf\u6539\u5199\u3002\n\n"
        "\u8bf7\u57fa\u4e8e\u5ba1\u8ba1\u95ee\u9898\u5bf9\u7b80\u5386\u8fdb\u884c\u9ad8\u8d28\u91cf\u4f18\u5316\uff0c\u4f18\u5148\u4fee\u590d high/medium \u98ce\u9669\u9879\u3002\n"
        "\u53ea\u8f93\u51fa JSON\uff0c\u4e14\u5fc5\u987b\u5305\u542b optimized_resume, changes\u3002\n\n"
        f"{deep_rewrite_block}"
        f"{source_context}"
        "\u3010JD \u753b\u50cf\u3011\n"
        f"{json.dumps(jd_profile, ensure_ascii=False)}\n\n"
        "\u3010\u5f53\u524d\u7b80\u5386 JSON\u3011\n"
        f"{json.dumps(resume_data, ensure_ascii=False)}\n\n"
        "\u3010\u5ba1\u8ba1\u7ed3\u679c JSON\u3011\n"
        f"{json.dumps(audit_result, ensure_ascii=False)}\n\n"
        "\u3010\u76ee\u6807 JD\uff08\u53ef\u9009\uff09\u3011\n"
        f"{safe_jd}\n\n"
        "\u3010\u786c\u7ea6\u675f\u3011\n"
        "- \u4e0d\u5f97\u634f\u9020\u7ecf\u5386/\u6280\u80fd/\u6570\u5b57\n"
        "- \u4e0d\u5f97\u66f4\u6539\u65f6\u95f4\u7ebf\u548c\u516c\u53f8\u4fe1\u606f\n"
        "- \u4fdd\u7559\u5173\u952e\u6280\u672f\u951a\u70b9\uff08\u6a21\u578b/\u7b97\u6cd5/\u6570\u636e\u96c6/\u8bba\u6587\u540d/\u7f29\u5199\uff09\n"
        "- \u4e0d\u8981\u628a\u5177\u4f53\u6280\u672f\u8868\u8ff0\u6539\u6210\u6cdb\u5316\u7a7a\u8bdd\n"
        "- changes \u5fc5\u987b\u9010\u6761\u5bf9\u5e94\u771f\u5b9e\u6539\u52a8\uff0creason \u8bf4\u660e\u4fee\u590d\u4e86\u4ec0\u4e48\u98ce\u9669"
    )

def revise_resume_data_with_llm(
    resume_data: dict[str, Any],
    revision_instructions: str,
    jd_text: Optional[str] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    safe_revision = sanitize_user_text(revision_instructions)
    safe_jd = sanitize_user_text(jd_text or "")
    prompt = (
        "请按用户反馈对简历 JSON 做增量修改。\n"
        "只改有明确指令的部分；不要重写整份简历。\n"
        "必须输出字段：resume_data, changes。\n\n"
        "【用户反馈】\n"
        f"{safe_revision}\n\n"
        "【目标 JD（可选）】\n"
        f"{safe_jd}\n\n"
        "【当前简历 JSON】\n"
        f"{json.dumps(resume_data, ensure_ascii=False)}\n\n"
        "【返回约束】\n"
        "- changes 每项必须包含 location, before, after, reason\n"
        "- before/after 要尽量具体到被修改文本\n"
        "- 如果用户只提格式要求，不要改动事实字段"
    )
    parsed = call_llm_typed(RevisionLLMOutput, REVISION_SYSTEM_PROMPT, prompt, temperature=0.2)
    new_resume = parsed.get("resume_data")
    changes = parsed.get("changes")
    if not isinstance(new_resume, dict):
        logger.warning("Revision LLM returned invalid resume_data; fallback to original resume_data")
        return _clone_json(resume_data), []
    if not isinstance(changes, list):
        changes = []
    return new_resume, changes


def revise_resume_data(
    resume_data: dict[str, Any],
    revision_instructions: str,
    jd_text: Optional[str] = None,
    revision_targets: Optional[list[RevisionTarget]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if revision_targets:
        return revise_resume_data_by_targets(
            resume_data=resume_data,
            revision_instructions=revision_instructions,
            revision_targets=revision_targets,
            jd_text=jd_text,
        )

    if not llm_enabled():
        raise HTTPException(status_code=500, detail="LLM is required for resume revision but is not configured")

    try:
        return revise_resume_data_with_llm(resume_data, revision_instructions, jd_text)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM revision failed: {exc}") from exc


def revise_resume_data_by_targets(
    resume_data: dict[str, Any],
    revision_instructions: str,
    revision_targets: list[RevisionTarget],
    jd_text: Optional[str] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not revision_targets:
        raise HTTPException(status_code=400, detail="revision_targets cannot be empty")

    source_resume = _clone_json(resume_data)
    llm_resume = source_resume
    if llm_enabled():
        try:
            llm_resume, _ = revise_resume_data_with_llm(source_resume, revision_instructions, jd_text)
            if not isinstance(llm_resume, dict):
                llm_resume = source_resume
        except Exception:
            logger.warning("Targeted LLM revision failed, fallback to local rewrite", exc_info=True)
            llm_resume = source_resume

    result_resume = _clone_json(source_resume)
    changes: list[dict[str, Any]] = []

    for idx, target in enumerate(revision_targets, start=1):
        found = _resolve_revision_target(source_resume, target)
        before = found["text"]

        if target.expected_before is not None:
            expected = _normalize_compare_text(target.expected_before)
            actual = _normalize_compare_text(before)
            if expected != actual:
                raise HTTPException(
                    status_code=409,
                    detail=f"revision target before-text mismatch at target #{idx}",
                )

        after = _get_bullet_by_path(
            llm_resume,
            found["exp_index"],
            found["project_index"],
            found["bullet_index"],
        ) or before

        # If LLM did not change the bullet, skip (no heuristic fallback)
        if _normalize_compare_text(after) == _normalize_compare_text(before):
            continue

        _set_bullet_by_path(
            result_resume,
            found["exp_index"],
            found["project_index"],
            found["bullet_index"],
            after,
        )

        if _normalize_compare_text(after) != _normalize_compare_text(before):
            changes.append(
                {
                    "location": f"experience[{found['exp_index']}].projects[{found['project_index']}].bullets[{found['bullet_index']}]",
                    "before": before,
                    "after": after,
                    "reason": f"按用户反馈定向改写：{revision_instructions[:120]}",
                }
            )

    if not changes:
        logger.info(
            "Targeted revision produced no diff; returning original resume without error. targets=%s",
            len(revision_targets),
        )
        return result_resume, []

    # Post-processing guards: AI phrase cleanup → anchor fidelity → shrink guard
    result_resume, ai_notes = _apply_revision_ai_phrase_cleanup(result_resume)
    result_resume, anchor_notes = _enforce_revision_anchor_fidelity(result_resume, source_resume)
    should_revert, revert_reason = _should_guard_resume_shrink(source_resume, result_resume)
    if should_revert:
        logger.warning("Shrink guard reverted revision | reason=%s", revert_reason)
        return _clone_json(source_resume), []

    # Update change descriptions if guards modified the bullets
    final_changes: list[dict[str, Any]] = []
    for change in changes:
        after = change.get("after", "")
        location = change.get("location", "")
        # Try to read the final bullet value
        try:
            parts = re.findall(r"\[(\d+)\]", location)
            if len(parts) == 3:
                exp_i, proj_i, bul_i = int(parts[0]), int(parts[1]), int(parts[2])
                final_after = _get_bullet_by_path(result_resume, exp_i, proj_i, bul_i)
                if final_after is not None:
                    after = final_after
        except Exception:
            pass
        if _normalize_compare_text(change.get("before", "")) != _normalize_compare_text(after):
            final_changes.append({
                "location": location,
                "before": change.get("before", ""),
                "after": after,
                "reason": change.get("reason", ""),
            })

    return result_resume, final_changes


def _apply_revision_ai_phrase_cleanup(resume_data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Remove AI jargon from all bullets in resume (lightweight guard for revise path)."""
    result = _clone_json(resume_data)
    notes: list[str] = []
    for exp in result.get("experience", []) if isinstance(result.get("experience"), list) else []:
        if not isinstance(exp, dict):
            continue
        for proj in exp.get("projects", []) if isinstance(exp.get("projects"), list) else []:
            if not isinstance(proj, dict):
                continue
            bullets = proj.get("bullets", [])
            if not isinstance(bullets, list):
                continue
            new_bullets = []
            for bullet in bullets:
                cleaned, changed = _remove_ai_phrases(str(bullet))
                new_bullets.append(cleaned)
                if changed:
                    notes.append("已清理 AI 套话表达")
            proj["bullets"] = new_bullets
    return result, notes


def _enforce_revision_anchor_fidelity(
    resume_data: dict[str, Any],
    source_resume_data: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Lightweight anchor fidelity check for revise path: revert bullets that lost key tech anchors."""
    result = _clone_json(resume_data)
    notes: list[str] = []

    source_exps = source_resume_data.get("experience", []) if isinstance(source_resume_data.get("experience"), list) else []
    target_exps = result.get("experience", []) if isinstance(result.get("experience"), list) else []

    for exp_idx, exp in enumerate(target_exps):
        if not isinstance(exp, dict):
            continue
        src_exp = source_exps[exp_idx] if exp_idx < len(source_exps) and isinstance(source_exps[exp_idx], dict) else {}
        src_projects = src_exp.get("projects", []) if isinstance(src_exp.get("projects"), list) else []
        projects = exp.get("projects", [])
        if not isinstance(projects, list):
            continue

        for proj_idx, proj in enumerate(projects):
            if not isinstance(proj, dict):
                continue
            src_proj = src_projects[proj_idx] if proj_idx < len(src_projects) and isinstance(src_projects[proj_idx], dict) else {}
            src_bullets = src_proj.get("bullets", []) if isinstance(src_proj.get("bullets"), list) else []
            bullets = proj.get("bullets", [])
            if not isinstance(bullets, list):
                continue

            new_bullets: list[str] = []
            for bullet_idx, current_bullet in enumerate(bullets):
                current_text = str(current_bullet)
                source_text = str(src_bullets[bullet_idx]) if bullet_idx < len(src_bullets) else ""
                if not source_text or _normalize_compare_text(current_text) == _normalize_compare_text(source_text):
                    new_bullets.append(current_text)
                    continue
                source_anchors = _extract_tech_anchor_tokens(source_text)
                current_anchors = _extract_tech_anchor_tokens(current_text)
                lost_anchors = source_anchors - current_anchors
                if lost_anchors and _looks_over_generic(current_text) and not _has_metric(current_text):
                    new_bullets.append(source_text)
                    notes.append("修订结果丢失技术锚点且回退为空泛表述，已恢复原文")
                else:
                    new_bullets.append(current_text)
            proj["bullets"] = new_bullets

    return result, notes


def _collect_source_constraints(source_resume_data: dict[str, Any]) -> dict[str, Any]:
    allowed_companies: set[str] = set()
    allowed_skills: set[str] = set()
    allowed_tech: set[str] = set()

    for exp in source_resume_data.get("experience", []) if isinstance(source_resume_data.get("experience"), list) else []:
        if not isinstance(exp, dict):
            continue
        company = str(exp.get("company", "")).strip()
        if company:
            allowed_companies.add(company.lower())
        for proj in exp.get("projects", []) if isinstance(exp.get("projects"), list) else []:
            if not isinstance(proj, dict):
                continue
            for tech in _normalize_text_list(proj.get("tech_stack")):
                allowed_tech.add(tech.lower())

    skills = source_resume_data.get("skills", {})
    if isinstance(skills, dict):
        for bucket in ("languages", "frameworks", "tools", "domains"):
            for skill in _normalize_text_list(skills.get(bucket)):
                allowed_skills.add(skill.lower())

    return {
        "allowed_companies": allowed_companies,
        "allowed_skills": allowed_skills,
        "allowed_tech": allowed_tech,
    }


def _apply_alignment_refinement(
    resume_data: dict[str, Any],
    source_resume_data: dict[str, Any],
    jd_text: Optional[str],
) -> tuple[dict[str, Any], list[str]]:
    refined = _clone_json(resume_data)
    notes: list[str] = []

    source_meta = source_resume_data.get("meta")
    if isinstance(source_meta, dict):
        refined["meta"] = _clone_json(source_meta)

    constraints = _collect_source_constraints(source_resume_data)
    allowed_companies = constraints["allowed_companies"]
    allowed_skills = constraints["allowed_skills"]
    allowed_tech = constraints["allowed_tech"]

    # Remove invented company names and AI phrases in bullets.
    source_experience = (
        source_resume_data.get("experience", [])
        if isinstance(source_resume_data.get("experience"), list)
        else []
    )
    experience = refined.get("experience", [])
    if isinstance(experience, list):
        for idx, exp in enumerate(experience):
            if not isinstance(exp, dict):
                continue

            company = str(exp.get("company", "")).strip().lower()
            if company and allowed_companies and company not in allowed_companies:
                if idx < len(source_experience) and isinstance(source_experience[idx], dict):
                    exp["company"] = source_experience[idx].get("company", exp.get("company", ""))
                else:
                    exp["company"] = "未提供"
                notes.append("已移除与原始简历不一致的公司信息")

            projects = exp.get("projects", [])
            if not isinstance(projects, list):
                continue
            for proj in projects:
                if not isinstance(proj, dict):
                    continue

                tech_stack = _normalize_text_list(proj.get("tech_stack"))
                if tech_stack and allowed_tech:
                    filtered = [t for t in tech_stack if t.lower() in allowed_tech]
                    if filtered != tech_stack:
                        proj["tech_stack"] = filtered
                        notes.append("已过滤未在原始简历出现的技术栈")

                bullets = proj.get("bullets", [])
                if isinstance(bullets, list):
                    new_bullets = []
                    for bullet in bullets:
                        bullet_text = str(bullet)
                        cleaned, changed = _remove_ai_phrases(bullet_text)
                        new_bullets.append(cleaned)
                        if changed:
                            notes.append("已清理 AI 套话表达")
                    proj["bullets"] = new_bullets

    # Remove invented skills.
    skills = refined.get("skills", {})
    if isinstance(skills, dict) and allowed_skills:
        for bucket in ("languages", "frameworks", "tools", "domains"):
            items = _normalize_text_list(skills.get(bucket))
            if not items:
                continue
            filtered = [item for item in items if item.lower() in allowed_skills]
            if filtered != items:
                skills[bucket] = filtered
                notes.append(f"已过滤 {bucket} 中未在原始简历出现的技能")

    # Safe JD keyword injection: only inject keywords that already exist in source text.
    if jd_text and jd_text.strip():
        source_text = resume_data_to_text(source_resume_data).lower()
        current_text = resume_data_to_text(refined).lower()
        jd_keywords = extract_jd_keywords(jd_text)
        injectable = [kw for kw in jd_keywords if kw.lower() in source_text and kw.lower() not in current_text]

        if injectable:
            inserted = False
            for exp in refined.get("experience", []) if isinstance(refined.get("experience"), list) else []:
                if not isinstance(exp, dict):
                    continue
                for proj in exp.get("projects", []) if isinstance(exp.get("projects"), list) else []:
                    if not isinstance(proj, dict):
                        continue
                    tech_stack = _normalize_text_list(proj.get("tech_stack"))
                    for kw in injectable:
                        if kw not in tech_stack and len(tech_stack) < 10:
                            tech_stack.append(kw)
                            inserted = True
                    proj["tech_stack"] = tech_stack
                    if inserted:
                        break
                if inserted:
                    break
            if inserted:
                notes.append("已按原始简历证据安全补齐部分 JD 关键词")

    return refined, notes


def _enforce_technical_anchor_fidelity(
    resume_data: dict[str, Any],
    source_resume_data: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Revert rewritten bullets that lose key technical anchors or become overly generic."""
    result = _clone_json(resume_data)
    notes: list[str] = []

    source_exps = source_resume_data.get("experience", []) if isinstance(source_resume_data.get("experience"), list) else []
    target_exps = result.get("experience", []) if isinstance(result.get("experience"), list) else []

    for exp_idx, exp in enumerate(target_exps):
        if not isinstance(exp, dict):
            continue
        src_exp = source_exps[exp_idx] if exp_idx < len(source_exps) and isinstance(source_exps[exp_idx], dict) else {}
        src_projects = src_exp.get("projects", []) if isinstance(src_exp.get("projects"), list) else []

        projects = exp.get("projects", [])
        if not isinstance(projects, list):
            continue

        for proj_idx, proj in enumerate(projects):
            if not isinstance(proj, dict):
                continue
            src_proj = src_projects[proj_idx] if proj_idx < len(src_projects) and isinstance(src_projects[proj_idx], dict) else {}
            src_bullets = src_proj.get("bullets", []) if isinstance(src_proj.get("bullets"), list) else []

            # Check tech_stack anchors
            src_tech_stack = src_proj.get("tech_stack", []) if isinstance(src_proj.get("tech_stack"), list) else []
            cur_tech_stack = proj.get("tech_stack", []) if isinstance(proj.get("tech_stack"), list) else []
            if src_tech_stack and not cur_tech_stack:
                proj["tech_stack"] = list(src_tech_stack)
                notes.append("已恢复被清空的项目 tech_stack")
            elif src_tech_stack and cur_tech_stack:
                src_ts_anchors = set()
                for ts in src_tech_stack:
                    src_ts_anchors.update(_extract_tech_anchor_tokens(str(ts)))
                cur_ts_anchors = set()
                for ts in cur_tech_stack:
                    cur_ts_anchors.update(_extract_tech_anchor_tokens(str(ts)))
                lost_ts = src_ts_anchors - cur_ts_anchors
                if lost_ts and len(lost_ts) >= len(src_ts_anchors) * 0.5:
                    proj["tech_stack"] = list(src_tech_stack)
                    notes.append("已恢复丢失过多技术锚点的 tech_stack")

            # Check project name for generic-ness
            src_proj_name = _normalize_compare_text(src_proj.get("name", ""))
            cur_proj_name = _normalize_compare_text(proj.get("name", ""))
            if src_proj_name and cur_proj_name and src_proj_name != cur_proj_name:
                src_name_anchors = _extract_tech_anchor_tokens(src_proj_name)
                cur_name_anchors = _extract_tech_anchor_tokens(cur_proj_name)
                if src_name_anchors and not cur_name_anchors:
                    proj["name"] = src_proj_name
                    notes.append("已恢复丢失技术锚点的项目标题")
                elif _looks_over_generic(cur_proj_name) and not _looks_over_generic(src_proj_name):
                    proj["name"] = src_proj_name
                    notes.append("已恢复过度空泛的项目标题")

            bullets = proj.get("bullets", [])
            if not isinstance(bullets, list):
                continue

            new_bullets: list[str] = []
            changed = False

            for bullet_idx, current_bullet in enumerate(bullets):
                current_text = str(current_bullet)
                source_text = str(src_bullets[bullet_idx]) if bullet_idx < len(src_bullets) else ""
                if not source_text:
                    new_bullets.append(current_text)
                    continue
                if _normalize_compare_text(current_text) == _normalize_compare_text(source_text):
                    new_bullets.append(current_text)
                    continue

                source_anchors = _extract_tech_anchor_tokens(source_text)
                kept_anchor_count = sum(1 for token in source_anchors if token in current_text.lower())

                should_revert_anchor_loss = bool(source_anchors) and kept_anchor_count == 0
                should_revert_generic = _looks_over_generic(current_text) and not _looks_over_generic(source_text)

                if should_revert_anchor_loss:
                    new_bullets.append(source_text)
                    changed = True
                    notes.append("已回退丢失关键技术锚点的改写")
                    continue

                if should_revert_generic:
                    new_bullets.append(source_text)
                    changed = True
                    notes.append("已回退过度空泛的改写")
                    continue

                new_bullets.append(current_text)

            if changed:
                proj["bullets"] = new_bullets

    # Also check top-level projects tech_stack
    src_top_projects = source_resume_data.get("projects", []) if isinstance(source_resume_data.get("projects"), list) else []
    cur_top_projects = result.get("projects", []) if isinstance(result.get("projects"), list) else []
    for proj_idx, proj in enumerate(cur_top_projects):
        if not isinstance(proj, dict):
            continue
        src_proj = src_top_projects[proj_idx] if proj_idx < len(src_top_projects) and isinstance(src_top_projects[proj_idx], dict) else {}
        src_ts = src_proj.get("tech_stack", []) if isinstance(src_proj.get("tech_stack"), list) else []
        cur_ts = proj.get("tech_stack", []) if isinstance(proj.get("tech_stack"), list) else []
        if src_ts and not cur_ts:
            proj["tech_stack"] = list(src_ts)
            notes.append("已恢复被清空的顶层项目 tech_stack")

    return result, notes


def optimize_resume_core(
    resume_data: dict[str, Any],
    audit_result: dict[str, Any],
    style: str,
    jd_text: Optional[str] = None,
    source_resume_data: Optional[dict[str, Any]] = None,
    source_text: Optional[str] = None,
) -> dict[str, Any]:
    if not llm_enabled():
        raise HTTPException(status_code=500, detail="LLM is required for resume optimization but is not configured")

    original_resume_data = _clone_json(resume_data)
    style = _normalize_style(style)
    quick_keywords = extract_jd_keywords(jd_text or "")
    jd_profile = {
        "job_family": "",
        "seniority": "",
        "must_have_keywords": quick_keywords[:12],
        "nice_to_have_keywords": quick_keywords[12:24],
        "core_responsibilities": [],
        "risk_notes": [],
    }

    safe_jd = sanitize_user_text(jd_text or "")
    llm_changes: list[dict[str, Any]] = []
    try:
        prompt = _build_optimize_prompt(
            resume_data=resume_data,
            audit_result=audit_result,
            jd_profile=jd_profile,
            safe_jd=safe_jd,
            style=style,
            source_text=source_text,
        )
        llm_out = call_llm_typed(
            OptimizeLLMOutput,
            OPTIMIZE_SYSTEM_PROMPT,
            prompt,
            temperature=0.3 if style == "aggressive" else 0.2,
            max_tokens=4096,
        )
        optimized = llm_out.get("optimized_resume")
        llm_changes = llm_out.get("changes", [])
        llm_recovery_warnings: list[str] = []
        if not isinstance(optimized, dict):
            logger.warning("Optimize LLM returned invalid optimized_resume; fallback to original resume_data")
            optimized = _clone_json(resume_data)
            llm_recovery_warnings.append("LLM 返回结果格式异常，已使用原始简历数据")
        if not isinstance(llm_changes, list):
            llm_changes = []
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM optimization failed: {exc}") from exc

    # Preserve standalone project section when optimizer accidentally drops it.
    source_top_projects = original_resume_data.get("projects", [])
    if (not isinstance(optimized.get("projects"), list) or not optimized.get("projects")) and isinstance(source_top_projects, list) and source_top_projects:
        optimized["projects"] = _clone_json(source_top_projects)

    resume_data = optimized
    master_data = source_resume_data if isinstance(source_resume_data, dict) else original_resume_data
    stage_notes: list[str] = []
    resume_data, refine_notes = _apply_alignment_refinement(resume_data, master_data, jd_text)
    if refine_notes:
        stage_notes.extend(sorted(set(refine_notes)))

    resume_data, fidelity_notes = _enforce_technical_anchor_fidelity(resume_data, master_data)
    if fidelity_notes:
        stage_notes.extend(sorted(set(fidelity_notes)))

    # Chain-of-Verification: detect and revert fabricated numbers
    resume_data, cov_notes = chain_of_verification(master_data, resume_data)
    if cov_notes:
        stage_notes.extend(sorted(set(cov_notes)))

    resume_data, trivial_notes = _revert_trivial_rewrites(resume_data, original_resume_data)
    if trivial_notes:
        stage_notes.extend(sorted(set(trivial_notes)))

    should_revert, revert_reason = _should_guard_resume_shrink(original_resume_data, resume_data)
    if should_revert:
        logger.warning("Shrink guard reverted optimized resume to source | reason=%s", revert_reason)
        resume_data = _clone_json(original_resume_data)
        stage_notes.append(f"已回滚异常缩水改写（{revert_reason}）")

    changes = _derive_structured_changes(original_resume_data, resume_data, audit_result)
    if not changes and llm_changes:
        changes = [item for item in llm_changes if isinstance(item, dict)]

    if _needs_forced_deep_rewrite(audit_result, changes, style):
        logger.info(
            "Force deep rewrite pass in optimize_resume_core | issues=%s substantive_changes=%s",
            len(audit_result.get("issues", [])) if isinstance(audit_result, dict) else 0,
            len(_collect_substantive_changes(changes)),
        )
        deep_out = call_llm_typed(
            OptimizeLLMOutput,
            OPTIMIZE_SYSTEM_PROMPT,
            _build_optimize_prompt(
                resume_data=original_resume_data,
                audit_result=audit_result,
                jd_profile=jd_profile,
                safe_jd=safe_jd,
                style="aggressive",
                force_deep_rewrite=True,
                source_text=source_text,
            ),
            temperature=0.35,
            max_tokens=4096,
        )
        deep_resume = deep_out.get("optimized_resume")
        deep_llm_changes = deep_out.get("changes", [])
        if isinstance(deep_resume, dict):
            deep_resume, _ = _apply_alignment_refinement(deep_resume, master_data, jd_text)
            deep_resume, _ = _enforce_technical_anchor_fidelity(deep_resume, master_data)
            deep_resume, _ = chain_of_verification(master_data, deep_resume)
            deep_resume, _ = _revert_trivial_rewrites(deep_resume, original_resume_data)
            deep_should_revert, _ = _should_guard_resume_shrink(original_resume_data, deep_resume)
            if not deep_should_revert:
                deep_changes = _derive_structured_changes(original_resume_data, deep_resume, audit_result)
                if not deep_changes and isinstance(deep_llm_changes, list):
                    deep_changes = [item for item in deep_llm_changes if isinstance(item, dict)]
                deep_substantive = _collect_substantive_changes(deep_changes)
                cur_substantive = _collect_substantive_changes(changes)
                if len(deep_substantive) > len(cur_substantive):
                    # Secondary quality check: don't adopt if deep rewrite introduces
                    # more anchor losses or generic bullets than current version
                    deep_anchor_notes, deep_generic_notes = 0, 0
                    cur_anchor_notes, cur_generic_notes = 0, 0
                    for item in _list_bullet_locations(original_resume_data):
                        src_text = item["text"]
                        deep_after = _get_bullet_by_path(deep_resume, item["exp_index"], item["project_index"], item["bullet_index"])
                        cur_after = _get_bullet_by_path(resume_data, item["exp_index"], item["project_index"], item["bullet_index"])
                        src_anchors = _extract_tech_anchor_tokens(src_text)
                        if deep_after is not None and src_anchors:
                            if not any(a in str(deep_after).lower() for a in src_anchors):
                                deep_anchor_notes += 1
                        if cur_after is not None and src_anchors:
                            if not any(a in str(cur_after).lower() for a in src_anchors):
                                cur_anchor_notes += 1
                        if deep_after is not None and _looks_over_generic(str(deep_after)) and not _looks_over_generic(src_text):
                            deep_generic_notes += 1
                        if cur_after is not None and _looks_over_generic(str(cur_after)) and not _looks_over_generic(src_text):
                            cur_generic_notes += 1
                    deep_quality_penalty = deep_anchor_notes + deep_generic_notes
                    cur_quality_penalty = cur_anchor_notes + cur_generic_notes
                    if deep_quality_penalty <= cur_quality_penalty:
                        resume_data = deep_resume
                        changes = deep_changes

    return {
        "optimized_resume": resume_data,
        "changes": changes,
        "warnings": llm_recovery_warnings,
    }


def run_single_optimize_with_audit_pass(
    resume_data: dict[str, Any],
    jd_text: Optional[str],
    style: str,
    source_resume_data: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not llm_enabled():
        raise HTTPException(status_code=500, detail="LLM is required for resume optimization but is not configured")

    original_resume_data = _clone_json(resume_data)
    master_data = source_resume_data if isinstance(source_resume_data, dict) else original_resume_data
    style = _normalize_style(style)
    safe_jd = sanitize_user_text(jd_text or "")
    quick_keywords = extract_jd_keywords(jd_text or "")

    prompt = (
        "请一次性完成优化和审计。\n"
        "返回字段必须包含 optimized_resume, audit_report, changes。\n\n"
        "【当前简历 JSON】\n"
        f"{json.dumps(resume_data, ensure_ascii=False)}\n\n"
        "【目标 JD（可选）】\n"
        f"{safe_jd}\n\n"
        "【优化目标】\n"
        "- 修复高/中风险可追问点\n"
        "- 保持事实保真，不捏造\n"
        "- 保留关键技术锚点\n"
        "- 首轮默认做实质性重写，不要只做语病/语序调整\n"
        "- 若问题较多，优先重写最弱的 2-4 个项目要点\n\n"
        "【关键词参考】\n"
        f"{json.dumps(quick_keywords[:20], ensure_ascii=False)}"
    )

    llm_out = call_llm_typed(
        OptimizeWithAuditLLMOutput,
        OPTIMIZE_WITH_AUDIT_SYSTEM_PROMPT,
        prompt,
        temperature=0.3 if style == "aggressive" else 0.2,
        max_tokens=4096,
    )

    optimized = llm_out.get("optimized_resume")
    raw_audit = llm_out.get("audit_report")
    llm_changes = llm_out.get("changes", [])
    single_pass_payload_invalid = False
    llm_recovery_warnings: list[str] = []
    if not isinstance(optimized, dict):
        logger.warning("Single-pass optimize LLM returned invalid optimized_resume; fallback to original resume_data")
        optimized = _clone_json(original_resume_data)
        single_pass_payload_invalid = True
        llm_recovery_warnings.append("LLM 返回结果格式异常，已使用原始简历数据")
    if not isinstance(raw_audit, dict):
        raw_audit = {}
        single_pass_payload_invalid = True
        llm_recovery_warnings.append("LLM 未返回有效审计报告")
    if not isinstance(llm_changes, list):
        llm_changes = []

    source_text_len = len(resume_data_to_text(original_resume_data))
    candidate_text_len = len(resume_data_to_text(optimized)) if isinstance(optimized, dict) else 0
    if source_text_len >= SHRINK_GUARD_MIN_SOURCE_CHARS and candidate_text_len <= max(120, int(source_text_len * 0.2)):
        single_pass_payload_invalid = True
        logger.warning(
            "Single-pass optimize payload is too sparse | source_text_len=%s candidate_text_len=%s",
            source_text_len,
            candidate_text_len,
        )

    if single_pass_payload_invalid and not raw_audit:
        logger.warning("Single-pass optimize payload invalid; fallback to standard optimize pass")
        seed_audit = audit_resume_core(resume_data_to_text(original_resume_data), jd_text)
        fallback_result = optimize_resume_core(
            resume_data=original_resume_data,
            audit_result=seed_audit,
            style=style,
            jd_text=jd_text,
            source_resume_data=master_data,
        )
        fallback_resume = fallback_result.get("optimized_resume")
        if isinstance(fallback_resume, dict):
            optimized = fallback_resume
        fallback_changes = fallback_result.get("changes", [])
        if isinstance(fallback_changes, list):
            llm_changes = fallback_changes
        fallback_warnings = fallback_result.get("warnings", [])
        if isinstance(fallback_warnings, list):
            llm_recovery_warnings.extend(fallback_warnings)
        raw_audit = seed_audit

    source_top_projects = original_resume_data.get("projects", [])
    if (not isinstance(optimized.get("projects"), list) or not optimized.get("projects")) and isinstance(source_top_projects, list) and source_top_projects:
        optimized["projects"] = _clone_json(source_top_projects)

    refined_resume = optimized
    stage_notes: list[str] = []
    refined_resume, refine_notes = _apply_alignment_refinement(refined_resume, master_data, jd_text)
    if refine_notes:
        stage_notes.extend(sorted(set(refine_notes)))

    refined_resume, fidelity_notes = _enforce_technical_anchor_fidelity(refined_resume, master_data)
    if fidelity_notes:
        stage_notes.extend(sorted(set(fidelity_notes)))

    # Chain-of-Verification: detect and revert fabricated numbers
    refined_resume, cov_notes = chain_of_verification(master_data, refined_resume)
    if cov_notes:
        stage_notes.extend(sorted(set(cov_notes)))

    refined_resume, trivial_notes = _revert_trivial_rewrites(refined_resume, original_resume_data)
    if trivial_notes:
        stage_notes.extend(sorted(set(trivial_notes)))

    should_revert, revert_reason = _should_guard_resume_shrink(original_resume_data, refined_resume)
    if should_revert:
        logger.warning("Shrink guard reverted single-pass optimized resume to source | reason=%s", revert_reason)
        refined_resume = _clone_json(original_resume_data)
        stage_notes.append(f"已回滚异常缩水改写（{revert_reason}）")

    optimized_text = resume_data_to_text(refined_resume)
    _log_resume_data_debug(
        stage="single_pass_optimize_resume_output",
        resume_data=refined_resume,
        extra={"raw_audit_present": bool(raw_audit)},
    )
    _log_parse_text_debug(
        stage="single_pass_optimize_text_output",
        resume_text=optimized_text,
        extra={"raw_audit_present": bool(raw_audit)},
    )
    if raw_audit:
        current_audit = normalize_audit_result(raw_audit, optimized_text, jd_text)
    else:
        logger.warning("run_single_optimize_with_audit_pass: empty audit_report; run standalone audit step")
        current_audit = audit_resume_core(optimized_text, jd_text)
        if not isinstance(current_audit, dict) or not current_audit:
            current_audit = _audit_fallback(
                optimized_text,
                jd_text,
                "run_single_optimize_with_audit_pass: empty audit_report",
            )

    changes = _derive_structured_changes(original_resume_data, refined_resume, current_audit)
    if not changes and llm_changes:
        changes = [item for item in llm_changes if isinstance(item, dict)]

    if _needs_forced_deep_rewrite(current_audit, changes, style):
        logger.info(
            "Escalate single-pass optimize to deep rewrite | issues=%s substantive_changes=%s",
            len(current_audit.get("issues", [])) if isinstance(current_audit, dict) else 0,
            len(_collect_substantive_changes(changes)),
        )
        fallback_result = optimize_resume_core(
            resume_data=original_resume_data,
            audit_result=current_audit,
            style="aggressive",
            jd_text=jd_text,
            source_resume_data=master_data,
        )
        fallback_resume = fallback_result.get("optimized_resume")
        fallback_changes = fallback_result.get("changes", [])
        fallback_warnings = fallback_result.get("warnings", [])
        if isinstance(fallback_warnings, list):
            llm_recovery_warnings.extend(fallback_warnings)
        if isinstance(fallback_resume, dict) and len(_collect_substantive_changes(fallback_changes if isinstance(fallback_changes, list) else [])) > len(_collect_substantive_changes(changes)):
            # Secondary quality check: don't adopt deep rewrite if it introduces more anchor loss or generic content
            _, fb_fidelity = _enforce_technical_anchor_fidelity(fallback_resume, master_data)
            fb_generic = sum(1 for exp in (fallback_resume.get("experience", []) or []) if isinstance(exp, dict)
                             for proj in (exp.get("projects", []) or []) if isinstance(proj, dict)
                             for b in (proj.get("bullets", []) or []) if _looks_over_generic(str(b)))
            orig_generic = sum(1 for exp in (refined_resume.get("experience", []) or []) if isinstance(exp, dict)
                               for proj in (exp.get("projects", []) or []) if isinstance(proj, dict)
                               for b in (proj.get("bullets", []) or []) if _looks_over_generic(str(b)))
            if len(fb_fidelity) <= 2 and fb_generic <= orig_generic:
                refined_resume = fallback_resume
                changes = fallback_changes if isinstance(fallback_changes, list) else changes
                optimized_text = resume_data_to_text(refined_resume)
                current_audit = audit_resume_core(optimized_text, jd_text)

    score = round(float(current_audit.get("overall_score", 0.0)), 1)
    history = [
        {
            "round": 1,
            "score": score,
            "issues_fixed": 0,
            "issues_remaining": len(current_audit.get("issues", [])),
            "score_before": score,
            "score_after": score,
        }
    ]
    return refined_resume, current_audit, history, changes, llm_recovery_warnings


# ═══════════════════════════════════════════════════════════════════════════════
# Patch-based Optimize (v3: FactLedger + per-bullet best-of-N)
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
from dataclasses import dataclass

from fact_ledger import FactBullet, FactLedger, build_ledger
from semantic_guard import BulletPatch, select_best_candidate


class _PatchOutput(BaseModel):
    """Pydantic model for single-bullet patch output."""
    bullet_id: str = ""
    new_text: str = ""


class _BulletAnalysisOutput(BaseModel):
    """LLM 输出：bullet 弱点诊断"""
    missing_situation: bool = False
    missing_task: bool = False
    missing_action: bool = False
    missing_result: bool = False
    missing_technical_detail: bool = False
    missing_metric: bool = False
    has_vague_language: bool = False


class _BulletVerdictOutput(BaseModel):
    """LLM 输出：改写结果校验"""
    is_safe: bool = True
    risk_tags: list[str] = []
    reason: str = ""


_NEW_PATCH_SYSTEM_PROMPT = """你是简历表达优化专家。基于诊断结果改写 bullet。

硬约束：
1. 已有指标（百分比、数字、规模）逐字保留，不能改数值
2. 公司名、岗位名、时间绝对不能动
3. 不提升角色层级：参与≠主导、协助≠负责
4. 不得编造原文没有的数字/指标或技术栈
5. 不得重复公司名和岗位名：同一段经历下多条 bullet 不需要每条都以"在XX公司担任XX期间"开头，直接用"负责/主导/参与"开头即可
6. 不得凭空编造技术细节：只有原文明确列为技能的技术词才能使用

改写方向（严格遵循诊断结果）：
- 若 missing_situation=true → 补充业务背景/场景描述
- 若 missing_technical_detail=true → 只能使用原文已有的技能词补充实现方式，不得编造
- 若 missing_metric=true → 如原文有指标则保留，无指标则写可验证口径
- 若 missing_result=true → 补充结果描述或验证方式
- 若 has_vague_language=true → 替换为具体描述

输出 JSON: {"bullet_id": "...", "new_text": "..."}"""

_ANALYZE_SYSTEM_PROMPT = """你是简历 bullet 诊断专家。分析下面这条 bullet 的弱点，输出结构化诊断。

判断标准：
- missing_situation: 是否缺少背景/场景/业务上下文
- missing_task: 是否缺少任务目标或用户问题
- missing_action: 是否缺少具体动作/技术方案/实现方式
- missing_result: 是否缺少结果/效果/验证
- missing_technical_detail: 是否缺少技术实现细节（框架/算法/工具/参数）
- missing_metric: 是否缺少可量化的指标
- has_vague_language: 是否使用了"先进""显著""大量"等空泛词

只输出 JSON。"""


def _scan_weak_bullets(ledger: FactLedger) -> list[str]:
    """Return bullet_ids that need rewriting. Rule-based, no LLM."""
    weak: list[str] = []
    for b in ledger.bullets:
        score = 0.0
        if not b.has_action:
            score += 1.0
        if not b.has_result:
            score += 2.0
        if len(b.source_text) < 25:
            score += 1.0
        if len(b.source_text) > 200:
            score += 0.5
        if score >= 1.0:
            weak.append(b.id)
    logger.info(
        "_scan_weak_bullets: %d/%d bullets flagged as weak",
        len(weak), len(ledger.bullets),
    )
    return weak


def analyze_bullet(bullet: FactBullet) -> Optional[dict[str, Any]]:
    """Step 1: Analyze bullet weakness. Returns structured diagnosis dict."""
    if not llm_enabled():
        return None
    prompt = (
        "【当前 bullet】\n"
        f"{bullet.source_text}\n\n"
        "【所属上下文】\n"
        f"{bullet.context}\n\n"
        "输出诊断 JSON。"
    )
    try:
        result = call_llm_text(
            _ANALYZE_SYSTEM_PROMPT,
            prompt,
            temperature=0.1,
            max_tokens=192,
        )
        parsed = parse_json_content(result)
        if not isinstance(parsed, dict) or not parsed:
            return None
        return parsed
    except Exception as exc:
        logger.warning("analyze_bullet failed for %s: %s", bullet.id, exc)
        return None


def conceive_material(
    bullet: FactBullet, ledger: FactLedger,
    resume_data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Step 2: Extract safe-to-use material from CV (rules, no LLM call).

    Only pulls tech keywords from structured skills (languages/frameworks).
    Does NOT regex raw_text — that picks up too much noise (Office/TAL/etc).

    Returns dict with:
      - available_tech: list[str] — tech keywords from skills section
      - available_metrics: list[str] — numbers/metrics from source
      - safe_angles: list[str] — description directions from entities
    """
    tech_keywords: list[str] = []
    metric_keywords: list[str] = []
    angle_keywords: list[str] = []

    # Collect tech keywords from structured skills (languages/frameworks only)
    if resume_data:
        skills = resume_data.get("skills", {})
        if isinstance(skills, dict):
            for bucket in ("languages", "frameworks"):
                items = skills.get(bucket, [])
                if isinstance(items, list):
                    for item in items:
                        s = str(item).strip()
                        if len(s) >= 2 and s not in tech_keywords:
                            tech_keywords.append(s)

    # Collect angle keywords from entities (role, project_name)
    for (kind, val_lower), entity in ledger.entities.items():
        if kind in ("role", "project_name"):
            val = entity.value.strip()
            if len(val) >= 2 and val not in angle_keywords:
                angle_keywords.append(val)

    # Collect metrics from bullet
    for m in bullet.metrics:
        if m not in metric_keywords:
            metric_keywords.append(m)

    return {
        "available_tech": tech_keywords[:8],
        "available_metrics": metric_keywords[:6],
        "safe_angles": angle_keywords[:5],
    }


_VERIFY_SYSTEM_PROMPT = """你是简历事实核查员。对比改写前后的 bullet，判断改写是否突破事实边界。

判断标准（任一 true 则 is_safe=false）：
1. fabricated_company: 原文没有的公司名、组织名
2. fabricated_metric: 原文没有的指标/数字
3. role_promotion: 角色层级被提升（参与→主导、协助→负责）
4. fabricated_tech: 技术栈/工具名完全不在原文
5. hallucinated_angle: 完全虚构的描述方向

只输出 JSON。"""


def verify_bullet(bullet: FactBullet, new_text: str, ledger: FactLedger) -> dict[str, Any]:
    """Step 4: Verify that the rewritten bullet is safe (no fabrication)."""
    if not llm_enabled():
        return {"is_safe": True, "risk_tags": [], "reason": "LLM disabled"}
    prompt = (
        "【原文 bullet】\n"
        f"{bullet.source_text}\n\n"
        "【改写后 bullet】\n"
        f"{new_text}\n\n"
        "【原始简历上下文】\n"
        f"{ledger.raw_text[:1500]}\n\n"
        "输出校验 JSON。"
    )
    try:
        result = call_llm_typed(
            _BulletVerdictOutput,
            _VERIFY_SYSTEM_PROMPT,
            prompt,
            temperature=0.1,
            max_tokens=128,
            prefill='{"is_safe":',
        )
        if isinstance(result, dict):
            return result
        return {"is_safe": True, "risk_tags": [], "reason": "parse fallback"}
    except Exception as exc:
        logger.warning("verify_bullet failed for %s: %s", bullet.id, exc)
        return {"is_safe": True, "risk_tags": [], "reason": "verify exception"}


def _rewrite_bullet(
    bullet: FactBullet,
    jd_keywords: list[str],
    ledger: FactLedger,
    analysis: Optional[dict[str, Any]] = None,
    material: Optional[dict[str, Any]] = None,
) -> str:
    """Step 3: Rewrite a single bullet given analysis + material. Sync wrapper."""
    # Build diagnosis-specific guidance
    analysis_block = ""
    if analysis:
        guides = []
        if analysis.get("missing_situation"):
            guides.append("补充业务背景：说明在什么场景或需求下做这件事")
        if analysis.get("missing_technical_detail"):
            guides.append("补充具体做法：说明技术方案或实现方式（仅使用原文已有的技能词）")
        if analysis.get("missing_result"):
            guides.append("补充结果：说明做完后的效果或验证方式")
        if analysis.get("missing_metric"):
            guides.append("已有指标保留，无指标时写可验证口径不编造数字")
        if analysis.get("has_vague_language"):
            guides.append("替换空泛词（先进/显著/大量等）为具体描述")
        if guides:
            analysis_block = "【重点改写方向】\n" + "\n".join(f"- {g}" for g in guides) + "\n\n"
    else:
        analysis_block = "【重点改写方向】\n- 按 STAR 重组表达，使描述更清晰\n\n"

    material_block = ""
    if material:
        parts = []
        if material.get("available_tech"):
            parts.append(f"已有技能词: {', '.join(material['available_tech'][:8])}")
        if material.get("available_metrics"):
            parts.append(f"已有指标: {', '.join(material['available_metrics'][:6])}")
        if parts:
            material_block = "\n".join(parts) + "\n"

    jd_str = ", ".join(jd_keywords[:8]) if jd_keywords else ""
    jd_block = f"【JD 关键词】\n{jd_str}\n\n" if jd_str else ""

    prompt = (
        "【不可变事实】\n"
        f"所属: {bullet.context}\n"
        f"已有指标: {', '.join(bullet.metrics[:6]) if bullet.metrics else '（无已有指标）'}\n\n"
        f"{analysis_block}"
        f"{material_block}"
        f"{jd_block}"
        f"【当前 bullet】\n{bullet.source_text}\n\n"
        f'输出 JSON: {{"bullet_id": "{bullet.id}", "new_text": "..."}}'
    )

    try:
        result = call_llm_typed(
            _PatchOutput,
            _NEW_PATCH_SYSTEM_PROMPT,
            prompt,
            temperature=0.35,
            max_tokens=512,
        )
        new_text = result.get("new_text", "") if isinstance(result, dict) else ""
        return str(new_text).strip() if new_text else ""
    except Exception as exc:
        logger.warning("Rewrite LLM call failed for bullet=%s: %s", bullet.id, exc)
        return ""


async def patch_optimize_weak_bullets(
    ledger: FactLedger,
    jd_keywords: list[str],
    n: int = 3,
    resume_data: Optional[dict[str, Any]] = None,
) -> list[BulletPatch]:
    """4-stage pipeline: Analyze -> Conceive -> Rewrite -> Verify, best-of-N.

    Args:
        ledger: immutable FactLedger from parse output
        jd_keywords: JD keywords extracted via extract_jd_keywords()
        n: number of candidates per bullet (default 3)
        resume_data: structured resume JSON (for skills extraction)

    Returns:
        list of BulletPatch, one per successfully rewritten bullet
    """
    if not llm_enabled():
        logger.info("patch_optimize: LLM disabled, skipping")
        return []

    weak_ids = _scan_weak_bullets(ledger)
    if not weak_ids:
        logger.info("patch_optimize: no weak bullets found")
        return []

    patches: list[BulletPatch] = []
    bullet_map: dict[str, FactBullet] = {b.id: b for b in ledger.bullets}

    # ── Step 1: Analyze all weak bullets (1 LLM each) ──
    analyses: dict[str, Optional[dict[str, Any]]] = {}
    for bid in weak_ids:
        bullet = bullet_map.get(bid)
        if bullet is None:
            continue
        analyses[bid] = await asyncio.to_thread(analyze_bullet, bullet)

    # ── Step 2: Conceive material for all (pure rules, 0 LLM) ──
    materials: dict[str, dict[str, Any]] = {}
    for bid in weak_ids:
        bullet = bullet_map.get(bid)
        if bullet is None:
            continue
        materials[bid] = conceive_material(bullet, ledger, resume_data=resume_data)

    # ── Step 3+4: Rewrite + Verify, best-of-N ──
    for bid in weak_ids:
        bullet = bullet_map.get(bid)
        if bullet is None:
            continue

        analysis = analyses.get(bid)
        material = materials.get(bid)

        # Generate N candidates concurrently (2 at a time for max_num_seqs=2)
        candidates: list[str] = []
        for batch_start in range(0, n, 2):
            batch_size = min(2, n - batch_start)
            tasks = []
            for _ in range(batch_size):
                tasks.append(asyncio.to_thread(
                    _rewrite_bullet, bullet, jd_keywords, ledger, analysis, material
                ))
            batch_results = await asyncio.gather(*tasks)

            # Step 4: Verify each candidate (DISABLED — too aggressive with watermark data)
            # for r in batch_results:
            #     if not r: continue
            #     verdict = verify_bullet(bullet, r, ledger)
            #     if verdict.get("is_safe", False):
            #         candidates.append(r)
            #     else:
            #         logger.info("Verify reject bullet=%s: %s", bid, verdict.get("reason", "unsafe"))
            candidates.extend(r for r in batch_results if r)

        if not candidates:
            logger.info("No candidates for bullet=%s, keeping original", bid)
            continue

        # Select best from candidates
        try:
            best = select_best_candidate(bullet, candidates, ledger, jd_keywords)
            if best is not None:
                patches.append(best)
            else:
                logger.info("All candidates rejected by selector for bullet=%s, keeping original", bid)
        except Exception as exc:
            logger.warning("select_best_candidate failed for bullet=%s: %s", bid, exc)

    logger.info(
        "patch_optimize: %d weak bullets -> %d patches applied (%d unchanged)",
        len(weak_ids), len(patches), len(weak_ids) - len(patches),
    )
    return patches


def _dedup_bullets(bullets: list[str], threshold: float = 0.65) -> list[str]:
    """Remove near-duplicate bullets within the same entry.

    Uses word-level Jaccard — fast, no external deps, works for Chinese.
    Cross-company duplicates are not removed (same company might have truly
    similar roles). Logs a warning when dedup occurs.
    """
    if len(bullets) <= 1:
        return bullets

    def _tokenize(text: str) -> frozenset[str]:
        # Simple character bigram tokenization works well for both Chinese and English
        chars = re.sub(r"\s+", "", str(text))
        bigrams = {chars[i:i+2] for i in range(len(chars) - 1)} if len(chars) >= 2 else {chars}
        # Also add full word tokens for English text
        words = set(re.findall(r"[a-zA-Z]{3,}", chars))
        return frozenset(bigrams | words)

    result = [bullets[0]]
    tokens = [_tokenize(bullets[0])]

    for b in bullets[1:]:
        bt = _tokenize(b)
        is_dup = False
        for prev_t in tokens:
            if not bt or not prev_t:
                continue
            union = len(bt | prev_t)
            if union == 0:
                continue
            jaccard = len(bt & prev_t) / union
            if jaccard > threshold:
                is_dup = True
                logger.info("Dedup: removed near-duplicate bullet (Jaccard=%.2f): %s", jaccard, b[:80])
                break
        if not is_dup:
            result.append(b)
            tokens.append(bt)

    if len(result) < len(bullets):
        logger.info("Dedup: %d/%d bullets kept after intra-entry dedup", len(result), len(bullets))
    return result


def apply_patches(resume_data: dict[str, Any], patches: list[BulletPatch]) -> dict[str, Any]:
    """Apply BulletPatch list back to resume_data. Only modifies bullet text fields."""
    import copy
    data = copy.deepcopy(resume_data)
    patch_map = {p.bullet_id: p.new_text for p in patches}

    def _get_bullet_list(entry: dict[str, Any]) -> list[str]:
        """Collect all bullet text fields from an entry as a flat list."""
        result: list[str] = []
        for key in ("bullets", "function_description", "result_description"):
            val = entry.get(key)
            if isinstance(val, str) and val.strip():
                result.append(val.strip())
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str) and item.strip():
                        result.append(item.strip())
        return result

    def _set_bullets(entry: dict[str, Any], new_bullets: list[str]) -> None:
        """Replace bullets field with new list."""
        # Write back to "bullets" field (standardized by normalize)
        entry["bullets"] = new_bullets
        # Keep function_description / result_description if they exist as separate fields
        for key in ("function_description", "result_description"):
            if isinstance(entry.get(key), str):
                entry[key] = ""  # Clear merged sources

    # Apply to experience
    for exp_idx, exp in enumerate(data.get("experience", []) or []):
        if not isinstance(exp, dict):
            continue
        bullets = _get_bullet_list(exp)
        changed = False
        for b_idx, _ in enumerate(bullets):
            bid = f"exp_{exp_idx}_b{b_idx}"
            if bid in patch_map:
                bullets[b_idx] = patch_map[bid]
                changed = True
        if changed:
            bullets = _dedup_bullets(bullets)
            _set_bullets(exp, bullets)

    # Apply to projects
    for proj_idx, proj in enumerate(data.get("projects", []) or []):
        if not isinstance(proj, dict):
            continue
        bullets = _get_bullet_list(proj)
        changed = False
        for b_idx, _ in enumerate(bullets):
            bid = f"proj_{proj_idx}_b{b_idx}"
            if bid in patch_map:
                bullets[b_idx] = patch_map[bid]
                changed = True
        if changed:
            bullets = _dedup_bullets(bullets)
            _set_bullets(proj, bullets)

    return data
