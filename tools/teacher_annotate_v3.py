#!/usr/bin/env python3
"""V3 教师标注管线：用内网 DeepSeek 驱动生产 run_v3_pipeline，产出冻结 schema 的训练 trace。

每条简历的真实路径: 文本 → DocumentGraph → FactGraph → semantic_compile(教师) →
plan → realize(教师) → audit。训练样本由管线自身的 training_examples 落盘
(schema_version + 指纹 + validation 状态), validation=ok 的进 SFT 池, 失败的进 DPO rejected 池。

用法:
  set -a; source ~/.config/claude-router/env; set +a
  python3 tools/teacher_annotate_v3.py --n 10 --concurrency 2
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "core"))  # core 模块间按顶层包互引
MATERIALS = Path("/mnt/disk1/zengzhitao/data/resume_copilot_materials")

API_URL = "http://172.28.4.52:8888/v1/chat/completions"
MODEL = "DeepSeek-V4-Flash-0731"

# ---------------- 教师调用（满足生产 llm_call 契约: (Model, system, user_json, temperature, max_tokens) -> dict） ----------------

def make_teacher_call(stats: dict):
    def teacher_call(schema_model, system_prompt: str, user_content: str,
                     temperature: float = 0.0, max_tokens: int = 4096) -> dict:
        # 对齐生产 _build_messages: 输出规则 + 目标 schema 注入 system + assistant prefill "{"
        rules = ("\n\n【输出规则】\n1) 只输出一个 JSON 对象，不要输出任何解释文字\n"
                 "2) 不要输出 markdown 代码块标记（不要```json）\n3) 不要在 JSON 前后添加任何额外内容\n"
                 '4) 字符串中的双引号必须转义为 \\"\n5) 不要在最后一个元素后加逗号')
        schema_hint = json.dumps(schema_model.model_json_schema(), ensure_ascii=False)
        # 标注期附加提示(不进训练样本): 针对教师实测高发错误
        model_name = getattr(schema_model, "__name__", "")
        if "Realizer" in model_name:
            annotation_hint = (
                "\n\n【常见错误警示】"
                "\n- schema_version 字段必须输出且严格等于 resume_compiler_v3.4"
                "\n- 第3条'逐字保留'的含义: claim.text 必须**包含**每条引用事实 source_text 的完整原文子串,"
                " 只允许在其间加连接词。示例: source_text='销售额提升30%' → 合法 claim.text='主导区域销售, 销售额提升30%, 覆盖20家客户';"
                " 非法 claim.text='显著提升销售业绩'(原文子串丢失)"
                "\n- summary claim 用 group_id='summary:profile', record_id=null"
            )
        else:
            annotation_hint = (
                "\n\n【常见错误警示】"
                "\n- schema_version 字段必须输出且严格等于 resume_compiler_v3.4"
                "\n- atom.fact_type 只能取 Schema 枚举值；intent/experience 不是合法 fact_type"
                "\n- [EMAIL]、[PHONE]、[姓名] 等方括号占位符一律归入 context_spans(reason=placeholder), 禁止作为 atom"
                "\n- 一条原文中占位符与事实混排时, 必须拆成独立原子, 不得混入同一 atom"
            )
        messages = [
            {"role": "system", "content": system_prompt + rules + annotation_hint + "\n\n【目标 JSON Schema】\n" + schema_hint},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "{"},
        ]
        body = {
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "continue_final_message": True,
            "add_generation_prompt": False,
        }
        last: Exception | None = None
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    API_URL, data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {os.environ['DEEPSEEK_V4_FLASH_LOCAL_API_KEY']}"},
                )
                with urllib.request.urlopen(req, timeout=600 if max_tokens > 5000 else 300) as resp:
                    data = json.loads(resp.read())
                stats["calls"] += 1
                stats["tokens"] += data.get("usage", {}).get("total_tokens", 0)
                content = data["choices"][0]["message"]["content"]
                # 与生产一致: 服务未回显 prefill 时补回 "{"
                if not content.lstrip().startswith("{"):
                    content = "{" + content
                return json.loads(content)
            except Exception as exc:  # noqa: BLE001
                last = exc
                stats["retries"] += 1
                time.sleep(min(10 * (2 ** attempt), 60))
        raise last  # type: ignore[misc]

    return teacher_call


# ---------------- 素材加载 ----------------

def load_resumes(n: int, seed: int, min_recall: float) -> list[dict]:
    """翻译库(达标) + JobResQA 原生中文, 混合抽样。"""
    pool: list[dict] = []
    full = MATERIALS / "translations/full_v1.jsonl"
    if full.exists():
        seen = set()
        for line in full.open():
            r = json.loads(line)
            if r.get("status") != "ok" or r["source_id"] in seen:
                continue
            seen.add(r["source_id"])
            if r["verify"]["number_recall"] >= min_recall and len(r["zh_resume_style"]) >= 600:
                pool.append({"id": r["source_id"], "text": r["zh_resume_style"],
                             "origin": "translated", "category": r.get("category", "")})
    for p in sorted((MATERIALS / "resumes/jobresqa_train").glob("*.txt")):
        text = p.read_text().strip()
        # JobResQA 中文是单行机翻文本, 确定性前端按行切分会塌缩成 1 个 fact, 暂跳过(待预分段)
        if len(text) >= 400 and text.count("\n") >= 5:
            pool.append({"id": p.stem, "text": text, "origin": "jobresqa", "category": ""})
    rng = random.Random(seed)
    rng.shuffle(pool)
    # 保证原生中文至少占 1/4（如果够）
    native = [x for x in pool if x["origin"] == "jobresqa"]
    trans = [x for x in pool if x["origin"] == "translated"]
    n_native = min(len(native), max(n // 4, 1 if n else 0))
    picked = native[:n_native] + trans[: max(n - n_native, 0)]
    rng.shuffle(picked)
    return picked[:n]


_JD_CACHE: list[str] | None = None

def load_jd_pool(limit: int = 4000) -> list[str]:
    global _JD_CACHE
    if _JD_CACHE is not None:
        return _JD_CACHE
    import csv
    csv.field_size_limit(sys.maxsize)
    pool = []
    with (MATERIALS / "raw/job_edu_parser/default_train_19w_0701.csv").open() as f:
        for row in csv.DictReader(f):
            text = row.get("user", "").strip()
            if 300 <= len(text) <= 2500:
                pool.append(text)
            if len(pool) >= limit:
                break
    _JD_CACHE = pool
    return pool


def match_jd(resume_text: str, pool: list[str], top_k: int = 50) -> str:
    """粗关键词重叠匹配：取简历中高频实词, 找 JD 覆盖最多者（模拟场景1 有相关性的 JD）。"""
    tokens = {t for t in re.findall(r"[A-Za-z][A-Za-z0-9+#.]{1,}|[一-鿿]{2,4}", resume_text)}
    tokens = {t for t in tokens if len(t) >= 2}
    best, best_score = "", -1.0
    sample = pool if len(pool) <= top_k * 20 else random.Random(0).sample(pool, top_k * 20)
    for jd in sample:
        jd_set = set(re.findall(r"[A-Za-z][A-Za-z0-9+#.]{1,}|[一-鿿]{2,4}", jd))
        score = len(tokens & jd_set) / max(len(tokens), 1)
        if score > best_score:
            best, best_score = jd, score
    return best


# ---------------- 主流程 ----------------

def annotate_one(resume: dict, jd: str, trace_dir: Path, stats: dict) -> dict:
    """分阶段驱动 V3: semantic(教师) → plan → realize(强制教师)。

    与生产 run_v3_pipeline 的差异: 生产在 semantic 存在 fallback 时禁用 realize LLM
    (信任边界), 但 fallback 事实仍是精确源事实, 对标注而言 realizer 输入依然良构,
    强制调用可同时拿到 realize 训练对。
    """
    from core.v3.contracts import SourcePolicy, TemplateAST
    from core.v3.fact_graph import build_fact_graph
    from core.v3.jd_graph import build_requirement_graph
    from core.v3.pipeline import _native_graph
    from core.v3.planner import plan_resume
    from core.v3.realizer_llm import realize_with_llm
    from core.v3.semantic_llm import compile_semantics
    from core.v3.training_examples import build_training_records

    rec = {"id": resume["id"], "origin": resume["origin"], "chars": len(resume["text"])}
    t0 = time.time()
    try:
        graph0 = _native_graph("cv", "cv", resume["text"])
        transport = build_fact_graph([graph0], SourcePolicy())
        n_candidates = len([f for f in transport.facts if f.source_type not in {"jd", "template"}])

        teacher = make_teacher_call(stats)
        semantic = compile_semantics(transport, use_llm=True, llm_call=teacher)
        graph = semantic.graph
        requirements = build_requirement_graph(jd)
        plan = plan_resume(graph, requirements, TemplateAST(mode="style_only"))
        realization = realize_with_llm(plan, graph, use_llm=True, llm_call=teacher)

        records = build_training_records(
            semantic_inputs=semantic.report.training_inputs,
            semantic_outputs=semantic.report.training_outputs,
            semantic_status=semantic.report.status,
            semantic_errors=semantic.report.errors,
            realizer_input=realization.report.training_input,
            realizer_output=realization.report.training_output,
            realizer_status=realization.report.status,
            realizer_violations=realization.report.violations,
        )
        trace_file = trace_dir / f"{resume['origin']}_{resume['id'].replace(':', '_')}.jsonl"
        trace_file.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records) + "\n",
            encoding="utf-8",
        )
        rec.update(
            status="ok",
            candidates=n_candidates,
            semantic_status=semantic.report.status,
            semantic_fallback_facts=len(semantic.report.fallback_fact_ids),
            invalid_atoms=semantic.report.invalid_atom_count,
            realizer_status=realization.report.status,
            realizer_violations=len(realization.report.violations),
            realizer_error=realization.report.error[:200],
            claims=len(realization.frozen.claims),
            latency_s=round(time.time() - t0, 1),
        )
    except Exception as exc:  # noqa: BLE001
        rec.update(status="pipeline_error", error=str(exc)[:300], latency_s=round(time.time() - t0, 1))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--min-recall", type=float, default=0.85)
    ap.add_argument("--trace-dir", default=str(MATERIALS / "training_data/v3_traces"))
    ap.add_argument("--summary", default=str(MATERIALS / "training_data/annotate_summary.jsonl"))
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)

    resumes = load_resumes(args.n, args.seed, args.min_recall)
    jd_pool = load_jd_pool()
    print(f"候选简历 {len(resumes)} 份 | JD 池 {len(jd_pool)} 条", flush=True)

    stats = {"calls": 0, "tokens": 0, "retries": 0}
    done_ids = set()
    if Path(args.summary).exists():
        done_ids = {json.loads(l)["id"] for l in open(args.summary)}
    todo = [r for r in resumes if r["id"] not in done_ids]
    print(f"断点: 跳过已完成 {len(resumes) - len(todo)} 份", flush=True)

    results = []
    with open(args.summary, "a") as fout, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(annotate_one, r, match_jd(r["text"], jd_pool), trace_dir, stats): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            results.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{i}/{len(todo)}] {rec['id']} status={rec['status']} "
                  f"latency={rec.get('latency_s')}s", flush=True)

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\n=== 完成 {len(ok)}/{len(results)} ===")
    print(f"教师调用 {stats['calls']} 次, tokens {stats['tokens']}, 重试 {stats['retries']}")
    if ok:
        sem_ok = sum(1 for r in ok if r.get("semantic_status") in {"ok", "schema_valid"})
        real_ok = sum(1 for r in ok if r.get("realizer_status") == "ok")
        print(f"semantic 全干净: {sem_ok}/{len(ok)} | realize ok: {real_ok}/{len(ok)}")
        print(f"claims 总计: {sum(r.get('claims', 0) for r in ok)}")
    print(f"trace 目录: {trace_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
