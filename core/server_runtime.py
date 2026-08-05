import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from llm_gateway import LLMDeadlineExceeded, LLMGateway

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pikepdf
except ImportError:
    pikepdf = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

API_BASE_URL = os.getenv("MODELHUB_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("MODELHUB_API_KEY", "")
MODEL_NAME = os.getenv("MODELHUB_MODEL_NAME", "/data/Qwen3.6-27B-AWQ-INT4")
DEFAULT_OUTPUT_FORMAT = os.getenv("DEFAULT_OUTPUT_FORMAT", "both")
DEFAULT_TEMPLATE = os.getenv("DEFAULT_TEMPLATE", "classic")
FIXED_OUTPUT_FORMAT = "docx"
ENABLE_LLM_JSON_REPAIR = os.getenv("ENABLE_LLM_JSON_REPAIR", "1").strip() != "0"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(Path(__file__).parent / "output")))
DRAFTS_DIR = OUTPUT_DIR / "drafts"
AVATAR_DIR = OUTPUT_DIR / "assets"
LLM_FAILURE_DUMP_DIR = OUTPUT_DIR / "debug" / "llm_json_failures"
ENABLE_LLM_FAILURE_DUMP = os.getenv("ENABLE_LLM_FAILURE_DUMP", "0").strip() == "1"
ENABLE_PARSE_DEBUG_LOG = os.getenv("ENABLE_PARSE_DEBUG_LOG", "0").strip() != "0"
ENABLE_HEURISTIC_AUDIT_FALLBACK = os.getenv("ENABLE_HEURISTIC_AUDIT_FALLBACK", "1").strip() != "0"
ENABLE_RESUME_SHRINK_GUARD = os.getenv("ENABLE_RESUME_SHRINK_GUARD", "1").strip() != "0"
ENABLE_TEXT_LAYOUT_NORMALIZATION = os.getenv("ENABLE_TEXT_LAYOUT_NORMALIZATION", "1").strip() != "0"
ENABLE_AVATAR_EXTRACTION = os.getenv("ENABLE_AVATAR_EXTRACTION", "1").strip() != "0"
ENABLE_AVATAR_FACE_SCORE = os.getenv("ENABLE_AVATAR_FACE_SCORE", "1").strip() != "0"
ENABLE_LLM_CLASSIFIER = os.getenv("ENABLE_LLM_CLASSIFIER", "1").strip() != "0"
ENABLE_LLM_REPLY = os.getenv("ENABLE_LLM_REPLY", "1").strip() != "0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None
_llm_gateway: Optional[LLMGateway] = None

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
if ENABLE_AVATAR_EXTRACTION:
    AVATAR_DIR.mkdir(parents=True, exist_ok=True)
if ENABLE_LLM_FAILURE_DUMP:
    LLM_FAILURE_DUMP_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_UPLOAD_EXTENSIONS = {
    ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".bmp", ".webp",
    ".gif", ".tif", ".tiff", ".txt", ".md",
}
SUPPORTED_FILE_PATH_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
MAX_FILE_SIZE = 20 * 1024 * 1024
def _safe_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except Exception:
        return default


PARSE_DEBUG_PREVIEW_CHARS = _safe_env_int("PARSE_DEBUG_PREVIEW_CHARS", 260)
PARSE_DEBUG_MAX_BULLETS = _safe_env_int("PARSE_DEBUG_MAX_BULLETS", 5)
SHRINK_GUARD_MIN_SOURCE_CHARS = _safe_env_int("SHRINK_GUARD_MIN_SOURCE_CHARS", 500)
MAX_AUDIT_ISSUES = _safe_env_int("MAX_AUDIT_ISSUES", 12)
LLM_TIMEOUT_SECONDS = _safe_env_int("LLM_TIMEOUT_SECONDS", 180)
REQUEST_TIMEOUT_SECONDS = _safe_env_int("REQUEST_TIMEOUT_SECONDS", 480)
LLM_DEADLINE_RESERVE_SECONDS = _safe_env_int("LLM_DEADLINE_RESERVE_SECONDS", 2)
LLM_RETRY_MIN_REMAINING_SECONDS = _safe_env_int("LLM_RETRY_MIN_REMAINING_SECONDS", 10)
LLM_INFLIGHT_LIMIT = _safe_env_int("LLM_INFLIGHT_LIMIT", 2)
DATA_RETENTION_SECONDS = _safe_env_int("DATA_RETENTION_SECONDS", 7 * 24 * 60 * 60)

# Bound all application-side LLM calls, including calls from different resume
# requests and the Composer's internal workers. vLLM owns one shared model and
# KV-cache pool, so this limit must match the production ``max-num-seqs`` value.
_LLM_INFLIGHT_SLOTS = threading.BoundedSemaphore(LLM_INFLIGHT_LIMIT)

_REQUEST_DEADLINE_AT: ContextVar[Optional[float]] = ContextVar(
    "resume_copilot_request_deadline_at",
    default=None,
)


def get_request_deadline() -> Optional[float]:
    """Return the inherited absolute monotonic request deadline, if any."""

    return _REQUEST_DEADLINE_AT.get()


def set_request_deadline(
    *,
    deadline_at: Optional[float] = None,
    timeout_seconds: Optional[float] = None,
) -> Token:
    """Set a request deadline without extending an inherited outer deadline.

    ``asyncio.to_thread`` copies context variables, so synchronous Composer and
    OpenAI calls made in worker threads continue to see this deadline.
    """

    candidates: list[float] = []
    inherited = get_request_deadline()
    if inherited is not None:
        candidates.append(inherited)
    if deadline_at is not None:
        candidates.append(float(deadline_at))
    if timeout_seconds is not None:
        candidates.append(time.monotonic() + max(0.0, float(timeout_seconds)))
    if not candidates:
        raise ValueError("deadline_at or timeout_seconds is required")
    return _REQUEST_DEADLINE_AT.set(min(candidates))


def reset_request_deadline(token: Token) -> None:
    """Restore the deadline context that existed before ``set``."""

    _REQUEST_DEADLINE_AT.reset(token)


def remaining_request_seconds() -> Optional[float]:
    """Return non-negative request time remaining, or ``None`` outside a request."""

    deadline_at = get_request_deadline()
    if deadline_at is None:
        return None
    return max(0.0, deadline_at - time.monotonic())


@contextmanager
def _llm_inflight_slot():
    """Acquire one global backend slot without waiting past the request budget."""

    started = time.perf_counter()
    remaining = remaining_request_seconds()
    if remaining is None:
        acquired = _LLM_INFLIGHT_SLOTS.acquire()
    else:
        usable = max(0.0, remaining - LLM_DEADLINE_RESERVE_SECONDS)
        acquired = _LLM_INFLIGHT_SLOTS.acquire(timeout=usable)
    if not acquired:
        raise LLMDeadlineExceeded("request deadline reached while waiting for an LLM slot")
    waited = time.perf_counter() - started
    if waited >= 0.05:
        logger.info(
            "LLM inflight slot acquired | waited_s=%.3f limit=%d",
            waited,
            LLM_INFLIGHT_LIMIT,
        )
    try:
        yield
    finally:
        _LLM_INFLIGHT_SLOTS.release()

SECTION_HEADING_KEYWORDS = (
    "个人简历",
    "personal resume",
    "基本信息",
    "求职信息",
    "教育经历",
    "教育背景",
    "工作/实习经历",
    "工作经历",
    "实习经历",
    "项目经历",
    "项目经验",
    "研究项目",
    "科研项目",
    "科研项目参与情况",
    "项目参与情况",
    "论文发表情况",
    "论文成果",
    "学术成果",
    "荣誉与奖项",
    "获奖情况",
    "英语及计算机技能",
    "专业技能",
    "个人技能",
    "自我评价",
    "专业素养培养",
    "参与专著写作情况",
    "协助本科生培养工作",
    "学术会议参与情况",
    "语言能力",
    "语言水平",
    "证书",
    "资格证书",
    "培训经历",
    "社会实践",
    "校园经历",
    "学生工作",
    "干部经历",
    "组织经历",
    "兴趣爱好",
    "特长",
    "发表论文",
    "专利",
    "著作",
)
PROJECT_SECTION_HEADERS = (
    "项目经历",
    "项目经验",
    "研究项目",
    "科研项目",
    "科研项目参与情况",
    "项目参与情况",
)
PROJECT_SECTION_STOP_HEADERS = (
    "教育经历",
    "教育背景",
    "实习经历",
    "工作经历",
    "工作/实习经历",
    "专业技能",
    "个人技能",
    "荣誉与奖项",
    "获奖情况",
    "论文发表情况",
    "论文成果",
    "学术成果",
    "英语及计算机技能",
    "自我评价",
    "专业素养培养",
    "学术会议参与情况",
    "参与专著写作情况",
    "协助本科生培养工作",
    "求职意向",
)
PERSONAL_SECTION_STOP_HEADERS = (
    "获奖情况",
    "荣誉与奖项",
    "英语及计算机技能",
    "论文发表情况",
    "论文成果",
    "学术成果",
    "项目经历",
    "项目经验",
    "工作经历",
    "工作/实习经历",
    "教育经历",
    "教育背景",
)
PROJECT_DATE_RANGE_PATTERN = re.compile(
    r"((?:19|20)\d{2}[./年]\d{1,2}(?:月)?\s*(?:[-–—~至到]+\s*(?:(?:19|20)\d{2}[./年]\d{1,2}(?:月)?|至今|Present|present)))",
    re.IGNORECASE,
)

TECH_KEYWORDS = {
    "redis", "mysql", "postgres", "postgresql", "kafka", "spark", "flink", "hadoop", "pytorch", "tensorflow",
    "transformer", "bert", "xgboost", "grpc", "k8s", "kubernetes", "docker", "nginx", "elasticsearch",
    "clickhouse", "prometheus", "grafana", "lua", "go", "python", "java", "c++", "cpp", "rust",
    "react", "vue", "angular", "svelte", "nextjs", "nuxt", "django", "flask", "fastapi", "spring",
    "celery", "airflow", "kubeflow", "terraform", "ansible", "jenkins", "github", "gitlab",
    "mongodb", "cassandra", "dynamodb", "sqlite", "mariadb", "cockroachdb",
    "rabbitmq", "pulsar", "nats", "zookeeper", "consul", "etcd",
    "memcached", "varnish", "envoy", "istio", "linkerd", "traefik",
    "hbase", "hive", "presto", "trino", "dbt", "snowflake",
    "onnx", "tensorrt", "cuda", "rocm",
    "git", "svn", "mercurial",
    "linux", "unix", "bash", "powershell",
    "aws", "azure", "gcp", "alibaba", "aliyun",
    "rest", "graphql", "websocket", "thrift",
    "mlops", "devops", "ci/cd", "sre",
}

ZH_TECH_TERMS = {
    "微调", "蒸馏", "对齐", "预训练", "推理", "训练", "部署", "量化", "剪枝", "蒸馏",
    "检索增强", "向量检索", "知识蒸馏", "数据增强", "特征工程", "模型压缩", "在线学习",
    "联邦学习", "对比学习", "自监督", "半监督", "迁移学习", "强化学习", "多任务学习",
    "注意力机制", "编解码", "序列到序列", "端到端", "多模态", "跨模态",
    "语音识别", "语音合成", "文本转语音", "语音增强", "声纹识别", "说话人分离",
    "目标检测", "语义分割", "实例分割", "图像生成", "图像修复", "超分辨率",
    "知识图谱", "命名实体识别", "关系抽取", "文本分类", "情感分析", "机器翻译",
    "搜索引擎", "推荐算法", "召回", "排序", "粗排", "精排", "重排",
    "负载均衡", "服务治理", "配置中心", "注册中心", "网关", "限流", "降级", "熔断",
    "微服务", "容器化", "持续集成", "持续部署", "灰度发布", "滚动更新",
    "流式计算", "批处理", "实时计算", "离线计算", "数据湖", "数据仓库",
}

DETAIL_HINT_WORDS = {
    "方案", "架构", "策略", "一致性", "分片", "限流", "降级", "熔断", "重试", "幂等", "旁路缓存", "ttl",
    "基线", "回归", "压测", "ab", "a/b", "trade-off", "瓶颈", "排查", "故障", "优化", "重构", "改造",
}

RESPONSIBILITY_WORDS = {
    "负责", "主导", "推动", "设计", "实现", "落地", "牵头", "独立", "协作", "owner", "led", "drove", "implemented",
}

ACTION_WORDS = {
    "设计", "实现", "开发", "优化", "重构", "建设", "搭建", "分析", "排查", "推进", "上线", "发布",
    "built", "designed", "implemented", "optimized", "improved", "migrated", "deployed",
}

VALID_AUDIT_DIMENSIONS = {
    "technical_depth",
    "quantification",
    "responsibility_clarity",
    "authenticity",
}

VALID_SEVERITIES = {"high", "medium", "low"}
SEVERITY_PRIORITY = {"high": 0, "medium": 1, "low": 2}

_FACE_CASCADE = None
if cv2 is not None:
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        _FACE_CASCADE = None if cascade.empty() else cascade
    except Exception:
        _FACE_CASCADE = None

if ENABLE_AVATAR_FACE_SCORE and _FACE_CASCADE is None:
    logger.info("Avatar face scoring disabled: OpenCV cascade unavailable (install opencv-python-headless + numpy)")

GENERIC_BULLET_HINTS = [
    "显著提升了模型性能",
    "提高了模型的准确性和可靠性",
    "取得了显著效果",
    "推进了自动驾驶感知领域的发展",
    "提升了模型的鲁棒性和泛化能力",
    "通过多模态数据融合和深度学习技术",
    # Extended: more Chinese AI/vague patterns
    "大幅改善了",
    "有效优化了",
    "成功落地了",
    "显著改善",
    "明显提升",
    "大幅提升",
    "有效提升",
    "整体优化",
    "全面优化",
    "性能显著提高",
    "效率显著提升",
    "取得了良好效果",
    "效果良好",
    "通过验证",
    "顺利上线",
    "成功交付",
    "完成了系统开发",
    "参与了项目开发",
    "负责日常维护",
    "配合团队完成",
    "完成了需求开发",
]

TECH_ANCHOR_STOPWORDS = {
    "and", "the", "for", "with", "this", "that", "from", "into", "using", "used",
    "model", "models", "framework", "method", "baseline", "dataset", "performance",
}

GENERIC_AUDIT_PATTERNS = [
    "未详细描述",
    "缺乏具体",
    "未明确个人职责",
    "技术方案和设计细节",
    "量化结果和基线对比",
]

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?above",
    r"forget\s+(everything|all)",
    r"new\s+instructions?:",
    r"system\s*:",
    r"<\s*/?\s*system\s*>",
    r"\[\s*inst\s*\]",
    r"\[\s*/\s*inst\s*\]",
]

AI_PHRASE_REPLACEMENTS = {
    # English AI jargon
    "spearheaded": "led",
    "orchestrated": "coordinated",
    "leveraged": "used",
    "utilized": "used",
    "synergy": "collaboration",
    "best-in-class": "high-quality",
    "world-class": "high-quality",
    "cutting-edge": "modern",
    "impactful": "effective",
    "in order to": "to",
    "at this point in time": "now",
    "---": ", ",
    "--": ", ",
    # Chinese AI jargon
    "赋能": "支持",
    "助力": "帮助",
    "打通链路": "连接",
    "降本增效": "降低成本",
    "提质增效": "提升效率",
    "全方位": "全面",
    "一站式": "统一",
    "闭环": "完整流程",
    "抓手": "方法",
    "拉齐": "对齐",
    "对焦": "聚焦",
    "沉淀": "积累",
    "颗粒度": "细节",
    "底层逻辑": "基本原理",
    "顶层设计": "整体设计",
    "组合拳": "方案",
    "打法": "策略",
}

GENERIC_MARKERS = [
    "显著", "明显效果", "提高了", "提升了", "推进了", "先进", "创新",
    "大幅", "有效", "成功落地", "良好效果", "顺利", "全面", "整体",
    "优化效果", "改善效果", "取得成效", "较好效果", "明显改善",
]


def llm_enabled() -> bool:
    return bool(OpenAI is not None and API_KEY and API_BASE_URL)


def get_client() -> OpenAI:
    global _client
    if not llm_enabled():
        raise RuntimeError("OpenAI SDK, MODELHUB_API_KEY or MODELHUB_BASE_URL is not configured")
    if _client is None:
        # SDK retries happen below the application deadline and cannot be
        # cancelled by ``asyncio.to_thread``. Keep retry policy in our gateway,
        # where every additional call is checked against the request budget.
        _client = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE_URL,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=0,
        )
    return _client


def get_llm_gateway() -> LLMGateway:
    global _llm_gateway
    if _llm_gateway is None:
        _llm_gateway = LLMGateway(
            client_factory=get_client,
            model_name=MODEL_NAME,
            logger=logger,
            enable_json_repair=ENABLE_LLM_JSON_REPAIR,
            call_timeout_seconds=LLM_TIMEOUT_SECONDS,
            request_time_remaining=remaining_request_seconds,
            deadline_reserve_seconds=LLM_DEADLINE_RESERVE_SECONDS,
            retry_min_remaining_seconds=LLM_RETRY_MIN_REMAINING_SECONDS,
            dump_failure_payload=lambda tag, raw_content, error: _dump_llm_failure_payload(
                tag=tag,
                raw_content=raw_content,
                error=error,
            ),
        )
    return _llm_gateway


def sanitize_user_text(text: str) -> str:
    cleaned = text
    for pattern in INJECTION_PATTERNS:
        cleaned = re.sub(pattern, "[REDACTED]", cleaned, flags=re.IGNORECASE)
    return cleaned


def _dump_llm_failure_payload(tag: str, raw_content: str, error: str) -> None:
    if not ENABLE_LLM_FAILURE_DUMP:
        return
    def _redact(text: str) -> str:
        text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
        text = re.sub(r"(?:\+?86[- ]?)?1[3-9]\d{9}", "[PHONE]", text)
        return text[:4000]

    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(tag or "unknown"))[:80] or "unknown"
        path = LLM_FAILURE_DUMP_DIR / f"{ts}_{safe_tag}.json"
        payload = {
            "tag": tag,
            "error": error,
            "raw_content": _redact(str(raw_content or "")),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.warning("Saved malformed LLM payload to %s", path)
    except Exception:
        logger.warning("Failed to dump malformed LLM payload", exc_info=True)



def call_llm_typed(
    output_model: type[BaseModel],
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    prefill: str = "{",
) -> dict[str, Any]:
    with _llm_inflight_slot():
        return get_llm_gateway().call_typed(
            output_model=output_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            prefill=prefill,
        )


def call_llm_text(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> str:
    """Call LLM and return raw text. No JSON parsing or schema injection."""
    with _llm_inflight_slot():
        return get_llm_gateway().call_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
