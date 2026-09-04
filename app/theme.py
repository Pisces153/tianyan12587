"""全局视觉：暗色产品化界面。

系统字体栈（不依赖外网字体）、顶部导航、卡片网格、等宽大数字、克制的
单一强调色。通过一次 CSS 注入覆盖 Streamlit 默认组件样式。
"""

from __future__ import annotations

from html import escape

import streamlit as st

_CSS = """
<style>
:root{
  --ink:#f4f4f5;--body:#c9c9cf;--muted:#8b8b93;--line:#222226;--line-2:#2e2e33;
  --bg:#09090b;--surface:#0f0f12;--surface-2:#15151a;--acc:#5b9cff;--acc-ink:#0b1a33;
  --good:#2fbf71;--warn:#f2b83b;--crit:#ef5b5b;
  --sans:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei","Segoe UI","Helvetica Neue",Arial,"Noto Sans SC",sans-serif;
  --mono:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,"JetBrains Mono","Liberation Mono",monospace;
}
html,body,.stApp,.stApp p,.stApp li,.stApp label,.stApp input,.stApp button p,.stApp div{font-family:var(--sans);}
.stApp [data-testid="stIconMaterial"]{font-family:"Material Symbols Rounded"!important;font-variation-settings:'FILL' 0,'wght' 400,'GRAD' 0,'opsz' 24;}
.stApp{background:var(--bg);color:var(--ink);}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:4.6rem;padding-bottom:5rem;max-width:1180px;}

/* 顶栏 + 顶部导航 */
header[data-testid="stHeader"]{background:rgba(9,9,11,.86)!important;backdrop-filter:blur(10px);border-bottom:1px solid var(--line);}
[data-testid="stTopNavSection"]{gap:.15rem;}
[data-testid="stTopNavLink"],[data-testid="stTopNavLink"] span,[data-testid="stTopNavLink"] p{font-family:var(--sans)!important;font-size:.92rem!important;font-weight:500!important;color:var(--muted)!important;letter-spacing:.01em;}
[data-testid="stTopNavLink"]{border-radius:8px!important;padding:.35rem .8rem!important;transition:all .15s;}
[data-testid="stTopNavLink"]:hover{background:var(--surface-2)!important;}
[data-testid="stTopNavLink"]:hover span,[data-testid="stTopNavLink"]:hover p{color:var(--ink)!important;}
[data-testid="stTopNavLink"][aria-current="page"],[data-testid="stTopNavLink"][aria-current="page"] span,[data-testid="stTopNavLink"][aria-current="page"] p{color:var(--ink)!important;font-weight:600!important;}
[data-testid="stTopNavLink"][aria-current="page"]{background:var(--surface-2)!important;box-shadow:inset 0 -2px 0 var(--acc);}
[data-testid="stHeaderLogo"] img,[data-testid="stLogo"]{height:1.6rem!important;max-height:1.6rem!important;}
[data-testid="stSidebarNav"],[data-testid="stSidebarCollapsedControl"]{display:none;}

/* 文字层级 */
h1,h2,h3,h4,h5{font-family:var(--sans)!important;color:var(--ink)!important;letter-spacing:-0.02em!important;}
h1{font-weight:800!important;font-size:clamp(2rem,4.2vw,3.2rem)!important;line-height:1.1!important;margin:0 0 .6rem 0!important;}
h2{font-weight:700!important;font-size:1.55rem!important;padding:1.4rem 0 .3rem 0!important;}
h3{font-weight:650!important;font-size:1.08rem!important;padding-top:.6rem!important;}
p,li{color:var(--body);line-height:1.75;font-size:.98rem;}
.stCaption,small,[data-testid="stCaptionContainer"] p{color:var(--muted)!important;font-size:.82rem!important;}
hr{border:0;border-top:1px solid var(--line)!important;margin:1.4rem 0!important;}
code{font-family:var(--mono)!important;background:var(--surface-2)!important;color:#dfe3ea!important;border-radius:5px!important;font-size:.85em!important;}
[data-testid="stCode"] pre,[data-testid="stCode"]{background:var(--surface)!important;border:1px solid var(--line)!important;border-radius:10px!important;}

/* 首页 hero */
.ae-hero{padding:1.2rem 0 1.4rem 0;}
.ae-kicker{font-family:var(--mono);font-size:.72rem;letter-spacing:.16em;color:var(--muted);text-transform:uppercase;margin-bottom:1.1rem;display:flex;align-items:center;gap:.6rem;}
.ae-kicker::before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--good);box-shadow:0 0 0 4px rgba(47,191,113,.18);}
.ae-title{font-weight:800;font-size:clamp(2.3rem,5vw,4.1rem);line-height:1.06;letter-spacing:-0.03em;color:var(--ink);margin:0;max-width:18em;}
.ae-title em{font-style:normal;color:var(--acc);}
.ae-lead{max-width:44rem;color:var(--body);font-size:1.08rem;line-height:1.7;margin-top:1.3rem;}

/* 页头（非首页） */
.ae-page{padding:.4rem 0 .6rem 0;border-bottom:1px solid var(--line);margin-bottom:1rem;}
.ae-page .ae-kicker::before{display:none;}
.ae-page h1{font-size:clamp(1.7rem,3vw,2.3rem)!important;}
.ae-page .ae-lead{font-size:1rem;margin-top:.6rem;max-width:52rem;}

/* CTA：page_link 做成按钮 */
[data-testid="stPageLink-NavLink"]{border-radius:10px!important;padding:.7rem 1.25rem!important;border:1px solid var(--line-2)!important;background:var(--surface)!important;transition:all .15s;}
[data-testid="stPageLink-NavLink"] p,[data-testid="stPageLink-NavLink"] span{color:var(--ink)!important;font-weight:600!important;font-size:.95rem!important;}
[data-testid="stPageLink-NavLink"]:hover{border-color:#3a3a42!important;background:var(--surface-2)!important;transform:translateY(-1px);}
.st-key-cta_primary [data-testid="stPageLink-NavLink"]{background:var(--acc)!important;border-color:var(--acc)!important;}
.st-key-cta_primary [data-testid="stPageLink-NavLink"] p,.st-key-cta_primary [data-testid="stPageLink-NavLink"] span{color:var(--acc-ink)!important;}
.st-key-cta_primary [data-testid="stPageLink-NavLink"]:hover{background:#79aeff!important;}

/* 统计条 */
.ae-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:12px;overflow:hidden;margin:1.6rem 0 .4rem 0;}
.ae-stat{background:var(--surface);padding:1.1rem 1.2rem;}
.ae-stat .k{font-family:var(--mono);font-size:.68rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);}
.ae-stat .v{font-family:var(--mono);font-size:1.75rem;font-weight:700;color:var(--ink);margin:.35rem 0 .15rem 0;letter-spacing:-0.02em;}
.ae-stat .d{font-size:.8rem;color:var(--muted);}

/* 卡片 */
.ae-card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:1.2rem 1.25rem 1rem 1.25rem;height:100%;display:flex;flex-direction:column;transition:border-color .15s,transform .15s;}
.ae-card .b{flex:1;}
.ae-card a{display:inline-block;margin-top:.9rem;color:var(--acc);font-size:.88rem;font-weight:600;text-decoration:none;}
.ae-card a:hover{text-decoration:underline;}
.ae-card:hover{border-color:var(--line-2);transform:translateY(-2px);}
.ae-card .n{font-family:var(--mono);font-size:.7rem;letter-spacing:.14em;color:var(--muted);text-transform:uppercase;display:flex;justify-content:space-between;align-items:center;}
.ae-card .t{font-size:1.08rem;font-weight:650;color:var(--ink);margin:.7rem 0 .4rem 0;letter-spacing:-0.01em;}
.ae-card .b{font-size:.9rem;color:var(--body);line-height:1.6;min-height:7.2em;}
.ae-tag{font-family:var(--mono);font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;padding:.18rem .5rem;border-radius:999px;border:1px solid;}
.ae-tag.live{color:var(--good);border-color:rgba(47,191,113,.45);background:rgba(47,191,113,.08);}
.ae-tag.ref{color:var(--warn);border-color:rgba(242,184,59,.45);background:rgba(242,184,59,.08);}
.ae-tag.info{color:var(--acc);border-color:rgba(91,156,255,.45);background:rgba(91,156,255,.08);}

/* 流水线 */
.ae-pipe{display:flex;flex-wrap:wrap;gap:.45rem;margin:.6rem 0 .2rem 0;}
.ae-step{display:inline-flex;align-items:center;gap:.5rem;padding:.42rem .7rem .42rem .5rem;border:1px solid var(--line);border-radius:999px;background:var(--surface);font-size:.85rem;color:var(--body);}
.ae-step b{font-family:var(--mono);font-size:.68rem;color:var(--acc-ink);background:var(--acc);border-radius:999px;padding:.12rem .42rem;font-weight:700;}

/* 状态块 */
.ae-status{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:.8rem;margin-top:.6rem;}
.ae-status .s{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:1rem 1.1rem;}
.ae-status .k{font-family:var(--mono);font-size:.68rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);}
.ae-status .v{font-family:var(--mono);font-size:.95rem;color:var(--ink);margin:.4rem 0 .35rem 0;word-break:break-all;}
.ae-status .d{font-size:.85rem;color:var(--muted);line-height:1.55;}

/* 左标签 / 右内容（工作台步骤） */
.ae-label{font-family:var(--mono);font-size:.7rem;letter-spacing:.14em;color:var(--muted);text-transform:uppercase;padding-top:.45rem;position:sticky;top:4.2rem;}
.ae-label b{display:block;color:var(--ink);font-size:1.5rem;font-weight:700;letter-spacing:-0.02em;margin-bottom:.15rem;font-family:var(--mono);}

/* 指标 */
[data-testid="stMetric"]{background:var(--surface)!important;border:1px solid var(--line)!important;border-radius:12px!important;padding:.9rem 1rem!important;}
[data-testid="stMetricLabel"] p{font-family:var(--mono)!important;font-size:.68rem!important;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)!important;}
[data-testid="stMetricValue"]{font-family:var(--mono)!important;font-weight:700!important;font-size:1.7rem!important;letter-spacing:-0.02em;color:var(--ink)!important;}
[data-testid="stMetricDelta"]{font-family:var(--mono)!important;font-size:.74rem!important;color:var(--muted)!important;}
[data-testid="stMetricDelta"] svg{display:none;}

/* 控件 */
.stApp button[data-testid="stBaseButton-secondary"],.stApp .stDownloadButton button{border-radius:10px!important;border:1px solid var(--line-2)!important;background:var(--surface)!important;color:var(--ink)!important;font-weight:600!important;padding:.6rem 1.2rem!important;transition:all .15s;}
.stApp button[data-testid="stBaseButton-primary"]{border-radius:10px!important;border:1px solid var(--acc)!important;background:var(--acc)!important;font-weight:700!important;padding:.65rem 1.4rem!important;transition:all .15s;}
.stApp button[data-testid="stBaseButton-primary"] p,.stApp button[data-testid="stBaseButton-primary"] span{color:var(--acc-ink)!important;}
.stApp button[data-testid="stBaseButton-primary"]:hover{background:#79aeff!important;transform:translateY(-1px);}
.stApp button[data-testid="stBaseButton-secondary"]:hover,.stApp .stDownloadButton button:hover{background:var(--surface-2)!important;transform:translateY(-1px);}
[data-testid="stFileUploader"] section{border:1px dashed #3a3a42!important;border-radius:12px!important;background:var(--surface)!important;}
[data-baseweb="input"],[data-baseweb="select"]>div,.stNumberInput div[data-baseweb]{border-radius:8px!important;background:var(--surface)!important;border-color:var(--line-2)!important;}
div[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:12px!important;background:var(--surface)!important;}
div[data-testid="stExpander"] summary{border-radius:12px!important;}
[data-testid="stAlert"]{border-radius:10px!important;border-left-width:3px!important;background:var(--surface)!important;}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:10px;overflow:hidden;}
div[data-testid="stTable"] table{border-collapse:collapse;}
div[data-testid="stTable"] td,div[data-testid="stTable"] th{border:0!important;border-bottom:1px solid var(--line)!important;padding:.65rem .5rem!important;}
div[data-testid="stTable"] th{font-family:var(--mono);font-size:.68rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)!important;}
.stTabs [data-baseweb="tab-list"]{gap:.2rem;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{padding:.6rem 1rem;border-radius:8px 8px 0 0;}
.stTabs [data-baseweb="tab"] p{font-size:.95rem!important;font-weight:600!important;color:var(--muted)!important;}
.stTabs [aria-selected="true"] p{color:var(--ink)!important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--acc)!important;}
.stTabs [data-baseweb="tab-border"]{display:none;}
.stRadio [role="radiogroup"]{gap:1rem;}

/* 进入动画 */
@keyframes ae-in{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.block-container>div{animation:ae-in .35s ease both;}
html{scroll-behavior:smooth;}
</style>
"""


def inject() -> None:
    st.html(_CSS)


def hero(kicker: str, title_html: str, lead: str) -> None:
    """首页 hero。title_html 允许 <em>/<br>。"""
    st.html(
        f'<div class="ae-hero"><div class="ae-kicker">{escape(kicker)}</div>'
        f'<div class="ae-title">{title_html}</div>'
        f'<div class="ae-lead">{escape(lead)}</div></div>'
    )


def page_title(kicker: str, title: str, lead: str = "") -> None:
    """非首页的页头。"""
    lead_html = f'<div class="ae-lead">{escape(lead)}</div>' if lead else ""
    st.html(
        f'<div class="ae-page"><div class="ae-kicker">{escape(kicker)}</div>'
        f"<h1>{escape(title)}</h1>{lead_html}</div>"
    )


def stats(items: list[tuple[str, str, str]]) -> None:
    """统计条：[(label, value, note), ...]"""
    cells = "".join(
        f'<div class="ae-stat"><div class="k">{escape(k)}</div><div class="v">{escape(v)}</div>'
        f'<div class="d">{escape(d)}</div></div>'
        for k, v, d in items
    )
    st.html(f'<div class="ae-stats">{cells}</div>')


def card(number: str, title: str, body: str, tag: str, tag_kind: str = "live",
         href: str | None = None, link: str = "") -> None:
    a = f'<a href="{escape(href)}" target="_self">{escape(link)} ↗</a>' if href else ""
    st.html(
        f'<div class="ae-card"><div class="n"><span>{escape(number)}</span>'
        f'<span class="ae-tag {tag_kind}">{escape(tag)}</span></div>'
        f'<div class="t">{escape(title)}</div><div class="b">{escape(body)}</div>{a}</div>'
    )


def pipeline(steps: list[str]) -> None:
    chips = "".join(
        f'<span class="ae-step"><b>{i:02d}</b>{escape(s)}</span>' for i, s in enumerate(steps, 1)
    )
    st.html(f'<div class="ae-pipe">{chips}</div>')


def status(items: list[tuple[str, str, str]]) -> None:
    cells = "".join(
        f'<div class="s"><div class="k">{escape(k)}</div><div class="v">{escape(v)}</div>'
        f'<div class="d">{escape(d)}</div></div>'
        for k, v, d in items
    )
    st.html(f'<div class="ae-status">{cells}</div>')


def row(number: str, label: str):
    """两栏：左侧粘性编号+标签，右侧内容容器。用法 `with row("01","上传"):`。"""
    left, right = st.columns([1, 4], gap="large")
    with left:
        st.html(f'<div class="ae-label"><b>{escape(number)}</b>{escape(label)}</div>')
    return right
