"""全局视觉：暗色极简 / 瑞士编辑式排版。

纯黑底、白字、灰次级文字；Noto Sans SC 巨型标题；细线分隔；无圆角无阴影；
等宽大数字。通过一次 CSS 注入覆盖 Streamlit 默认组件样式。
"""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;700;900&family=JetBrains+Mono:wght@400;700&display=swap');
:root{--ink:#f2f2f2;--muted:#8a8a8a;--line:#262626;--bg:#000;}
html,body,.stApp,.stApp p,.stApp li,.stApp label,.stApp input,.stApp button p{font-family:"Noto Sans SC",-apple-system,sans-serif;}
/* Material 图标字体必须保留，否则图标名以文字泄漏 */
.stApp [data-testid="stIconMaterial"]{font-family:"Material Symbols Rounded"!important;font-variation-settings:'FILL' 0,'wght' 400,'GRAD' 0,'opsz' 24;}
.stApp{background:var(--bg);color:var(--ink);}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:2.5rem;padding-bottom:6rem;max-width:1200px;}

/* 标题 */
h1,h2,h3{font-family:"Noto Sans SC"!important;letter-spacing:-0.03em!important;color:var(--ink)!important;}
h1{font-weight:900!important;font-size:clamp(2.2rem,5vw,4.2rem)!important;line-height:1.05!important;margin:0 0 .6rem 0!important;}
h2{font-weight:700!important;font-size:1.9rem!important;padding:1.6rem 0 .4rem 0!important;border-top:1px solid var(--line);}
h3{font-weight:700!important;font-size:1.15rem!important;color:var(--ink)!important;}
p,li{color:#d6d6d6;line-height:1.75;}
.stCaption,small,[data-testid="stCaptionContainer"]{color:var(--muted)!important;}
hr{border:0;border-top:1px solid var(--line)!important;margin:1.6rem 0!important;}

/* 编辑式 hero */
.ae-hero{padding:1rem 0 2.2rem 0;border-bottom:1px solid var(--line);margin-bottom:1.6rem;}
.ae-kicker{font-family:"JetBrains Mono",monospace;font-size:.72rem;letter-spacing:.18em;color:var(--muted);text-transform:uppercase;margin-bottom:1.2rem;}
.ae-title{font-weight:900;font-size:clamp(2.4rem,5.4vw,4.6rem);line-height:1.04;letter-spacing:-0.035em;color:var(--ink);margin:0;}
.ae-abstract{max-width:52rem;color:#bdbdbd;font-size:1.05rem;line-height:1.7;margin-top:1.4rem;}

/* 左标签 / 右内容 */
.ae-label{font-family:"JetBrains Mono",monospace;font-size:.72rem;letter-spacing:.16em;color:var(--muted);text-transform:uppercase;padding-top:.45rem;position:sticky;top:4rem;}
.ae-label b{display:block;color:var(--ink);font-size:1.6rem;font-weight:700;letter-spacing:-0.02em;margin-bottom:.2rem;font-family:"JetBrains Mono",monospace;}

/* 指标：等宽大数字 */
[data-testid="stMetric"]{background:transparent!important;border:0!important;border-top:1px solid var(--line)!important;border-radius:0!important;padding:.9rem 0 0 0!important;}
[data-testid="stMetricLabel"] p{font-family:"JetBrains Mono",monospace!important;font-size:.7rem!important;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)!important;}
[data-testid="stMetricValue"]{font-family:"JetBrains Mono",monospace!important;font-weight:700!important;font-size:2.1rem!important;letter-spacing:-0.02em;color:var(--ink)!important;}
[data-testid="stMetricDelta"]{font-family:"JetBrains Mono",monospace!important;font-size:.75rem!important;color:var(--muted)!important;}
[data-testid="stMetricDelta"] svg{display:none;}

/* 控件：方角、细线 */
.stApp button[data-testid="stBaseButton-secondary"],.stApp button[data-testid="stDownloadButton"],.stApp .stDownloadButton button{border-radius:0!important;border:1px solid var(--ink)!important;background:transparent!important;color:var(--ink)!important;font-weight:700!important;letter-spacing:.06em;padding:.7rem 1.4rem!important;transition:all .15s;}
.stApp button[data-testid="stBaseButton-primary"]{border-radius:0!important;border:1px solid var(--ink)!important;background:var(--ink)!important;color:#000!important;font-weight:700!important;letter-spacing:.06em;padding:.8rem 1.8rem!important;transition:all .15s;}
.stApp button[data-testid="stBaseButton-primary"] p,.stApp button[data-testid="stBaseButton-primary"] span{color:#000!important;}
.stApp button[data-testid="stBaseButton-primary"]:hover{background:#d9d9d9!important;transform:translateX(4px);}
.stApp button[data-testid="stBaseButton-secondary"]:hover,.stApp .stDownloadButton button:hover{transform:translateX(4px);background:#111!important;}
[data-testid="stFileUploader"] section{border:1px dashed #444!important;border-radius:0!important;background:transparent!important;}
[data-baseweb="input"],[data-baseweb="select"]>div,.stNumberInput div[data-baseweb]{border-radius:0!important;background:#0a0a0a!important;border-color:#333!important;}
div[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:0!important;background:transparent!important;}
[data-testid="stAlert"]{border-radius:0!important;border-left-width:3px!important;background:#0a0a0a!important;}
[data-testid="stDataFrame"]{border:1px solid var(--line);}
div[data-testid="stTable"] table{border-collapse:collapse;}
div[data-testid="stTable"] td,div[data-testid="stTable"] th{border:0!important;border-bottom:1px solid var(--line)!important;padding:.7rem .4rem!important;}
div[data-testid="stTable"] th{font-family:"JetBrains Mono",monospace;font-size:.7rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)!important;}

/* 侧栏 */
section[data-testid="stSidebar"]{background:#000!important;border-right:1px solid var(--line);}
section[data-testid="stSidebar"] .stRadio label p{font-size:.92rem;color:#cfcfcf;transition:all .15s;}
section[data-testid="stSidebar"] .stRadio label:hover p{color:#fff;transform:translateX(3px);}
section[data-testid="stSidebar"] [role="radiogroup"] label{padding:.32rem 0;border-bottom:1px solid #141414;}
section[data-testid="stSidebar"] h3{font-family:"JetBrains Mono",monospace!important;font-size:.72rem!important;letter-spacing:.18em;text-transform:uppercase;color:var(--muted)!important;font-weight:400!important;}

/* 主体单选 = 水平 tab 感 */
.main [role="radiogroup"]{gap:1.4rem;}
code{font-family:"JetBrains Mono",monospace!important;background:#111!important;color:#ddd!important;border-radius:0!important;}

/* 隐藏 Streamlit 自带多页导航（sections 已改名，双保险） */
[data-testid="stSidebarNav"]{display:none;}
/* 滚动淡入 */
@keyframes ae-in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.block-container>div{animation:ae-in .5s ease both;}
html{scroll-behavior:smooth;}
</style>
"""


def inject() -> None:
    st.html(_CSS)


def hero(kicker: str, title: str, abstract: str) -> None:
    st.html(
        f'<div class="ae-hero"><div class="ae-kicker">{kicker}</div>'
        f'<div class="ae-title">{title}</div>'
        f'<div class="ae-abstract">{abstract}</div></div>'
    )


def row(number: str, label: str):
    """瑞士式两栏：左侧粘性编号+标签，右侧内容容器。用法 `with row("01","上传"):`。"""
    left, right = st.columns([1, 4], gap="large")
    with left:
        st.html(f'<div class="ae-label"><b>{number}</b>{label}</div>')
    return right
