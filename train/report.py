"""growth.jsonl → 1枚のHTML（成長タイムライン）
  python report.py out/tiny                 → out/tiny/report.html
  python report.py out/nano out/tiny out/small -o compare.html   （サイズ違いを切り替えて比較）
外部ファイル不要・オフラインで開ける。
"""
import argparse
import json
import os


def load_run(d):
    info = json.load(open(os.path.join(d, "run.json"), encoding="utf-8"))
    growth = [json.loads(l) for l in open(os.path.join(d, "growth.jsonl"), encoding="utf-8") if l.strip()]
    loss = []
    lp = os.path.join(d, "loss.jsonl")
    if os.path.exists(lp):
        loss = [json.loads(l) for l in open(lp) if l.strip()]
    qp = os.path.join(d, "quant.json")
    quant = json.load(open(qp, encoding="utf-8")) if os.path.exists(qp) else None
    return {"name": info["run"], "params": info["params"], "config": info["config"],
            "data": info.get("data", {}), "growth": growth, "loss": loss, "quant": quant}


HTML = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f7f6f2;--panel:#fff;--ink:#1d1d1b;--mute:#6b6a65;--line:#e3e1da;--acc:#c2410c;--acc2:#0f766e;--chip:#f1efe8;--bar:#fed7aa}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151514;--panel:#1e1e1c;--ink:#ecebe6;--mute:#9b9990;--line:#33322e;--acc:#fb923c;--acc2:#2dd4bf;--chip:#2a2926;--bar:#7c2d12}}
:root[data-theme="dark"]{--bg:#151514;--panel:#1e1e1c;--ink:#ecebe6;--mute:#9b9990;--line:#33322e;--acc:#fb923c;--acc2:#2dd4bf;--chip:#2a2926;--bar:#7c2d12}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 "Hiragino Sans","Noto Sans JP",system-ui,sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--mute);font-size:13px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:16px}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
select,button{font:inherit;background:var(--chip);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:4px 10px}
input[type=range]{flex:1;min-width:200px;accent-color:var(--acc)}
.big{font-variant-numeric:tabular-nums;font-weight:700;font-size:20px}
.k{color:var(--mute);font-size:12px}
svg text{fill:var(--mute);font-size:11px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px;margin-top:12px}
.card{border:1px solid var(--line);border-radius:10px;padding:12px;background:var(--bg)}
.prompt{font-size:12px;color:var(--mute)}
.gen{white-space:pre-wrap;word-break:break-all;margin:4px 0}
.gen b{color:var(--acc)}
.samp{color:var(--acc2)}
.cands{display:flex;gap:4px;flex-wrap:wrap;margin-top:6px}
.cand{background:var(--chip);border-radius:6px;padding:2px 6px;font-size:11px;line-height:1.4}
.cand .p{display:block;height:3px;background:var(--acc);border-radius:2px;margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{color:var(--mute);font-weight:500;white-space:nowrap}
tr.sel td{background:var(--chip)}
td.step{font-variant-numeric:tabular-nums;white-space:nowrap;color:var(--mute)}
td.txt{white-space:pre-wrap;word-break:break-all}
.tabs button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
</style></head><body><main>
<h1>__TITLE__</h1>
<div class="sub" id="meta"></div>

<div class="panel">
  <div class="row">
    <label class="k">モデル <select id="run"></select></label>
    <span class="k">チェックポイント</span>
    <input type="range" id="slider" min="0" value="0">
    <button id="play">▶ 再生</button>
  </div>
  <div class="row" style="margin-top:8px">
    <div><div class="k">step</div><div class="big" id="step"></div></div>
    <div><div class="k">学習トークン</div><div class="big" id="tok"></div></div>
    <div><div class="k">val loss</div><div class="big" id="vl"></div></div>
  </div>
  <svg id="chart" width="100%" height="180" viewBox="0 0 1000 180" preserveAspectRatio="none"></svg>
</div>

<div class="panel" id="summary" style="display:none">
  <div class="k" style="margin-bottom:6px">モデル比較（最終チェックポイント・val loss は小さいほど良い。int8 は Core2 の Q8 推論と同じ計算）</div>
  <div style="overflow-x:auto"><table id="sumtab"></table></div>
  <div id="qgen" style="margin-top:12px"></div>
</div>

<div class="panel">
  <div class="row tabs"><button id="tabCards" class="on">このチェックポイントの全問いかけ</button><button id="tabTimeline">1つの問いかけの成長</button>
  <select id="promptSel" style="display:none"></select></div>
  <div id="cardsView"></div>
  <div id="timelineView" style="display:none"></div>
</div>
<p class="sub">太字＝問いかけへの返答（greedy: 一番確からしい語を選び続けた結果）。青緑＝温度0.8でサンプリングした結果。小さなチップは最初の数トークンでモデルが迷った候補と確率。</p>
</main>
<script>
const RUNS = __DATA__;
const $ = id => document.getElementById(id);
let run = RUNS[0], idx = 0, timer = null, view = "cards";
const esc = s => String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const fmt = n => n >= 1e6 ? (n/1e6).toFixed(1)+"M" : n >= 1e3 ? (n/1e3).toFixed(0)+"K" : String(n);
const pshow = p => p === "" ? "（何も与えない）" : p;
const firstTurn = s => { const i = s.indexOf("Ａ："); return i > 0 ? s.slice(0, i) : s; };

RUNS.forEach((r, i) => { const o = document.createElement("option"); o.value = i; o.textContent = `${r.name} (${fmt(r.params)} params)`; $("run").appendChild(o); });
$("run").onchange = () => { run = RUNS[+$("run").value]; setup(); };
$("tabCards").onclick = () => { view = "cards"; render(); };
$("tabTimeline").onclick = () => { view = "timeline"; render(); };
$("promptSel").onchange = render;
$("slider").oninput = () => { idx = +$("slider").value; render(); };
$("play").onclick = () => {
  if (timer) { clearInterval(timer); timer = null; $("play").textContent = "▶ 再生"; return; }
  if (idx >= run.growth.length - 1) idx = 0;
  $("play").textContent = "■ 停止";
  timer = setInterval(() => { idx++; if (idx >= run.growth.length) { clearInterval(timer); timer = null; $("play").textContent = "▶ 再生"; idx = run.growth.length - 1; } $("slider").value = idx; render(); }, 900);
};

function summary() {
  const has = RUNS.some(r => r.quant);
  if (RUNS.length < 2 && !has) return;
  $("summary").style.display = "";
  const f = v => v == null ? "-" : v.toFixed(3);
  const mb = v => v == null ? "-" : (v / 1e6).toFixed(2) + "MB";
  let h = `<tr><th>モデル</th><th>パラメータ</th><th>語彙</th><th>val loss fp32</th><th>int8</th><th>int4(重み)</th><th>サイズ fp32 → int8</th></tr>`;
  RUNS.forEach(r => {
    const q = r.quant || {}, last = r.growth[r.growth.length - 1];
    h += `<tr><td>${esc(r.name)}</td><td class="step">${fmt(r.params)}</td><td class="step">${r.config.vocab_size}</td>
      <td class="step">${f(q.loss_fp32 ?? last.val_loss)}</td><td class="step">${f(q.loss_int8)}</td><td class="step">${f(q.loss_int4w)}</td>
      <td class="step">${mb(q.size_fp32)} → ${mb(q.size_q8)}</td></tr>`;
  });
  $("sumtab").innerHTML = h;
}

function qgen() {
  const q = run.quant;
  if (!q) { $("qgen").innerHTML = ""; return; }
  $("qgen").innerHTML = `<div class="k">${esc(run.name)}: 同じ問いかけの greedy 生成（fp32 / int8）</div><table><tr><th>問いかけ</th><th>fp32</th><th>int8</th></tr>` +
    q.generations.map(g => `<tr><td class="prompt">${esc(pshow(g.prompt))}</td><td class="txt">${esc(firstTurn(g.fp32))}</td><td class="txt samp">${esc(firstTurn(g.int8))}</td></tr>`).join("") + `</table>`;
}

function setup() {
  qgen();
  const c = run.config;
  $("meta").textContent = `${run.name} — dim ${c.dim} / ${c.n_layers}層 / heads ${c.n_heads} (kv ${c.n_kv_heads}) / 語彙 ${c.vocab_size} / ${fmt(run.params)} params / 学習データ ${fmt(run.data.train_tokens || 0)} tokens`;
  $("slider").max = run.growth.length - 1;
  idx = Math.min(idx, run.growth.length - 1); $("slider").value = idx;
  $("promptSel").innerHTML = "";
  run.growth[0].samples.forEach((s, i) => { const o = document.createElement("option"); o.value = i; o.textContent = pshow(s.prompt); $("promptSel").appendChild(o); });
  if (run.growth[0].samples.length > 1) $("promptSel").value = 1;
  render();
}

function chart() {
  const g = run.growth, L = run.loss, W = 1000, H = 180, P = 28;
  const steps = g.map(x => x.step).concat(L.map(x => x.step)).filter(s => s > 0);
  const maxS = Math.max(...steps, 2), ys = g.map(x => x.val_loss).concat(L.map(x => x.loss)).filter(v => v != null);
  const yMax = Math.max(...ys), yMin = Math.min(...ys);
  const X = s => P + (Math.log10(Math.max(s, 1)) / Math.log10(maxS)) * (W - 2*P);
  const Y = v => H - P + 8 - ((v - yMin) / ((yMax - yMin) || 1)) * (H - 2*P);
  let s = "";
  for (let d = 1; d <= maxS; d *= 10) s += `<line x1="${X(d)}" x2="${X(d)}" y1="8" y2="${H-P+8}" stroke="var(--line)"/><text x="${X(d)+3}" y="${H-6}">${fmt(d)}</text>`;
  s += `<text x="${P}" y="18">${yMax.toFixed(2)}</text><text x="${P}" y="${H-P}">${yMin.toFixed(2)}</text>`;
  if (L.length) s += `<polyline fill="none" stroke="var(--line)" stroke-width="1.5" points="${L.filter(x=>x.step>0).map(x => X(x.step)+","+Y(x.loss)).join(" ")}"/>`;
  s += `<polyline fill="none" stroke="var(--acc)" stroke-width="2.5" points="${g.map(x => X(x.step)+","+Y(x.val_loss)).join(" ")}"/>`;
  g.forEach((x, i) => s += `<circle cx="${X(x.step)}" cy="${Y(x.val_loss)}" r="${i===idx?7:3.5}" fill="${i===idx?"var(--acc)":"var(--panel)"}" stroke="var(--acc)" stroke-width="2" style="cursor:pointer" onclick="idx=${i};$('slider').value=${i};render()"/>`);
  s += `<text x="${W-P}" y="18" text-anchor="end">val loss（太線・●）／ train loss（細線）</text>`;
  $("chart").innerHTML = s;
}

function candHTML(c) {
  return `<div class="cands">${c.map(step => `<span class="cand">${step.slice(0,3).map(([t,p]) => `${esc(JSON.stringify(t).slice(1,-1))} ${Math.round(p*100)}%`).join(" · ")}<span class="p" style="width:${Math.round(step[0][1]*100)}%"></span></span>`).join("")}</div>`;
}

function render() {
  const r = run.growth[idx];
  $("step").textContent = fmt(r.step); $("tok").textContent = fmt(r.tokens_seen); $("vl").textContent = r.val_loss.toFixed(3);
  $("tabCards").classList.toggle("on", view === "cards"); $("tabTimeline").classList.toggle("on", view === "timeline");
  $("promptSel").style.display = view === "timeline" ? "" : "none";
  $("cardsView").style.display = view === "cards" ? "" : "none";
  $("timelineView").style.display = view === "timeline" ? "" : "none";
  chart();
  if (view === "cards") {
    $("cardsView").innerHTML = `<div class="cards">${r.samples.map(s => `<div class="card"><div class="prompt">${esc(pshow(s.prompt))}</div>
      <div class="gen"><b>${esc(firstTurn(s.greedy)) || "（無言）"}</b>${esc(s.greedy.slice(firstTurn(s.greedy).length))}</div>
      <div class="gen samp">${esc(s.sampled) || "（無言）"}</div>${candHTML(s.cands || [])}</div>`).join("")}</div>`;
  } else {
    const p = +$("promptSel").value;
    $("timelineView").innerHTML = `<table><tr><th>step</th><th>val</th><th>greedy</th><th>サンプリング</th></tr>${run.growth.map((g, i) => `<tr class="${i===idx?"sel":""}" onclick="idx=${i};$('slider').value=${i};render()"><td class="step">${fmt(g.step)}</td><td class="step">${g.val_loss.toFixed(2)}</td><td class="txt">${esc(firstTurn(g.samples[p].greedy))}</td><td class="txt samp">${esc(firstTurn(g.samples[p].sampled))}</td></tr>`).join("")}</table>`;
  }
}
summary();
setup();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("-o", "--out")
    args = ap.parse_args()
    runs = [load_run(d) for d in args.runs]
    out = args.out or os.path.join(args.runs[0], "report.html")
    title = "育っていく極小LLM" if len(runs) == 1 else "極小LLM サイズ比較"
    data = json.dumps(runs, ensure_ascii=False).replace("</", "<\\/")
    with open(out, "w", encoding="utf-8") as f:
        f.write(HTML.replace("__TITLE__", title).replace("__DATA__", data))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
