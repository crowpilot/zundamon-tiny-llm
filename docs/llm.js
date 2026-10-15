// llm.js — Core2 の llm_core.c（llama2.c 由来, MIT）と同じ計算をする JavaScript 版。int8(Q8) 形式専用。
// ブラウザでも Node でも動く（グローバル TinyLLM を定義）。
(function (root) {
  const Q8_MAGIC = 0x38515454;
  const MARKERS = ["Ａ：", "Ｂ："];

  function parseModel(buf) {
    const dv = new DataView(buf);
    if (dv.getInt32(0, true) !== Q8_MAGIC) throw new Error("int8(Q8) 形式のモデルではありません");
    const c = {};
    ["dim", "hidden_dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "seq_len"]
      .forEach((k, i) => (c[k] = dv.getInt32(8 + i * 4, true)));
    const gs = dv.getInt32(36, true);
    const L = c.n_layers, d = c.dim, hd = c.hidden_dim, hs = d / c.n_heads, kv = c.n_kv_heads * hs;
    let off = 40;
    const f32 = (n) => { const a = new Float32Array(buf, off, n); off += n * 4; return a; };
    const q8 = (n) => { const q = new Int8Array(buf, off, n); off += n; const s = new Float32Array(buf, off, n / gs); off += (n / gs) * 4; return { q, s }; };
    const w = { rms_att: f32(L * d), rms_ffn: f32(L * d), rms_final: f32(d) };
    w.tok_emb = q8(c.vocab_size * d);
    const per = (n) => Array.from({ length: L }, () => q8(n));
    w.wq = per(d * d); w.wk = per(d * kv); w.wv = per(d * kv); w.wo = per(d * d);
    w.w1 = per(d * hd); w.w2 = per(hd * d); w.w3 = per(d * hd);
    if (off !== buf.byteLength) throw new Error(`モデルの長さが合いません (${off} != ${buf.byteLength})`);
    return { cfg: c, gs, w };
  }

  function parseTokenizer(buf, vocabSize) {
    const dv = new DataView(buf), td = new TextDecoder("utf-8");
    let off = 4;
    const vocab = [], scores = new Float32Array(vocabSize), bytes = [];
    for (let i = 0; i < vocabSize; i++) {
      scores[i] = dv.getFloat32(off, true); off += 4;
      const len = dv.getInt32(off, true); off += 4;
      const b = new Uint8Array(buf, off, len); off += len;
      vocab.push(td.decode(b));
      const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(vocab[i]);
      bytes.push(m ? Uint8Array.of(parseInt(m[1], 16)) : new TextEncoder().encode(vocab[i]));
    }
    const lookup = new Map(vocab.map((p, i) => [p, i]));
    return { vocab, scores, bytes, lookup };
  }

  class Model {
    constructor(modelBuf, tokBuf, maxSteps) {
      const { cfg, gs, w } = parseModel(modelBuf);
      this.cfg = cfg; this.gs = gs; this.w = w;
      this.tok = parseTokenizer(tokBuf, cfg.vocab_size);
      this.T = Math.min(maxSteps || cfg.seq_len, cfg.seq_len);
      const d = cfg.dim, hd = cfg.hidden_dim, kv = (cfg.dim * cfg.n_kv_heads) / cfg.n_heads;
      this.kvDim = kv;
      this.x = new Float32Array(d); this.xb = new Float32Array(d); this.xb2 = new Float32Array(d);
      this.hb = new Float32Array(hd); this.hb2 = new Float32Array(hd); this.q = new Float32Array(d);
      this.kc = new Float32Array(cfg.n_layers * this.T * kv); this.vc = new Float32Array(cfg.n_layers * this.T * kv);
      this.att = new Float32Array(cfg.n_heads * this.T); this.logits = new Float32Array(cfg.vocab_size);
      const nmax = Math.max(d, hd);
      this.xq = new Int8Array(nmax); this.xs = new Float32Array(nmax / gs + 1);
      this.markerIds = MARKERS.map((m) => this.tok.lookup.get(m));
    }

    rmsnorm(o, x, wgt, woff, n) {
      let ss = 0; for (let j = 0; j < n; j++) ss += x[j] * x[j];
      ss = 1 / Math.sqrt(ss / n + 1e-5);
      for (let j = 0; j < n; j++) o[j] = wgt[woff + j] * (ss * x[j]);
    }

    // W(d,n) @ x(n)。x を gs 個ずつ int8 に丸め、整数の積和に倍率を掛ける（C 版と同じ）
    matmul(out, outOff, x, m, n, d) {
      const gs = this.gs, ng = n / gs, xq = this.xq, xs = this.xs;
      for (let g = 0; g < ng; g++) {
        let amax = 0;
        for (let j = 0; j < gs; j++) { const v = Math.abs(x[g * gs + j]); if (v > amax) amax = v; }
        const sc = amax > 0 ? amax / 127 : 1;
        xs[g] = sc;
        for (let j = 0; j < gs; j++) { const r = x[g * gs + j] / sc; xq[g * gs + j] = r < 0 ? -Math.round(-r) : Math.round(r); }  // C の roundf と同じ（0.5 は 0 から遠い側）
      }
      const q = m.q, s = m.s;
      for (let i = 0; i < d; i++) {
        let val = 0; const ro = i * n, so = i * ng;
        for (let g = 0; g < ng; g++) {
          let acc = 0; const a = ro + g * gs, b = g * gs;
          for (let j = 0; j < gs; j++) acc += q[a + j] * xq[b + j];
          val += acc * s[so + g] * xs[g];
        }
        out[outOff + i] = val;
      }
    }

    forward(token, pos) {
      const c = this.cfg, w = this.w, d = c.dim, hd = c.hidden_dim, kvDim = this.kvDim;
      const hs = d / c.n_heads, kvMul = c.n_heads / c.n_kv_heads, T = this.T, gs = this.gs;
      const x = this.x;
      const e = w.tok_emb, eo = token * d;
      for (let j = 0; j < d; j++) x[j] = e.q[eo + j] * e.s[Math.floor((eo + j) / gs)];
      for (let l = 0; l < c.n_layers; l++) {
        this.rmsnorm(this.xb, x, w.rms_att, l * d, d);
        const loff = l * T * kvDim, ko = loff + pos * kvDim;
        this.matmul(this.q, 0, this.xb, w.wq[l], d, d);
        this.matmul(this.kc, ko, this.xb, w.wk[l], d, kvDim);
        this.matmul(this.vc, ko, this.xb, w.wv[l], d, kvDim);
        for (let i = 0; i < d; i += 2) {
          const freq = 1 / Math.pow(10000, (i % hs) / hs), val = pos * freq;
          const fcr = Math.cos(val), fci = Math.sin(val);
          let v0 = this.q[i], v1 = this.q[i + 1];
          this.q[i] = v0 * fcr - v1 * fci; this.q[i + 1] = v0 * fci + v1 * fcr;
          if (i < kvDim) {
            v0 = this.kc[ko + i]; v1 = this.kc[ko + i + 1];
            this.kc[ko + i] = v0 * fcr - v1 * fci; this.kc[ko + i + 1] = v0 * fci + v1 * fcr;
          }
        }
        const scale = Math.sqrt(hs);
        for (let h = 0; h < c.n_heads; h++) {
          const qo = h * hs, ao = h * T, ho = Math.floor(h / kvMul) * hs;
          let mx = -Infinity;
          for (let t = 0; t <= pos; t++) {
            let sc = 0; const kk = loff + t * kvDim + ho;
            for (let i = 0; i < hs; i++) sc += this.q[qo + i] * this.kc[kk + i];
            sc /= scale; this.att[ao + t] = sc; if (sc > mx) mx = sc;
          }
          let sum = 0;
          for (let t = 0; t <= pos; t++) { const v = Math.exp(this.att[ao + t] - mx); this.att[ao + t] = v; sum += v; }
          for (let i = 0; i < hs; i++) this.xb[qo + i] = 0;
          for (let t = 0; t <= pos; t++) {
            const a = this.att[ao + t] / sum, vv = loff + t * kvDim + ho;
            for (let i = 0; i < hs; i++) this.xb[qo + i] += a * this.vc[vv + i];
          }
        }
        this.matmul(this.xb2, 0, this.xb, w.wo[l], d, d);
        for (let i = 0; i < d; i++) x[i] += this.xb2[i];
        this.rmsnorm(this.xb, x, w.rms_ffn, l * d, d);
        this.matmul(this.hb, 0, this.xb, w.w1[l], d, hd);
        this.matmul(this.hb2, 0, this.xb, w.w3[l], d, hd);
        for (let i = 0; i < hd; i++) { const v = this.hb[i]; this.hb[i] = (v / (1 + Math.exp(-v))) * this.hb2[i]; }
        this.matmul(this.xb, 0, this.hb, w.w2[l], hd, d);
        for (let i = 0; i < d; i++) x[i] += this.xb[i];
      }
      this.rmsnorm(x, x, w.rms_final, 0, d);
      this.matmul(this.logits, 0, x, w.tok_emb, d, c.vocab_size);
      return this.logits;
    }

    // C 版 llm_encode と同じ: 話者マーカーを先に拾い、1文字ずつ（無ければバイト片）にしてからスコア順に結合
    encode(text, bos) {
      const t = this.tok, ids = [];
      if (bos) ids.push(1);
      const chars = Array.from(text);
      for (let i = 0; i < chars.length;) {
        const two = chars[i] + (chars[i + 1] || "");
        if (MARKERS.includes(two) && t.lookup.has(two)) { ids.push(t.lookup.get(two)); i += 2; continue; }
        const ch = chars[i];
        if (t.lookup.has(ch)) ids.push(t.lookup.get(ch));
        else for (const b of new TextEncoder().encode(ch)) {
          const hex = "<0x" + b.toString(16).toUpperCase().padStart(2, "0") + ">";
          ids.push(t.lookup.has(hex) ? t.lookup.get(hex) : b + 3);
        }
        i += 1;
      }
      const start = bos ? 1 : 0;
      for (;;) {
        let best = -1e10, bestId = -1, bestIdx = -1;
        for (let i = start; i < ids.length - 1; i++) {
          const id = t.lookup.get(t.vocab[ids[i]] + t.vocab[ids[i + 1]]);
          if (id !== undefined && t.scores[id] > best) { best = t.scores[id]; bestId = id; bestIdx = i; }
        }
        if (bestIdx < 0) break;
        ids.splice(bestIdx, 2, bestId);
      }
      return ids;
    }

    pieceBytes(id) { return this.tok.bytes[id]; }

    sample(logits, temperature, topp, rand) {
      const V = logits.length;
      if (!temperature) { let b = 0; for (let i = 1; i < V; i++) if (logits[i] > logits[b]) b = i; return b; }
      let mx = -Infinity; for (let i = 0; i < V; i++) mx = Math.max(mx, logits[i] / temperature);
      const p = new Float32Array(V); let sum = 0;
      for (let i = 0; i < V; i++) { p[i] = Math.exp(logits[i] / temperature - mx); sum += p[i]; }
      const idx = Array.from({ length: V }, (_, i) => i).sort((a, b) => p[b] - p[a]);
      let cum = 0, last = V - 1;
      for (let k = 0; k < V; k++) { cum += p[idx[k]] / sum; if (cum > topp) { last = k; break; } }
      let r = rand() * cum, acc = 0;
      for (let k = 0; k <= last; k++) { acc += p[idx[k]] / sum; if (r < acc) return idx[k]; }
      return idx[last];
    }

    topk(logits, k) {
      let mx = -Infinity; for (const v of logits) mx = Math.max(mx, v);
      let sum = 0; const p = Array.from(logits, (v) => { const e = Math.exp(v - mx); sum += e; return e; });
      return p.map((v, i) => [i, v / sum]).sort((a, b) => b[1] - a[1]).slice(0, k);
    }
  }

  root.TinyLLM = { Model, parseModel, parseTokenizer };
})(typeof window !== "undefined" ? window : globalThis);
