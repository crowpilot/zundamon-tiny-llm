// tiny-ja-llm / Core2 デモ
// 会話モデル（語彙に「Ａ：」がある）なら「話しかけモード」、それ以外は「自由生成モード」で起動する。
//   話しかけモード: 話しかけ一覧（openers.h）から相手の台詞を1つ選んで渡し、ずんだもん（Ｂ）の返事だけを生成する。
//                   返事が終わると5秒後に次の話しかけ。会話は文脈が埋まるまで続き、埋まったら画面を消して最初から。
//   自由生成モード: 何も渡さずに文章（会話なら相手の台詞も）を書き続ける（従来の動作）
//   画面タップ / ボタンA : 次へ（話しかけ / 新しい話）
//   ボタンB              : greedy(t=0) ⇔ サンプリング(t=0.8) 切替
//   ボタンC              : 「迷い」表示（次トークン候補の上位3つと確率）ON/OFF
//   ボタンC 長押し        : 話しかけモード ⇔ 自由生成モード
// モデルは flash_core2.sh が src/model_data.h に埋め込む。
#include <M5Unified.h>
#include <cstring>
#include "llm_core.h"
#include "model_data.h"   // model_bin[], model_bin_len, tok_bin[], tok_bin_len, MODEL_NAME
#include "openers.h"      // OPENERS[], N_OPENERS

static const int MAX_STEPS = 256;     // KVキャッシュ長の上限（モデルの seq_len がこれより短ければそちら）
static const int STATUS_H = 36;       // 下部ステータス欄の高さ
static const int MAX_REPLY = 48;      // 1回の返事の最大トークン数
static const uint32_t NEXT_MS = 5000; // 返事のあと次の話しかけまでの間
#ifndef SERIAL_ECHO
#define SERIAL_ECHO 0                 // 1 にすると生成文をシリアルにも流す（書き込み時の雑音になるので既定は0）
#endif

static Llm llm;
static int token = 1, pos = 0;
static bool running = true, show_cand = true, chat_mode = false, can_chat = false;
static uint32_t t_start = 0, end_at = 0;
static int gen_count = 0, reply_len = 0, last_opener = -1;
static float cur_temp = 0.8f;
static float *last_logits = nullptr;
static int id_a = -1, id_b = -1;      // 話者マーカー「Ａ：」「Ｂ：」のトークン番号

static void *psram_alloc(size_t n) {
    void *p = ps_malloc(n);
    return p ? p : malloc(n);
}

static void draw_status(const char *line2) {
    // drawString はカーソルもスクロールも動かさないので本文に影響しない
    auto &d = M5.Display;
    int y0 = d.height() - STATUS_H;
    d.fillRect(0, y0, d.width(), STATUS_H, TFT_NAVY);
    float tps = (gen_count > 1 && millis() > t_start) ? (gen_count - 1) * 1000.0f / (millis() - t_start) : 0;
    char buf[96];
    snprintf(buf, sizeof(buf), "%s %s t=%.1f %d/%d %.1ftok/s", MODEL_NAME, chat_mode ? "話" : "自", cur_temp,
             pos, llm.max_steps, tps);
    d.setTextColor(TFT_WHITE, TFT_NAVY);
    d.drawString(buf, 4, y0 + 1);
    if (line2) { d.setTextColor(TFT_YELLOW, TFT_NAVY); d.drawString(line2, 4, y0 + 18); }
    d.setTextColor(TFT_WHITE, TFT_BLACK);
}

static void clear_text() {
    auto &d = M5.Display;
    d.fillScreen(TFT_BLACK);
    d.setScrollRect(0, 0, d.width(), d.height() - STATUS_H);
    d.setCursor(0, 0);
    d.setTextColor(TFT_WHITE, TFT_BLACK);
}

static int single_token(const char *s) {
    int toks[8], n = 0;
    llm_encode(&llm, s, 0, 0, toks, &n);
    return n == 1 ? toks[0] : -1;
}

// ---------------------------------------------------------------- 自由生成モード
static void new_story() {
    clear_text();
    llm.temperature = cur_temp;
    llm.rng = ((unsigned long long)esp_random() << 32) | esp_random();
    token = 1; pos = 0; gen_count = 0; running = true; end_at = 0;
    t_start = millis();
    Serial.println("\n----- new story -----");
    draw_status(nullptr);
}

// ---------------------------------------------------------------- 話しかけモード
static void chat_turn() {
    auto &d = M5.Display;
    llm.temperature = cur_temp;
    // 文脈（KVキャッシュ）に今回の話しかけ＋返事が入りきらなければ、最初からやり直す
    bool reset = (pos == 0) || pos + 24 + MAX_REPLY >= llm.max_steps;
    if (reset) { clear_text(); pos = 0; }
    int k;
    do { k = (int)(esp_random() % N_OPENERS); } while (N_OPENERS > 1 && k == last_opener);
    last_opener = k;
    char text[160];
    snprintf(text, sizeof(text), "Ａ：%sＢ：", OPENERS[k]);
    d.setTextColor(TFT_CYAN, TFT_BLACK);
    d.printf("%sＡ：%s\n", reset ? "" : "\n", OPENERS[k]);
    d.setTextColor(TFT_GREENYELLOW, TFT_BLACK);
    d.print("Ｂ：");
    d.setTextColor(TFT_WHITE, TFT_BLACK);
    draw_status("考え中…");
    static int toks[192];
    int n = 0;
    llm_encode(&llm, text, reset ? 1 : 0, 0, toks, &n);
    for (int i = 0; i < n && pos < llm.max_steps; i++) last_logits = llm_forward(&llm, toks[i], pos++);
    token = toks[n - 1];
    running = true; end_at = 0; reply_len = 0; gen_count = 0;
    t_start = millis();
}

static void make_cand(const float *logits, char *cand, size_t len) {
    LlmProbIndex top[3];
    int n = llm_topk(&llm, logits, 3, top);
    int off = 0;
    cand[0] = 0;
    for (int i = 0; i < n && off < (int)len - 1; i++) {
        const char *p = llm_decode(&llm, token, top[i].index);
        char clean[24]; int k = 0;
        for (; *p && k < 20; p++) clean[k++] = (*p == '\n') ? ' ' : *p;
        clean[k] = 0;
        off += snprintf(cand + off, len - off, "%s[%s]%d%%", i ? " " : "", clean, (int)(top[i].prob * 100 + 0.5f));
    }
}

static void restart() { if (chat_mode) { pos = 0; chat_turn(); } else new_story(); }

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);
    Serial.begin(115200);
    auto &d = M5.Display;
    d.setFont(&fonts::efontJA_16);   // ASCIIと日本語の両方を表示できるフォント
    d.setTextWrap(true, true);
    d.setTextScroll(true);
    d.fillScreen(TFT_BLACK);
    d.setCursor(0, 0);
    d.printf("loading %s ...\nPSRAM free: %u bytes\n", MODEL_NAME, (unsigned)ESP.getFreePsram());

    int rc = llm_init(&llm, model_bin, model_bin_len, tok_bin, tok_bin_len, MAX_STEPS, psram_alloc);
    if (rc != 0) {
        d.setTextColor(TFT_RED);
        d.printf("llm_init failed: %d\n", rc);
        Serial.printf("llm_init failed: %d\n", rc);
        while (true) delay(1000);
    }
    LlmConfig &c = llm.config;
    Serial.printf("config dim=%d hidden=%d layers=%d heads=%d kv=%d vocab=%d seq=%d\n",
                  c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads, c.vocab_size, c.seq_len);
    d.printf("dim=%d layers=%d vocab=%d\n%s max %d tokens\n", c.dim, c.n_layers, c.vocab_size,
             llm.gs ? "int8" : "fp32", llm.max_steps);
    id_a = single_token("Ａ：");
    id_b = single_token("Ｂ：");
    can_chat = id_a >= 0 && id_b >= 0;
    chat_mode = can_chat;
    llm.rng = ((unsigned long long)esp_random() << 32) | esp_random();
    delay(1200);
    restart();
}

void loop() {
    M5.update();
    bool tapped = M5.Touch.getCount() > 0 && M5.Touch.getDetail().wasPressed()
                  && M5.Touch.getDetail().y < M5.Display.height() - STATUS_H;
    if (M5.BtnA.wasPressed() || tapped) { if (chat_mode && !running) chat_turn(); else restart(); return; }
    if (M5.BtnB.wasPressed()) { cur_temp = cur_temp > 0 ? 0.0f : 0.8f; llm.temperature = cur_temp; draw_status(nullptr); }
    if (M5.BtnC.wasHold() && can_chat) { chat_mode = !chat_mode; pos = 0; restart(); return; }
    if (M5.BtnC.wasClicked()) { show_cand = !show_cand; draw_status(nullptr); }

    if (!running) {
        if (end_at && millis() - end_at > (chat_mode ? NEXT_MS : 6000)) { if (chat_mode) chat_turn(); else new_story(); }
        delay(10);
        return;
    }

    char cand[96] = {0};
    int next;
    if (chat_mode) {
        // 直前の forward の結果から次を選ぶ。「Ａ：」が出たら＝ずんだもんが話し終えたので止める
        if (show_cand) make_cand(last_logits, cand, sizeof(cand));
        next = llm_sample(&llm, last_logits);
        if (next == id_a || next == id_b || next == 1 || next == 2 || pos >= llm.max_steps || reply_len >= MAX_REPLY) {
            running = false; end_at = millis();
            draw_status("tap / A: 次の話しかけ  B: 温度  C長押し: 自由生成");
            return;
        }
        const char *piece = llm_decode(&llm, token, next);
        M5.Display.print(piece);
        if (SERIAL_ECHO) Serial.print(piece);
        token = next;
        last_logits = llm_forward(&llm, token, pos++);
        reply_len++;
    } else {
        float *logits = llm_forward(&llm, token, pos);
        if (show_cand) make_cand(logits, cand, sizeof(cand));
        next = llm_sample(&llm, logits);
        pos++;
        if (next == 1 || next == 2 || pos >= llm.max_steps) {
            M5.Display.setTextColor(TFT_DARKGREY, TFT_BLACK);
            M5.Display.print("\n(おわり)");
            running = false; end_at = millis();
            draw_status("tap / A: 次の話   B: 温度   C長押し: 話しかけ");
            return;
        }
        const char *piece = llm_decode(&llm, token, next);
        // 会話モデルの話者マーカー（Ａ：/Ｂ：）の前で改行して読みやすくする
        if (pos > 1 && (next == id_a || next == id_b)) {
            M5.Display.print("\n");
            if (SERIAL_ECHO) Serial.print("\n");
        }
        M5.Display.print(piece);
        if (SERIAL_ECHO) Serial.print(piece);
        token = next;
    }
    gen_count++;
    if (gen_count == 1) t_start = millis();
    draw_status(show_cand ? cand : nullptr);
}
