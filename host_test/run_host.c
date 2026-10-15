// PC上で llm_core を動かす確認用ツール（Core2に焼く前の動作確認・学習済みモデルの試運転用）
// build: gcc -O2 -I../core2/src -o run_host run_host.c ../core2/src/llm_core.c -lm
// usage: ./run_host model.bin tok.bin [-t temp] [-n steps] [-i prompt] [-k topk表示数] [-s seed]
#include "llm_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open failed: %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);
    void *buf = malloc(*len);
    if (fread(buf, 1, *len, f) != *len) { fprintf(stderr, "read failed\n"); exit(1); }
    fclose(f);
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.bin tok.bin [-t temp] [-n steps] [-i prompt] [-k topk] [-s seed]\n", argv[0]); return 1; }
    float temp = 0.0f; int steps = 256, topk = 0; const char *prompt = ""; unsigned long long seed = (unsigned long long)time(NULL);
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "-t")) temp = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "-n")) steps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-i")) prompt = argv[i + 1];
        else if (!strcmp(argv[i], "-k")) topk = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-s")) seed = strtoull(argv[i + 1], NULL, 10);
    }
    size_t mlen, tlen;
    void *model = slurp(argv[1], &mlen), *tok = slurp(argv[2], &tlen);
    Llm m;
    int rc = llm_init(&m, model, mlen, tok, tlen, steps, NULL);
    if (rc) { fprintf(stderr, "llm_init failed: %d\n", rc); return 1; }
    m.temperature = temp; m.rng = seed ? seed : 1;
    LlmConfig *c = &m.config;
    fprintf(stderr, "config: dim=%d hidden=%d layers=%d heads=%d kv=%d vocab=%d seq=%d\n",
            c->dim, c->hidden_dim, c->n_layers, c->n_heads, c->n_kv_heads, c->vocab_size, c->seq_len);

    int *ptoks = malloc((strlen(prompt) + 3) * sizeof(int)), np;
    llm_encode(&m, prompt, 1, 0, ptoks, &np);
    int token = ptoks[0], next, pos = 0;
    clock_t start = 0;
    LlmProbIndex top[8];
    if (topk > 8) topk = 8;
    while (pos < m.max_steps) {
        float *logits = llm_forward(&m, token, pos);
        if (pos < np - 1) next = ptoks[pos + 1];
        else {
            if (topk > 0) {
                int n = llm_topk(&m, logits, topk, top);
                fprintf(stderr, "[");
                for (int i = 0; i < n; i++) fprintf(stderr, "%s'%s'%.2f", i ? " " : "", llm_decode(&m, token, top[i].index), top[i].prob);
                fprintf(stderr, "] ");
            }
            next = llm_sample(&m, logits);
        }
        pos++;
        if (next == 1) break;
        const char *piece = llm_decode(&m, token, next);
        if (pos > 1 && (!strncmp(piece, "Ａ：", strlen("Ａ：")) || !strncmp(piece, "Ｂ：", strlen("Ｂ：")))) printf("\n");
        printf("%s", piece);
        fflush(stdout);
        token = next;
        if (!start) start = clock();
    }
    printf("\n");
    if (pos > 1) fprintf(stderr, "tok/s: %.1f\n", (pos - 1) / ((double)(clock() - start) / CLOCKS_PER_SEC));
    return 0;
}
