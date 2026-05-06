# SpecGuard Quality + Cost Comparison

Per assertion four boolean axes (Tier 1, structural):

- **lint** — pyslang accepts the syntax
- **resolved** — every identifier appears in the design's signal map (no hallucination)
- **non-trivial** — not a bare `$bits()==const`, not a `1'b1` body
- **unique** — not a string-normalised duplicate of an earlier assertion

**N_T1** = lint ∧ resolved ∧ non-trivial ∧ unique.

**T1 stratification** (within N_T1):
- **simple** — bare equalities, single-signal expressions, no implication
- **complex** — uses `$past`/`$stable`/`$rose`/`$fell`/`$onehot`/`$countones`/`inside`/`|->` (with multi-signal antecedent)/`|=>`/`throughout`/`until`

Tier 2 (coverage of documented intent):

- **docs covered / total** — documented sentences (in `docs/`) where ≥50% of the docprop's signals appear in some N_T1 assertion's identifier set (and at least one signal overlaps)

Headline (lower `tok/useful` = better):

- **useful_score** = 1×T1_simple + 2×T1_complex + 1×docs_covered
- **cost_useful** = total_tokens / useful_score
- **cost_per_T1** = total_tokens / N_T1   (diagnostic, no complexity weighting)


## Functional coverage — the headline

Each design is decomposed into discrete coverage events across three axes:
- **reset** — one event per resettable register (from `facts.reset_values`)
- **doc** — one event per documented sentence with at least one design signal
- **case** — one event per (case_selector, case_value, lhs) triple from the AST extractor

An event is **hit** when at least one N_T1 assertion's identifier set overlaps the event's required signals at ≥50%. Case events additionally require the case value literal to appear in the assertion text.

**cost_per_cov** = total_tokens / events_hit  (lower is better).
**FC%** = events_hit / events_total — directly comparable across methods on the same design.

| Method | Designs | Tokens μ | Events μ | Hit μ | **FC%** | Reset% | Doc% | Case% | **tok/cov** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| assertain | 10 | 38,437 | 401.9 | 0.0 | **0%** | 0% | 0% | 0% | **—** |
| assertllm | 8 | 68,438 | 236.0 | 0.0 | **0%** | 0% | 0% | 0% | **—** |
| chiraag | 12 | 15,329 | 259.4 | 67.8 | **16%** | 17% | 8% | 0% | **677** |
| specguard | 11 | 296,893 | 390.4 | 359.5 | **85%** | 91% | 35% | 9% | **1,789** |

## Structural quality (T1) — diagnostic

| Method | Designs | Tokens μ | Asserts μ | N_T1 μ | T1_S | T1_C | Cmplx% | DocsCov μ | Cov% | Lint% | Resolve% | tok/T1 | tok/useful |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| assertain | 10 | 38,437 | 13.8 | 0.0 | 0.0 | 0.0 | 0% | 0.0 | 0% | 91% | 0% | — | — |
| assertllm | 8 | 68,438 | 6.5 | 0.1 | 0.0 | 0.1 | 12% | 0.0 | 0% | 100% | 100% | 83,841 | 41,920 |
| chiraag | 12 | 15,329 | 6.2 | 2.0 | 1.4 | 0.6 | 9% | 1.5 | 8% | 56% | 16% | 2,282 | 1,140 |
| specguard | 11 | 296,893 | 61.8 | 59.7 | 25.2 | 34.5 | 35% | 5.0 | 35% | 96% | 97% | 26,972 | 18,636 |

## Per (method, design) cell — functional coverage

| Method | Design | Runs | Tokens μ | Events | Hit | **FC%** | Reset% | Doc% | Case% | **tok/cov** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| assertain | cdma | 1 | 28,884 | 989 | 0 | **0%** | 0% | 0% | 0% | **—** |
| assertain | cdp | 1 | 26,917 | 897 | 0 | **0%** | 0% | 0% | 0% | **—** |
| assertain | cmac | 1 | 30,426 | 69 | 0 | **0%** | 0% | 0% | 0% | **—** |
| assertain | cmacfull | 1 | 30,943 | 164 | 0 | **0%** | 0% | 0% | 0% | **—** |
| assertain | csc | 1 | 30,563 | 464 | 0 | **0%** | 0% | 0% | 0% | **—** |
| assertain | fifox | 1 | 33,679 | 21 | 0 | **0%** | 0% | 0% | 0% | **—** |
| assertain | i2c | 1 | 118,800 | 2 | 0 | **0%** | 0% | 0% | 0% | **—** |
| assertain | pdp | 1 | 26,724 | 440 | 0 | **0%** | 0% | 0% | 0% | **—** |
| assertain | rubik | 1 | 27,752 | 182 | 0 | **0%** | 0% | 0% | 0% | **—** |
| assertain | sdp | 1 | 29,679 | 791 | 0 | **0%** | 0% | 0% | 0% | **—** |
| assertllm | cacc | 1 | 83,841 | 257 | 0 | **0%** | 0% | 0% | 0% | **—** |
| assertllm | cdp | 1 | 100,514 | 897 | 0 | **0%** | 0% | 0% | 0% | **—** |
| assertllm | cmac | 1 | 30,098 | 69 | 0 | **0%** | 0% | 0% | 0% | **—** |
| assertllm | fifox | 1 | 40,429 | 21 | 0 | **0%** | 0% | 0% | 0% | **—** |
| assertllm | i2c | 1 | 51,065 | 2 | 0 | **0%** | 0% | 0% | 0% | **—** |
| assertllm | pdp | 1 | 128,105 | 440 | 0 | **0%** | 0% | 0% | 0% | **—** |
| assertllm | rubik | 1 | 80,616 | 182 | 0 | **0%** | 0% | 0% | 0% | **—** |
| assertllm | rvv | 1 | 32,832 | 20 | 0 | **0%** | 0% | 0% | 0% | **—** |
| chiraag | cacc | 1 | 22,070 | 0 | 0 | **0%** | 0% | 0% | 0% | **—** |
| chiraag | cdma | 1 | 7,040 | 989 | 0 | **0%** | 0% | 0% | 0% | **—** |
| chiraag | cdp | 1 | 8,564 | 897 | 0 | **0%** | 0% | 0% | 0% | **—** |
| chiraag | cmac | 1 | 17,661 | 69 | 19 | **28%** | 100% | 20% | 0% | **930** |
| chiraag | cmacfull | 1 | 7,874 | 164 | 0 | **0%** | 0% | 0% | 0% | **—** |
| chiraag | csc | 1 | 20,941 | 0 | 0 | **0%** | 0% | 0% | 0% | **—** |
| chiraag | fifox | 1 | 14,148 | 21 | 13 | **62%** | 0% | 62% | 0% | **1,088** |
| chiraag | i2c | 1 | 22,846 | 0 | 0 | **0%** | 0% | 0% | 0% | **—** |
| chiraag | pdp | 1 | 20,631 | 0 | 0 | **0%** | 0% | 0% | 0% | **—** |
| chiraag | rubik | 1 | 6,640 | 182 | 0 | **0%** | 0% | 0% | 0% | **—** |
| chiraag | rvv | 1 | 24,126 | 0 | 0 | **0%** | 0% | 0% | 0% | **—** |
| chiraag | sdp | 1 | 11,411 | 791 | 781 | **99%** | 100% | 17% | 0% | **15** |
| specguard | cacc | 2 | 941,140 | 257 | 251 | **98%** | 100% | 57% | 0% | **3,750** |
| specguard | cdma | 2 | 274,794 | 989 | 848 | **86%** | 98% | 80% | 0% | **324** |
| specguard | cdp | 2 | 27,844 | 897 | 843 | **94%** | 100% | 0% | 0% | **33** |
| specguard | cmac | 2 | 119,144 | 69 | 60 | **87%** | 100% | 65% | 94% | **1,986** |
| specguard | cmacfull | 2 | 609,592 | 164 | 119 | **73%** | 100% | 46% | 0% | **5,123** |
| specguard | csc | 1 | 547,511 | 464 | 446 | **96%** | 98% | 14% | 0% | **1,228** |
| specguard | fifox | 2 | 43,288 | 21 | 12 | **57%** | 0% | 57% | 0% | **3,607** |
| specguard | pdp | 2 | 77,076 | 440 | 410 | **93%** | 100% | 12% | 0% | **188** |
| specguard | rubik | 2 | 250,632 | 182 | 174 | **96%** | 100% | 0% | 0% | **1,440** |
| specguard | rvv | 2 | 18,578 | 20 | 12 | **60%** | 100% | 56% | 0% | **1,548** |
| specguard | sdp | 1 | 356,226 | 791 | 780 | **99%** | 100% | 0% | 0% | **457** |

## Per (method, design) cell — structural T1 (diagnostic)

| Method | Design | Runs | Tokens μ | A μ | N_T1 | T1_S | T1_C | Cmplx% | Docs | Cov | Cov% | Lint% | Resolve% | tok/T1 | tok/useful |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| assertain | cdma | 1 | 28,884 | 12.0 | 0.0 | 0.0 | 0.0 | 0% | 5.0 | 0.0 | 0% | 92% | 0% | — | — |
| assertain | cdp | 1 | 26,917 | 12.0 | 0.0 | 0.0 | 0.0 | 0% | 5.0 | 0.0 | 0% | 92% | 0% | — | — |
| assertain | cmac | 1 | 30,426 | 13.0 | 0.0 | 0.0 | 0.0 | 0% | 20.0 | 0.0 | 0% | 92% | 0% | — | — |
| assertain | cmacfull | 1 | 30,943 | 11.0 | 0.0 | 0.0 | 0.0 | 0% | 13.0 | 0.0 | 0% | 91% | 0% | — | — |
| assertain | csc | 1 | 30,563 | 13.0 | 0.0 | 0.0 | 0.0 | 0% | 7.0 | 0.0 | 0% | 92% | 0% | — | — |
| assertain | fifox | 1 | 33,679 | 5.0 | 0.0 | 0.0 | 0.0 | 0% | 21.0 | 0.0 | 0% | 80% | 0% | — | — |
| assertain | i2c | 1 | 118,800 | 39.0 | 0.0 | 0.0 | 0.0 | 0% | 2.0 | 0.0 | 0% | 97% | 0% | — | — |
| assertain | pdp | 1 | 26,724 | 13.0 | 0.0 | 0.0 | 0.0 | 0% | 8.0 | 0.0 | 0% | 92% | 0% | — | — |
| assertain | rubik | 1 | 27,752 | 13.0 | 0.0 | 0.0 | 0.0 | 0% | 4.0 | 0.0 | 0% | 92% | 0% | — | — |
| assertain | sdp | 1 | 29,679 | 7.0 | 0.0 | 0.0 | 0.0 | 0% | 6.0 | 0.0 | 0% | 86% | 0% | — | — |
| assertllm | cacc | 1 | 83,841 | 7.0 | 1.0 | 0.0 | 1.0 | 100% | 14.0 | 0.0 | 0% | 100% | 100% | 83,841 | 41,920 |
| assertllm | cdp | 1 | 100,514 | 4.0 | 0.0 | 0.0 | 0.0 | 0% | 5.0 | 0.0 | 0% | 100% | 100% | — | — |
| assertllm | cmac | 1 | 30,098 | 8.0 | 0.0 | 0.0 | 0.0 | 0% | 20.0 | 0.0 | 0% | 100% | 100% | — | — |
| assertllm | fifox | 1 | 40,429 | 2.0 | 0.0 | 0.0 | 0.0 | 0% | 21.0 | 0.0 | 0% | 100% | 100% | — | — |
| assertllm | i2c | 1 | 51,065 | 2.0 | 0.0 | 0.0 | 0.0 | 0% | 2.0 | 0.0 | 0% | 100% | 100% | — | — |
| assertllm | pdp | 1 | 128,105 | 7.0 | 0.0 | 0.0 | 0.0 | 0% | 8.0 | 0.0 | 0% | 100% | 100% | — | — |
| assertllm | rubik | 1 | 80,616 | 7.0 | 0.0 | 0.0 | 0.0 | 0% | 4.0 | 0.0 | 0% | 100% | 100% | — | — |
| assertllm | rvv | 1 | 32,832 | 15.0 | 0.0 | 0.0 | 0.0 | 0% | 18.0 | 0.0 | 0% | 100% | 100% | — | — |
| chiraag | cacc | 1 | 22,070 | 0.0 | 0.0 | 0.0 | 0.0 | 0% | 0.0 | 0.0 | 0% | 0% | 0% | — | — |
| chiraag | cdma | 1 | 7,040 | 12.0 | 0.0 | 0.0 | 0.0 | 0% | 5.0 | 0.0 | 0% | 83% | 0% | — | — |
| chiraag | cdp | 1 | 8,564 | 5.0 | 0.0 | 0.0 | 0.0 | 0% | 5.0 | 0.0 | 0% | 100% | 0% | — | — |
| chiraag | cmac | 1 | 17,661 | 13.0 | 12.0 | 9.0 | 3.0 | 25% | 20.0 | 4.0 | 20% | 100% | 92% | 1,472 | 930 |
| chiraag | cmacfull | 1 | 7,874 | 10.0 | 0.0 | 0.0 | 0.0 | 0% | 13.0 | 0.0 | 0% | 90% | 0% | — | — |
| chiraag | csc | 1 | 20,941 | 0.0 | 0.0 | 0.0 | 0.0 | 0% | 0.0 | 0.0 | 0% | 0% | 0% | — | — |
| chiraag | fifox | 1 | 14,148 | 17.0 | 9.0 | 7.0 | 2.0 | 22% | 21.0 | 13.0 | 62% | 100% | 59% | 1,572 | 590 |
| chiraag | i2c | 1 | 22,846 | 0.0 | 0.0 | 0.0 | 0.0 | 0% | 0.0 | 0.0 | 0% | 0% | 0% | — | — |
| chiraag | pdp | 1 | 20,631 | 0.0 | 0.0 | 0.0 | 0.0 | 0% | 0.0 | 0.0 | 0% | 0% | 0% | — | — |
| chiraag | rubik | 1 | 6,640 | 7.0 | 0.0 | 0.0 | 0.0 | 0% | 4.0 | 0.0 | 0% | 100% | 0% | — | — |
| chiraag | rvv | 1 | 24,126 | 0.0 | 0.0 | 0.0 | 0.0 | 0% | 0.0 | 0.0 | 0% | 0% | 0% | — | — |
| chiraag | sdp | 1 | 11,411 | 11.0 | 3.0 | 1.0 | 2.0 | 67% | 6.0 | 1.0 | 17% | 100% | 45% | 3,804 | 1,902 |
| specguard | cacc | 2 | 941,140 | 396.5 | 389.0 | 90.0 | 299.0 | 77% | 14.0 | 8.0 | 57% | 100% | 98% | 2,419 | 1,352 |
| specguard | cdma | 2 | 274,794 | 11.5 | 10.5 | 7.0 | 3.5 | 33% | 5.0 | 4.0 | 80% | 91% | 100% | 26,171 | 15,266 |
| specguard | cdp | 2 | 27,844 | 18.0 | 18.0 | 8.0 | 10.0 | 56% | 5.0 | 0.0 | 0% | 100% | 100% | 1,547 | 994 |
| specguard | cmac | 2 | 119,144 | 56.5 | 56.0 | 44.5 | 11.5 | 21% | 20.0 | 13.0 | 65% | 99% | 100% | 2,128 | 1,480 |
| specguard | cmacfull | 2 | 609,592 | 16.0 | 14.5 | 11.0 | 3.5 | 24% | 13.0 | 6.0 | 46% | 97% | 94% | 42,041 | 25,400 |
| specguard | csc | 1 | 547,511 | 4.0 | 4.0 | 2.0 | 2.0 | 50% | 7.0 | 1.0 | 14% | 100% | 100% | 136,878 | 78,216 |
| specguard | fifox | 2 | 43,288 | 26.0 | 26.0 | 10.5 | 15.5 | 60% | 21.0 | 12.0 | 57% | 100% | 100% | 1,665 | 809 |
| specguard | pdp | 2 | 77,076 | 58.5 | 57.5 | 27.5 | 30.0 | 52% | 8.0 | 1.0 | 12% | 99% | 99% | 1,340 | 871 |
| specguard | rubik | 2 | 250,632 | 9.0 | 8.0 | 7.5 | 0.5 | 6% | 4.0 | 0.0 | 0% | 94% | 94% | 31,329 | 29,486 |
| specguard | rvv | 2 | 18,578 | 75.0 | 66.0 | 62.5 | 3.5 | 5% | 18.0 | 10.0 | 56% | 100% | 88% | 281 | 234 |
| specguard | sdp | 1 | 356,226 | 9.0 | 7.0 | 7.0 | 0.0 | 0% | 6.0 | 0.0 | 0% | 78% | 89% | 50,889 | 50,889 |
