# STRIDE-Aligned Behavioural Feature Engineering

Raw network-flow features record *what happened* (byte counts, packet sizes, durations, protocol flags) but not *which security property is being violated*. STRIDE dimensions correspond to security properties — Authentication, Integrity, Non-repudiation, Confidentiality, Availability, Authorisation — and their violations manifest as specific **behavioural patterns**, not simple statistical deviations.

To close this gap we engineer **eight** behavioural signals. Each has a clear security-theoretic justification, is computable on **both** UNSW-NB15 and CIC-IDS-2018, and uses `log1p` transforms to compress heavily skewed traffic distributions. Division-by-zero is handled with a small constant; infinities are clipped to zero.

The eight signals are appended to **37 raw numeric** + **157 one-hot categorical** features, yielding a final matrix **X ∈ ℝ^(N×202)**, standardised with `StandardScaler` (fit on train only).

---

## The Eight Signals

### 1. Resource Exhaustion Index — `feat_resource_exhaustion`

$$f_{\text{resource}} = \log(1 + \text{sload}) + \frac{\log(1 + \text{spkts})}{\text{dur}}$$

**Rationale.** DoS attacks exhaust resources. Captures raw source load plus packet-generation rate per unit time. Legitimate high-bandwidth sessions (e.g. streaming) have high load but moderate packet rates; DoS maximises both.
**STRIDE target:** Denial of Service (D)

### 2. Connection Burst Rate — `feat_conn_burst_rate`

$$f_{\text{burst}} = \frac{\text{ct\_srv\_src} + \text{ct\_srv\_dst}}{\text{dur}}$$

**Rationale.** Scanning/flooding establishes many connections in a short window. A high burst rate with short duration indicates automated scanning — consuming connection resources (DoS) from a potentially falsified source (Spoofing).
**STRIDE target:** Denial of Service (D), Spoofing (S)

### 3. Traffic Asymmetry Ratio — `feat_asymmetry_ratio`

$$f_{\text{asym}} = \frac{|\text{sbytes} - \text{dbytes}|}{\text{sbytes} + \text{dbytes} + 1}$$

**Rationale.** Normal interactive sessions are roughly symmetric. Covert exfiltration and command-and-control channels are pronouncedly asymmetric. Bounded in [0, 1). **Ablation:** removing it drops Repudiation F1 by 0.0538; **SHAP:** ranked #1 for Repudiation.
**STRIDE target:** Information Disclosure (I), Repudiation (R)

### 4. Session Entropy Index — `feat_session_entropy`

$$f_{\text{entropy}} = \log(1 + \text{sjit} + \text{djit}) + \log(1 + |\text{smean} - \text{dmean}|)$$

**Rationale.** Fuzzing/tampering generates irregular, high-variance traffic. Combines timing irregularity (jitter) with payload irregularity (mean packet-size difference). Fuzzers maximise both.
**STRIDE target:** Tampering (T)

### 5. Privilege Escalation Score — `feat_privesc_score`

$$f_{\text{privesc}} = 2 \cdot \mathbb{1}_{\text{ftp}} + \log(1 + \text{trans\_depth}) + \log(1 + \text{res\_body\_len})$$

**Rationale.** EoP exploits application-layer mechanisms. FTP login is weighted 2× (direct escalation vector); transaction depth signals nested exploitation; response body length reflects command-execution output.
**STRIDE target:** Elevation of Privilege (E)

### 6. Stealth Probe Index — `feat_stealth_probe`

$$f_{\text{stealth}} = \frac{|\text{sttl} - \text{dttl}|}{\text{dur} \cdot \log(1 + \text{sbytes})}$$

**Rationale.** Reconnaissance probes topology while evading detection, often manipulating TTL. Isolates the TTL-discrepancy signal from legitimate high-TTL traffic. SHAP independently confirms `sttl` as the strongest single predictor across dimensions.
**STRIDE target:** Spoofing (S), Information Disclosure (I)

### 7. Payload Anomaly Score — `feat_payload_anomaly`

$$f_{\text{payload}} = \frac{\log(1 + \text{smean}) \cdot \log(1 + \text{dmean})}{\log(1 + \text{spkts} + \text{dpkts})}$$

**Rationale.** Shellcode/exploit delivery concentrates large payloads into few packets. Distinct from large file transfers (which have high packet counts).
**STRIDE target:** Tampering (T), Elevation of Privilege (E)

### 8. Repudiation Risk Score — `feat_repudiation_risk`

$$f_{\text{repud}} = 1.5 \cdot \text{ct\_ftp} + \log(1 + \text{trans\_depth}) \cdot \log(1 + \text{sload})$$

**Rationale.** Repudiation hides evidence of activity. FTP command counts weighted 1.5× (covert file transfer with minimal logging); the interaction term captures deeply-nested, high-volume covert data movement.
**STRIDE target:** Repudiation (R)

---

## Dual Validation

The signals are validated by two independent methods that **agree**:

| Method | Evidence |
|---|---|
| **Ablation study** | Removing all 8 signals drops Macro F1 by 0.0307; largest drops in EoP (−0.0698) and Repudiation (−0.0538) — the dimensions `feat_privesc_score` and `feat_asymmetry_ratio` target. |
| **SHAP analysis** | `feat_asymmetry_ratio` is the #1 feature for Repudiation; `feat_session_entropy` appears in the top-3 for Tampering and Information Disclosure. |

Directional agreement between design intent and empirical impact provides strong evidence the signals capture genuine threat-relevant patterns rather than spurious correlations.

> Implementation: see the feature-engineering cell in [`../notebooks/stride_multilabel_ids.ipynb`](../notebooks/stride_multilabel_ids.ipynb).
