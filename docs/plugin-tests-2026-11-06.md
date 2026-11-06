# DMR and CapUnlocker: short Tavi on Vita

Device tests on 2026-11-06, actual firmware 3.65 (public version API spoofed to
3.74). Tests began after PS Vita Alive was closed with the user's authorization.
PSVshellPlus reported **444 MHz inside METEOR001** throughout the matched runs;
the helper title reported 500 MHz. Do not attribute these timings to 500 MHz.

## Controlled workload

System message: `You are a helpful assistant. Follow the instructions and answer concisely. If you do not know, say so.`

User: `Mira is older than Soren. Soren is older than Tavi. Who is youngest? Answer with only the name.`

Same Qwen3.5-0.8B model files, Q4/int8 kernels, app UI, sampling, and chat
template as v0.3.0. The 32 fixed system/template tokens were prepared at startup;
the remaining 35 prompt tokens were processed for every run. No user prompt or
answer was cached. Time includes restoring the shared template and generation
through natural EOS, excluding app/model startup. Each comparison used two
controls, two candidate runs, and a return-to-control run in one process.

| Experiment | Control runs | Candidate runs | Return control | Decision |
|---|---|---|---|---|
| DMR + 27,136 additional cached output rows | 34.350, 34.376 s | 33.875, 33.914 s | 34.392 s | Leave out: only ~0.47 s / 1.4% improvement |
| CapUnlocker + four workers | 34.447, 34.505 s | 30.689, 30.696 s | 34.539 s | Keep: ~3.78 s / 11.0% shorter against initial controls |

Every run returned exactly `Tavi` followed by EOS. Hashes of the complete float
logit vectors matched at each generation step. Four-worker runs observed work
on cores 0, 1, 2, and 3 (`cores=15`); controls observed cores 0, 1, and 2 only
(`cores=7`). Host checks independently verified exact full-logit equality when
switching workers 3 → 4 → 3 → 1, plus cancellation recovery.

Raw evidence: [CapUnlocker](plugin-cap-tavi.txt), [DMR](plugin-dmr-tavi.txt),
[host engine checks](plugin-engine-check.txt).

## DMR details and limitations

Official DMR v1.00, source revision
`e73f6149f4173f9840f773d28a5a103815c76ff0`, was staged from
https://github.com/GrapheneCt/DMR. The released binary SHA-256 was
`862f2132a771d422b61420460fc2c67023eb1706000ecb660760725a10135749`.
The runtime loader reported `0x8002d021`; a source-compatible diagnostic build
was also attempted. Therefore the log does not establish which DMR binary
supplied the active hooks. Functionally, memory growth changed from failing
at every tested size before activation to succeeding at 16 MiB afterwards
(24 MiB failed). An additional 15,630,336 bytes of unchanged output weights were
successfully allocated and read into RAM.

Output I/O fell from ~572 ms to ~53 ms over the answer. The small elapsed-time
benefit has a direct I/O explanation, but is too small to justify retaining this
additional kernel patch and RAM configuration for the requested short workload.
This does not measure potential benefits on longer/full-vocabulary answers.
No DMR code or extra cache is included in the normal app. A reboot cleared the
temporary kernel state; no taiHEN boot entries were added.

## Retained app change

The normal UI uses four matrix/attention workers when a fourth-core thread can
be created. It probes first, attempts to load the bundled official CapUnlocker
v1.4 only if needed, and otherwise falls back to three workers. Helpers use cores
0, 1, and 3; the inference coordinator uses core 2 during each parallel job and
then restores its original affinity. Quantization and model data are unchanged.

The unchanged CapUnlocker binary, license, source pin, rollback marker, and
runtime scope are documented in [plugins/README.md](../plugins/README.md).
The normal app is included in the refreshed v0.3.0 release VPK. Its matching
source archive is `MeteorLLM-v0.3.0-four-core-source.zip`; the original tag is
preserved, so GitHub's automatically generated tag archives predate this patch.

The [normal-app verification after reboot](plugin-normal-tavi.txt) passed:
automatic CapUnlocker loading returned a positive module UID, four workers were
selected, and the app returned `Tavi` through EOS in **30.139907 seconds**
(first token 25.842369 seconds), with 32 cached and 35 uncached prompt tokens.
The normal app does not query PSVshellPlus, so this final smoke run does not
independently establish its effective clock. Use the matched runs above for
the controlled speed comparison. Free user system RAM was **4 MiB** with the UI
and four-worker engine active. The installed eboot SHA-256 is
`796e5f78605bb13cb91aabc436b829c1bdd03a698acabbd0f78a7d3877214eb4`.

VitaDesk's eboot and `ur0:tai/config.txt` were verified unchanged against their
pre-test SHA-256 backups. The normal app remained open at its completed Tavi
answer. This is a short smoke test, not an exhaustive stability or long-answer
benchmark.
