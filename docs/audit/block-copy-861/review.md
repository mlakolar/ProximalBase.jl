# Independent Codex review receipt

Reviewer: separate **GPT-6.1-Sol**, **high** effort agent, with independently
read source/tests, generated-code evidence, raw CSVs, summaries and logs.
Review covered the production copy path, correctness references, benchmark
state restoration, historical adapters, allocation measurements, statistical
pairing, provenance and final interpretation. The reviewer performed static
and lightweight data checks; package execution and CI logs are separate evidence.

Substantive findings addressed:

1. Benchmark callbacks captured reassigned variables in boxes. Explicit `let`
   bindings fixed this before final timing.
2. Initial process rotation was unbalanced. Latin rotation replaced it before
   timing; each role appears seven times in each position over 28 rounds.
3. The analyzer verified samples and summarized observations without joining
   them. It now checks exact fixture/observation linkage, seeds and configured
   round×role×case completeness before computing results.
4. Clearing constructed `zero(T)` per entry. Hoisting it once per group preserves
   BigFloat allocation/object reuse. A focused identity/value/state test passes;
   Float native-code equivalence supports retaining the pre-hoist campaign.
5. A focal allocating-copy confirmation crossed the 5% screen. The additional
   balanced 24-round copy-only control includes independent A/A processes for
   both sources; all original observations remain available.

The reviewer independently checked all 33,600 main observations, reproduced all
300 paired median ratios and confirmed allocation bytes/counts match master in
every round. No repeated >5% median regression remained in main confirmation.
The final-source historical fixture adapter and native equivalence artifacts
were also inspected and accepted.

Final copy-control assessment: all four cross-source 81-group medians are
1.011–1.037; pooled ratios are 1.029 at 81 groups and 1.020 at 819 groups.
Both versions allocate 164,120 bytes / 10 objects in every control trial.

Final disposition: **accepted with an explicit reporting limitation**. A small
contextual allocating-copy cost of approximately 1–4% remains possible; no
reproducible >5% median regression or production correctness issue remains.
Report remaining historical raw-copy gaps and preserve the flagged data.
No further benchmark run was requested. Final Linux/current-Julia CI is recorded
on the PR; this review does not authorize an automatic merge.
