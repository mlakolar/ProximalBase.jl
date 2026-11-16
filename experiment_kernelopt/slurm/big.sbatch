#!/bin/bash
# Replication of the million-entry cases: node diagnostics, then the formal job script unchanged.
echo "== node $(hostname) $(date)"; numactl --hardware 2>/dev/null | head -12; free -g | head -3
cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null
cat /proc/sys/kernel/numa_balancing 2>/dev/null; grep -E "AnonHugePages|HugePages_Total|MemFree" /proc/meminfo
ps -eo user:12,pcpu,pmem,comm --sort=-pcpu | head -8
bash $KO_ROOT/experiment_kernelopt/slurm/formal.sbatch
rc=$?
echo "== end $(date)"; grep -E "AnonHugePages|MemFree" /proc/meminfo; ps -eo user:12,pcpu,pmem,comm --sort=-pcpu | head -6
exit $rc
