KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0 OMP_NUM_THREADS=28 numactl --physcpubind=0-27 --membind=0 python dummy_transformer.py