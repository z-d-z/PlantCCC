#!/bin/bash
#===============================================================================
# BLAST+ 蛋白质同源比对流程
# 将拟南芥(Arabidopsis thaliana)蛋白映射到毛果杨(Populus trichocarpa)
#===============================================================================

set -e

#-------------------------------------------------------------------------------
# 配置参数 (根据实际情况修改)
#-------------------------------------------------------------------------------
ATH_PROTEIN="Athaliana_167_protein_primaryTranscriptOnly.fa"
PTR_PROTEIN="Ptrichocarpa_533_v4.1.protein.fa"

# BLAST参数
EVALUE="1e-10"
NUM_THREADS=8
MAX_TARGET_SEQS=10

# 输出目录
mkdir -p blast_db
mkdir -p blast_output

#-------------------------------------------------------------------------------
# Step 1: 构建杨树蛋白质BLAST数据库
#-------------------------------------------------------------------------------
echo "=========================================="
echo "Step 1: 构建杨树蛋白质BLAST数据库"
echo "=========================================="

makeblastdb \
    -in ${PTR_PROTEIN} \
    -dbtype prot \
    -out blast_db/ptr_protein \
    -parse_seqids

echo "✓ 数据库构建完成"

#-------------------------------------------------------------------------------
# Step 2: 运行BLASTp比对 (拟南芥 vs 杨树)
#-------------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Step 2: 运行BLASTp比对"
echo "=========================================="

blastp \
    -query ${ATH_PROTEIN} \
    -db blast_db/ptr_protein \
    -out blast_output/ath_vs_ptr.txt \
    -evalue ${EVALUE} \
    -num_threads ${NUM_THREADS} \
    -max_target_seqs ${MAX_TARGET_SEQS} \
    -outfmt "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen qcovs"

echo "✓ BLAST比对完成"
echo "结果保存至: blast_output/ath_vs_ptr.txt"

#-------------------------------------------------------------------------------
# Step 3: 统计信息
#-------------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Step 3: 比对统计"
echo "=========================================="
echo "总比对数: $(wc -l < blast_output/ath_vs_ptr.txt)"
echo "唯一query数: $(cut -f1 blast_output/ath_vs_ptr.txt | sort -u | wc -l)"
echo "唯一subject数: $(cut -f2 blast_output/ath_vs_ptr.txt | sort -u | wc -l)"

echo ""
echo "=========================================="
echo "BLAST完成! 请运行R脚本进行L-R映射"
echo "=========================================="
