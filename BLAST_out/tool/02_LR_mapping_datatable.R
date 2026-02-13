#===============================================================================
# R脚本: BLAST结果解析与L-R同源映射 (纯data.table版本)
# 功能: 将拟南芥L-R对映射到毛果杨
# 使用方法: Rscript 02_LR_mapping_datatable.R
#===============================================================================

# 清空环境
rm(list = ls())

#-------------------------------------------------------------------------------
# 0. 加载包
#-------------------------------------------------------------------------------
if (!require("data.table")) install.packages("data.table")
library(data.table)

cat("========================================\n")
cat("BLAST结果解析与L-R同源映射\n")
cat("(纯data.table版本 - 无需tidyverse)\n")
cat("========================================\n\n")

#-------------------------------------------------------------------------------
# 1. 参数设置 (★根据需要修改★)
#-------------------------------------------------------------------------------
params <- list(
  # 输入文件
  blast_file    = "blast_output/ath_vs_ptr.txt",
  lr_pair_file  = "LR_pair_ath.csv",
  
  # 过滤阈值 (可调整)
  evalue_cutoff   = 1e-10,   # E-value阈值
  identity_cutoff = 30,      # 序列一致性阈值 (%)
  coverage_cutoff = 50,      # 查询覆盖度阈值 (%)
  
  # 输出文件
  output_dir = "results"
)

# 创建输出目录
dir.create(params$output_dir, showWarnings = FALSE, recursive = TRUE)

#-------------------------------------------------------------------------------
# 2. 读取BLAST结果
#-------------------------------------------------------------------------------
cat("Step 1: 读取BLAST结果...\n")

# BLAST outfmt 6 列名
blast_cols <- c(
  "qseqid",    # 查询序列ID (拟南芥)
  "sseqid",    # 目标序列ID (杨树)
  "pident",    # 序列一致性 (%)
  "length",    # 比对长度
  "mismatch",  # 错配数
  "gapopen",   # gap数
  "qstart",    # 查询起始
  "qend",      # 查询结束
  "sstart",    # 目标起始
  "send",      # 目标结束
  "evalue",    # E-value
  "bitscore",  # Bit score
  "qlen",      # 查询序列长度
  "slen",      # 目标序列长度
  "qcovs"      # 查询覆盖度 (%)
)

blast_raw <- fread(
  params$blast_file,
  header = FALSE,
  col.names = blast_cols,
  sep = "\t"
)

cat(sprintf("  原始比对结果: %d 条\n", nrow(blast_raw)))

#-------------------------------------------------------------------------------
# 3. 清洗序列ID
#-------------------------------------------------------------------------------
cat("\nStep 2: 清洗序列ID...\n")

# 拟南芥ID清洗: "AT1G01010.1|PACid:xxx" -> "AT1G01010"
# 杨树ID清洗: "Potri.001G000100.1.p" -> "Potri.001G000100"

blast_clean <- copy(blast_raw)

# 拟南芥: 提取基因ID
blast_clean[, ath_gene := sub("\\|.*", "", qseqid)]           # 去掉|后面的内容
blast_clean[, ath_gene := sub("\\.\\d+$", "", ath_gene)]      # 去掉.1等版本号

# 杨树: 提取基因locus
blast_clean[, ptr_gene := sub("\\.p$", "", sseqid)]           # 去掉.p后缀
blast_clean[, ptr_gene := sub("\\.\\d+$", "", ptr_gene)]      # 去掉转录本版本号

# 保留原始转录本ID用于追溯
blast_clean[, ath_transcript := sub("\\|.*", "", qseqid)]
blast_clean[, ptr_transcript := sub("\\.p$", "", sseqid)]

cat(sprintf("  唯一拟南芥基因: %d\n", uniqueN(blast_clean$ath_gene)))
cat(sprintf("  唯一杨树基因: %d\n", uniqueN(blast_clean$ptr_gene)))

#-------------------------------------------------------------------------------
# 4. 过滤BLAST结果
#-------------------------------------------------------------------------------
cat("\nStep 3: 过滤BLAST结果...\n")
cat(sprintf("  过滤条件:\n"))
cat(sprintf("    - E-value < %s\n", format(params$evalue_cutoff, scientific = TRUE)))
cat(sprintf("    - Identity >= %d%%\n", params$identity_cutoff))
cat(sprintf("    - Coverage >= %d%%\n", params$coverage_cutoff))

blast_filtered <- blast_clean[
  evalue < params$evalue_cutoff & 
  pident >= params$identity_cutoff & 
  qcovs >= params$coverage_cutoff
]

cat(sprintf("  过滤后比对结果: %d 条 (保留 %.1f%%)\n", 
            nrow(blast_filtered), 
            100 * nrow(blast_filtered) / nrow(blast_raw)))

#-------------------------------------------------------------------------------
# 5. 选择最佳匹配 (Best Hit)
#-------------------------------------------------------------------------------
cat("\nStep 4: 选择最佳匹配...\n")

# 按bitscore降序、evalue升序排序，每个拟南芥基因只保留最佳匹配
setorder(blast_filtered, ath_gene, -bitscore, evalue)
best_hits <- blast_filtered[, .SD[1], by = ath_gene]

cat(sprintf("  最佳匹配对数: %d\n", nrow(best_hits)))

# 保存完整的同源映射表
ortholog_table <- best_hits[, .(
  Ath_Gene = ath_gene,
  Ath_Transcript = ath_transcript,
  Ptr_Gene = ptr_gene,
  Ptr_Transcript = ptr_transcript,
  Identity = pident,
  Coverage = qcovs,
  Evalue = evalue,
  Bitscore = bitscore
)]

fwrite(
  ortholog_table,
  file.path(params$output_dir, "ortholog_mapping_table.csv")
)
cat("  保存: results/ortholog_mapping_table.csv\n")

#-------------------------------------------------------------------------------
# 6. 读取拟南芥L-R对
#-------------------------------------------------------------------------------
cat("\nStep 5: 读取拟南芥L-R对...\n")

lr_ath <- fread(params$lr_pair_file)

# 自动检测列名 (适应不同格式)
col_names <- colnames(lr_ath)
cat(sprintf("  检测到列名: %s\n", paste(col_names, collapse = ", ")))

# 假设前两列是Ligand和Receptor (根据实际文件调整)
if (ncol(lr_ath) >= 2) {
  setnames(lr_ath, 1:2, c("Ligand_Ath", "Receptor_Ath"))
}

# 清洗L-R基因ID (与BLAST结果格式统一)
lr_ath[, Ligand_Ath_clean := sub("\\.\\d+$", "", Ligand_Ath)]
lr_ath[, Receptor_Ath_clean := sub("\\.\\d+$", "", Receptor_Ath)]

cat(sprintf("  L-R对数: %d\n", nrow(lr_ath)))
cat(sprintf("  唯一Ligand: %d\n", uniqueN(lr_ath$Ligand_Ath_clean)))
cat(sprintf("  唯一Receptor: %d\n", uniqueN(lr_ath$Receptor_Ath_clean)))

#-------------------------------------------------------------------------------
# 7. 映射L-R到杨树
#-------------------------------------------------------------------------------
cat("\nStep 6: 映射L-R到杨树...\n")

# 创建映射字典
ath2ptr <- setNames(best_hits$ptr_gene, best_hits$ath_gene)

# 映射Ligand和Receptor
lr_ptr <- copy(lr_ath)
lr_ptr[, Ligand_Ptr := ath2ptr[Ligand_Ath_clean]]
lr_ptr[, Receptor_Ptr := ath2ptr[Receptor_Ath_clean]]

# 标记映射状态
lr_ptr[, Ligand_Mapped := !is.na(Ligand_Ptr)]
lr_ptr[, Receptor_Mapped := !is.na(Receptor_Ptr)]
lr_ptr[, Both_Mapped := Ligand_Mapped & Receptor_Mapped]

#-------------------------------------------------------------------------------
# 8. 统计映射结果
#-------------------------------------------------------------------------------
cat("\nStep 7: 映射统计...\n")

mapping_stats <- list(
  total_pairs = nrow(lr_ptr),
  ligand_mapped = sum(lr_ptr$Ligand_Mapped),
  receptor_mapped = sum(lr_ptr$Receptor_Mapped),
  both_mapped = sum(lr_ptr$Both_Mapped),
  partial_mapped = sum(xor(lr_ptr$Ligand_Mapped, lr_ptr$Receptor_Mapped)),
  none_mapped = sum(!lr_ptr$Ligand_Mapped & !lr_ptr$Receptor_Mapped)
)

cat(sprintf("  总L-R对数:           %d\n", mapping_stats$total_pairs))
cat(sprintf("  Ligand映射成功:      %d (%.1f%%)\n", 
            mapping_stats$ligand_mapped,
            100 * mapping_stats$ligand_mapped / mapping_stats$total_pairs))
cat(sprintf("  Receptor映射成功:    %d (%.1f%%)\n", 
            mapping_stats$receptor_mapped,
            100 * mapping_stats$receptor_mapped / mapping_stats$total_pairs))
cat(sprintf("  双端映射成功:        %d (%.1f%%)\n", 
            mapping_stats$both_mapped,
            100 * mapping_stats$both_mapped / mapping_stats$total_pairs))
cat(sprintf("  部分映射:            %d\n", mapping_stats$partial_mapped))
cat(sprintf("  完全未映射:          %d\n", mapping_stats$none_mapped))

#-------------------------------------------------------------------------------
# 9. 输出结果文件
#-------------------------------------------------------------------------------
cat("\nStep 8: 保存结果文件...\n")

# 9.1 完整映射结果 (包含所有信息)
# 为Ligand添加比对信息
ligand_info <- best_hits[, .(
  ath_gene,
  Ligand_Identity = pident,
  Ligand_Coverage = qcovs,
  Ligand_Evalue = evalue,
  Ligand_Bitscore = bitscore
)]

# 为Receptor添加比对信息
receptor_info <- best_hits[, .(
  ath_gene,
  Receptor_Identity = pident,
  Receptor_Coverage = qcovs,
  Receptor_Evalue = evalue,
  Receptor_Bitscore = bitscore
)]

lr_ptr_full <- copy(lr_ptr)
lr_ptr_full <- merge(lr_ptr_full, ligand_info, 
                     by.x = "Ligand_Ath_clean", by.y = "ath_gene", 
                     all.x = TRUE)
lr_ptr_full <- merge(lr_ptr_full, receptor_info, 
                     by.x = "Receptor_Ath_clean", by.y = "ath_gene", 
                     all.x = TRUE)

fwrite(
  lr_ptr_full,
  file.path(params$output_dir, "LR_pair_ptr_full.csv")
)
cat("  ✓ 完整结果: results/LR_pair_ptr_full.csv\n")

# 9.2 简洁版杨树L-R对 (仅双端映射成功的)
lr_ptr_clean <- lr_ptr[Both_Mapped == TRUE, .(
  Ligand_Ath = Ligand_Ath_clean,
  Receptor_Ath = Receptor_Ath_clean,
  Ligand_Ptr,
  Receptor_Ptr
)]

fwrite(
  lr_ptr_clean,
  file.path(params$output_dir, "LR_pair_ptr_clean.csv")
)
cat(sprintf("  ✓ 简洁版(双端映射): results/LR_pair_ptr_clean.csv (%d对)\n", 
            nrow(lr_ptr_clean)))

# 9.3 纯杨树L-R对 (可直接用于下游分析)
lr_ptr_final <- unique(lr_ptr_clean[, .(
  Ligand = Ligand_Ptr,
  Receptor = Receptor_Ptr
)])

fwrite(
  lr_ptr_final,
  file.path(params$output_dir, "LR_pair_Ptrichocarpa.csv")
)
cat(sprintf("  ✓ 最终杨树L-R: results/LR_pair_Ptrichocarpa.csv (%d对)\n", 
            nrow(lr_ptr_final)))

# 9.4 未映射的基因列表 (便于排查)
unmapped_ligands <- unique(lr_ath[!Ligand_Ath_clean %in% names(ath2ptr), Ligand_Ath_clean])
unmapped_receptors <- unique(lr_ath[!Receptor_Ath_clean %in% names(ath2ptr), Receptor_Ath_clean])

writeLines(unmapped_ligands, file.path(params$output_dir, "unmapped_ligands.txt"))
writeLines(unmapped_receptors, file.path(params$output_dir, "unmapped_receptors.txt"))
cat(sprintf("  ✓ 未映射基因: unmapped_ligands.txt (%d), unmapped_receptors.txt (%d)\n",
            length(unmapped_ligands), length(unmapped_receptors)))

#-------------------------------------------------------------------------------
# 10. 生成报告
#-------------------------------------------------------------------------------
report <- sprintf("
================================================================================
                    L-R同源映射报告
================================================================================

日期: %s

【输入文件】
  BLAST结果: %s
  L-R对文件: %s

【过滤参数】
  E-value阈值:    < %s
  Identity阈值:   >= %d%%
  Coverage阈值:   >= %d%%

【BLAST统计】
  原始比对数:     %d
  过滤后比对数:   %d
  最佳匹配对数:   %d

【L-R映射统计】
  拟南芥L-R对:    %d
  双端映射成功:   %d (%.1f%%)
  部分映射:       %d
  完全未映射:     %d

【输出文件】
  1. LR_pair_ptr_full.csv        - 完整映射结果(含所有比对信息)
  2. LR_pair_ptr_clean.csv       - 简洁版(双端映射,含拟南芥ID)
  3. LR_pair_Ptrichocarpa.csv    - 最终杨树L-R对(可直接使用)
  4. ortholog_mapping_table.csv  - 完整同源基因映射表
  5. unmapped_ligands.txt        - 未映射的Ligand基因
  6. unmapped_receptors.txt      - 未映射的Receptor基因

================================================================================
",
  Sys.time(),
  params$blast_file,
  params$lr_pair_file,
  format(params$evalue_cutoff, scientific = TRUE),
  params$identity_cutoff,
  params$coverage_cutoff,
  nrow(blast_raw),
  nrow(blast_filtered),
  nrow(best_hits),
  mapping_stats$total_pairs,
  mapping_stats$both_mapped,
  100 * mapping_stats$both_mapped / mapping_stats$total_pairs,
  mapping_stats$partial_mapped,
  mapping_stats$none_mapped
)

cat(report)
writeLines(report, file.path(params$output_dir, "mapping_report.txt"))

cat("\n========================================\n")
cat("映射完成!\n")
cat("========================================\n")
