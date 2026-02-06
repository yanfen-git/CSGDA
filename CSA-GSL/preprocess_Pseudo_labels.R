####################

#1 data数据集更换基因名保存

####################

# --- 配置 ---
drug_list <- c(
  "Gefitinib", "Afatinib", "AR-42", "Cetuximab", "Etoposide",
  "NVP-TAE684", "PLX4720", "PLX4720_451Lu", "Sorafenib", "Vorinostat"
)

map_file <- "C:/Users/yf/Desktop/preprocessing/geneDict/HGNC_symbol_all_genes.tsv"
input_base <- "C:/Users/yf/Desktop/preprocessing/data"
output_base <- "C:/Users/yf/Desktop/preprocessing/newCS4"
file_types <- c("Source", "Target")

# --- 加载基因映射表 ---
map <- read.delim(map_file, header = TRUE, check.names = FALSE, quote = "", stringsAsFactors = FALSE)
ncbi_to_symbol <- setNames(map$`Approved symbol`, map$`NCBI Gene ID`)

# --- 批处理 ---
for (drug in drug_list) {
  for (type in file_types) {
    
    # 正确拼接文件名（根据 Source / Target 使用不同前缀）
    if (type == "Source") {
      filename <- paste0("Source_exprs_resp_z.", drug, ".tsv")
    } else if (type == "Target") {
      filename <- paste0("Target_expr_resp_z.", drug, ".tsv")
    }
    
    # 构建完整路径
    input_file <- file.path(input_base, drug, filename)
    output_dir <- file.path(output_base, drug)
    output_file <- file.path(output_dir, filename)
    
    if (!file.exists(input_file)) {
      cat("⚠ 文件不存在，跳过：", input_file, "\n")
      next
    }
    
    # 读取表达矩阵
    expr <- read.table(input_file, header = TRUE, sep = "\t", check.names = FALSE)
    
    ######################################################
    # === 新增：数据清洗 ===
    ######################################################
    clean_numeric_data <- function(expr, type) {
      # 确定数值列范围
      start_col <- if(type == "Source") 4 else 3  # Source有3列元数据，Target有2列
      num_cols <- start_col:ncol(expr)
      
      # 转换并清洗数值列
      for(j in num_cols) {
        # 转换为字符型以处理特殊值
        col_vals <- as.character(expr[[j]])
        
        # 替换特殊字符为NA
        col_vals[col_vals == "-"] <- NA
        col_vals[col_vals == ""] <- NA
        col_vals[col_vals == "NA"] <- NA
        
        # 转换为数值型
        col_vals <- as.numeric(col_vals)
        
        # 填充缺失值
        if(any(is.na(col_vals))) {
          col_mean <- mean(col_vals, na.rm = TRUE)
          col_vals[is.na(col_vals)] <- col_mean
        }
        expr[[j]] <- col_vals
      }
      return(expr)
    }
    
    # 执行清洗
    expr <- clean_numeric_data(expr, type)
    ######################################################
    
    # 替换列名（从第3列起）
    colnames(expr)[3:ncol(expr)] <- ifelse(
      colnames(expr)[3:ncol(expr)] %in% names(ncbi_to_symbol),
      ncbi_to_symbol[colnames(expr)[3:ncol(expr)]],
      colnames(expr)[3:ncol(expr)]  # 未匹配的列保持不变
    )
    
    # 创建目录
    if (!dir.exists(output_dir)) {
      dir.create(output_dir, recursive = TRUE)
      cat("📁 创建目录：", output_dir, "\n")
    }
    
    # 保存
    write.table(expr, file = output_file, sep = "\t", quote = FALSE, row.names = FALSE)
    cat("✅ 已保存：", output_file, "\n")
  }
}




####################

#2 整理细胞状态 → 基因名映射字典

####################


# 设置状态文件目录
status_dir <- "C:/Users/yf/Desktop/preprocessing/cellStatus"

# 获取所有 txt 文件路径
status_files <- list.files(status_dir, pattern = "\\.txt$", full.names = TRUE)

# 初始化列表
state_gene_dict <- list()

# 逐个文件处理
for (file_path in status_files) {
  # 从文件名中提取状态名（去掉路径和扩展名）
  state_name <- tools::file_path_sans_ext(basename(file_path))
  
  # 读取文件（假设有列名，列名包含 GeneName）
  df <- read.table(file_path, sep = "\t", header = TRUE)
  
  # 你可以根据实际情况指定列名，这里默认列名为 GeneName
  gene_names <- as.character(df$GeneName)
  
  # 存入列表
  state_gene_dict[[state_name]] <- gene_names
}

# 打印检查
cat("✔ 读取到的状态名称:\n")
print(names(state_gene_dict))
cat("✔ Proliferation 状态的前几个基因：\n")
print(head(state_gene_dict[["Proliferation"]]))
saveRDS(state_gene_dict, file = "C:/Users/yf/Desktop/preprocessing/geneDict/state_gene_dict.rds")


####################

#3 筛选出状态相关基因列（与表达矩阵取交集）

####################

# 配置路径与状态基因字典
drug_list <- c("Gefitinib", "Afatinib", "AR-42", "Cetuximab", "Etoposide",
               "NVP-TAE684", "PLX4720", "PLX4720_451Lu", "Sorafenib", "Vorinostat")
data_dir <- "C:/Users/yf/Desktop/preprocessing/newCS4"
state_gene_dict <- readRDS("C:/Users/yf/Desktop/preprocessing/geneDict/state_gene_dict.rds")

for (drug in drug_list) {
  for (type in c("Source", "Target")) {
    
    # 构建文件路径
    filename <- if (type == "Source") {
      paste0("Source_exprs_resp_z.", drug, ".tsv")
    } else {
      paste0("Target_expr_resp_z.", drug, ".tsv")
    }
    expr_path <- file.path(data_dir, drug, filename)
    
    if (!file.exists(expr_path)) {
      cat("⚠ 跳过不存在文件：", expr_path, "\n")
      next
    }
    
    cat("📂 正在处理：", expr_path, "\n")
    
    # 读取表达矩阵
    expr <- read.table(expr_path, header = TRUE, sep = "\t", check.names = FALSE)
    
    ######################################################
    # === 新增：数据清洗 ===
    ######################################################
    clean_numeric_data <- function(expr, type) {
      # ... [与步骤#1相同的函数内容] ...
    }
    
    # 执行清洗
    expr <- clean_numeric_data(expr, type)
    ######################################################
    
    colnames(expr) <- make.names(colnames(expr), unique = TRUE)
    
    # ✅ 动态设置 meta_cols
    if (grepl("Source", basename(expr_path))) {
      meta_cols <- colnames(expr)[1:3]
    } else {
      meta_cols <- colnames(expr)[1:2]
    }
    
    expr_by_state <- list()
    
    for (state in names(state_gene_dict)) {
      cat("➡️ 状态：", state, "\n")
      
      gene_list <- state_gene_dict[[state]]
      matched_genes <- intersect(gene_list, colnames(expr))
      selected_cols <- c(meta_cols, matched_genes)
      selected_cols_final <- intersect(selected_cols, colnames(expr))
      
      if (length(selected_cols_final) > length(meta_cols)) {
        expr_state <- expr[, selected_cols_final, drop = FALSE]
        expr_by_state[[state]] <- expr_state
        cat("✅ 匹配基因数：", length(matched_genes), "\n")
      } else {
        cat("⚠ 无有效基因列，跳过。\n")
      }
    }
    
    # ✅ 保存 expr_by_state 为 .rds 文件
    rds_path <- file.path(data_dir, drug, paste0(type, "_expr_by_state.rds"))
    saveRDS(expr_by_state, file = rds_path)
    cat("💾 已保存 expr_by_state 至：", rds_path, "\n\n")
  }
}

####################
# 4 + 5 计算状态得分 + 合并伪标签回原始表达矩阵并保存
####################

# ✅ 配置标签名和编号映射
selected_labels <- c("Cell_Cycle", "DNA_repair", "EMT", "Inflammation")
label2id <- setNames(0:3, selected_labels)

# ✅ 用于后续伪标签训练输入
for (drug in drug_list) {
  for (type in c("Source", "Target")) {
    
    # 构建路径
    filename <- if (type == "Source") {
      paste0("Source_exprs_resp_z.", drug, ".tsv")
    } else {
      paste0("Target_expr_resp_z.", drug, ".tsv")
    }
    
    expr_path <- file.path(data_dir, drug, filename)
    if (!file.exists(expr_path)) {
      cat("⚠ 表达矩阵不存在，跳过：", expr_path, "\n")
      next
    }
    
    # 读取表达矩阵（列名已替换）
    expr <- read.table(expr_path, header = TRUE, sep = "\t", check.names = FALSE)
    colnames(expr) <- make.names(colnames(expr), unique = TRUE)
    
    # 加载状态子集表达矩阵
    rds_path <- file.path(data_dir, drug, paste0(type, "_expr_by_state.rds"))
    if (!file.exists(rds_path)) {
      cat("⚠ 状态表达数据不存在，跳过：", rds_path, "\n")
      next
    }
    
    expr_by_state <- readRDS(rds_path)
    
    # ---------- 第 4 步：计算每个状态得分 ----------
    state_score_list <- list()
    
    for (state in names(expr_by_state)) {
      df_state <- expr_by_state[[state]]
      n_meta <- if (grepl("IC50", colnames(df_state)[3], ignore.case = TRUE)) 3 else 2
      
      ######################################################
      # === 新增：确保表达矩阵为数值型 ===
      ######################################################
      # 提取表达部分
      expr_mat <- df_state[, (n_meta + 1):ncol(df_state), drop = FALSE]
      
      # 确保所有列为数值型
      for(j in 1:ncol(expr_mat)) {
        if(!is.numeric(expr_mat[, j])) {
          expr_mat[, j] <- as.numeric(as.character(expr_mat[, j]))
        }
        
        # 处理可能的缺失值
        if(any(is.na(expr_mat[, j]))) {
          col_mean <- mean(expr_mat[, j], na.rm = TRUE)
          expr_mat[is.na(expr_mat[, j]), j] <- col_mean
        }
      }
      ######################################################
      
      # expr_mat <- df_state[, (n_meta + 1):ncol(df_state), drop = FALSE]
      scores <- rowMeans(expr_mat, na.rm = TRUE)
      names(scores) <- df_state[[1]]
      state_score_list[[state]] <- scores
    }
    
    # ---------- 第 5 步：生成伪标签 ----------
    score_df <- as.data.frame(do.call(cbind, state_score_list))
    rownames(score_df) <- names(state_score_list[[1]])
    pseudo_labels <- colnames(score_df)[apply(score_df, 1, which.max)]
    
    # ⚠️ 新增：转换为编号（非 8 类标记为 NA）
    label_ids <- ifelse(pseudo_labels %in% selected_labels,
                        label2id[pseudo_labels],
                        NA)
    
    result_df <- data.frame(CellID = rownames(score_df),
                            PseudoLabel = pseudo_labels,
                            LabelID = label_ids,
                            score_df,
                            row.names = NULL)
    
    # ---------- 合并伪标签回原始表达矩阵 ----------
    expr$CellID <- expr[[1]]
    expr_with_label <- merge(expr, result_df[, c("CellID", "PseudoLabel", "LabelID")],
                             by = "CellID", all.x = TRUE)
    
    # 保存最终带伪标签的表达矩阵
    final_out_path <- file.path(data_dir, drug, paste0(type, "_exprs_with_label.tsv"))
    write.table(expr_with_label, file = final_out_path, sep = "\t", quote = FALSE, row.names = FALSE)
    
    cat("✅ 已生成伪标签 + LabelID 并保存至：", final_out_path, "\n\n")
  }
}

####################
# 6 保存伪标签映射表
####################


# 保存为 DataFrame
label_map_df <- data.frame(PseudoLabel = names(label2id),
                           LabelID = as.integer(label2id))

# 保存路径（根据你的目录结构）
write.csv(label_map_df,
          file = "C:/Users/yf/Desktop/preprocessing/geneDict/pseudo_label_mapping_5class.csv",
          row.names = FALSE)

cat("✅ 已保存伪标签编号映射至：geneDict/pseudo_label_mapping_5class.csv")


########################################

#⚠ ⚠ ⚠ ⚠ ⚠ ⚠ ⚠ ⚠ 一些测试 不需要运行⚠ ⚠ ⚠ ⚠ ⚠ ⚠ ⚠ ⚠ 

########################################

####################

#4 对每个状态计算一个细胞的“状态表达得分”，筛选出当前细胞 × 当前状态基因子集的表达矩阵，对每一行（每个细胞）计算它这些基因的平均表达量

# expr_by_state 是一个 list，每个状态一个子表达矩阵（含样本信息 + 状态相关基因）
# 列顺序：前2或3列是 meta 信息，之后是状态相关基因表达值

####################

state_score_list <- list()

for (state in names(expr_by_state)) {
  df_state <- expr_by_state[[state]]
  
  # 假设 meta 信息占前2列或3列，根据列数判断
  n_meta <- if (grepl("IC50", colnames(df_state)[3], ignore.case = TRUE)) 3 else 2
  
  # 提取表达部分（即状态相关基因）
  expr_mat <- df_state[, (n_meta + 1):ncol(df_state), drop = FALSE]
  
  # 计算每个细胞的平均表达作为得分
  scores <- rowMeans(expr_mat, na.rm = TRUE)
  
  # 保留与样本对应关系
  names(scores) <- df_state[[1]]  # 第一列一般是样本ID
  state_score_list[[state]] <- scores
}

cat("✅ 所有状态得分已计算完毕。\n")

####################

#5 合并所有状态得分 → 找最大得分对应的状态作为伪标签

####################

# 将所有状态得分合并为一个矩阵
score_df <- as.data.frame(do.call(cbind, state_score_list))

# 补充：确保行名为 cell ID
rownames(score_df) <- names(state_score_list[[1]])

# 输出检查（每个细胞在每个状态的得分）
cat("📊 状态得分矩阵前几行：\n")
print(head(score_df))

# 每个细胞选择得分最高的状态作为伪标签
pseudo_labels <- colnames(score_df)[apply(score_df, 1, which.max)]

# 创建结果表（细胞ID + 伪标签 + 所有状态得分）
result_df <- cbind(CellID = rownames(score_df),
                   PseudoLabel = pseudo_labels,
                   score_df)

# 输出结果示例
cat("🎯 生成伪标签结果预览：\n")
print(head(result_df))

####################

#6 独立统计伪标签数量

####################

# 配置
drug_list <- c("Gefitinib", "Afatinib", "AR-42", "Cetuximab", "Etoposide",
               "NVP-TAE684", "PLX4720", "PLX4720_451Lu", "Sorafenib", "Vorinostat")
data_dir <- "C:/Users/yf/Desktop/preprocessing/newCS4"
file_types <- c("Source", "Target")

# 创建总表格用于汇总所有drug的标签统计（可选）
all_counts <- data.frame()

for (drug in drug_list) {
  for (type in file_types) {
    
    # 拼接文件路径（含伪标签的最终表达矩阵）
    filename <- paste0(type, "_exprs_with_label.tsv")
    file_path <- file.path(data_dir, drug, filename)
    
    if (!file.exists(file_path)) {
      cat("⚠ 文件不存在，跳过：", file_path, "\n")
      next
    }
    
    # 读取表达数据
    expr <- read.table(file_path, header = TRUE, sep = "\t", check.names = FALSE)
    
    # 统计伪标签频次
    label_count <- table(expr$PseudoLabel)
    
    cat("📊", drug, type, "伪标签计数：\n")
    print(label_count)
    
    # 保存为 CSV
    out_csv <- file.path(data_dir, drug, paste0(type, "_pseudo_label_count.csv"))
    write.csv(as.data.frame(label_count), file = out_csv, row.names = FALSE)
    
    # 可选：加入总统计表
    df_temp <- data.frame(Drug = drug, Type = type,
                          State = names(label_count),
                          Count = as.vector(label_count))
    all_counts <- rbind(all_counts, df_temp)
  }
}

# 读取前面保存的伪标签统计汇总文件
all_counts <- read.csv("C:/Users/yf/Desktop/preprocessing/newCS4/all_pseudo_label_counts.csv")

# 按状态累加细胞数
total_counts_per_state <- aggregate(Count ~ State, data = all_counts, FUN = sum)

# 降序排列，方便选前8个
total_counts_per_state <- total_counts_per_state[order(-total_counts_per_state$Count), ]

# 输出总计数
print(total_counts_per_state)

# 可选：保存成 CSV
write.csv(total_counts_per_state,
          "C:/Users/yf/Desktop/preprocessing/newCS4/pseudolabel_total_counts_per_state.csv",
          row.names = FALSE)




# 配置路径
data_dir <- "C:/Users/yf/Desktop/preprocessing/newCS4"
drug_list <- c("Gefitinib", "Afatinib", "AR-42", "Cetuximab", "Etoposide",
               "NVP-TAE684", "PLX4720", "PLX4720_451Lu", "Sorafenib", "Vorinostat")
file_types <- c("Source", "Target")

for (drug in drug_list) {
  for (type in file_types) {
    
    input_file <- file.path(data_dir, drug, paste0(type, "_exprs_with_label.tsv"))
    output_file <- file.path(data_dir, drug, paste0(type, "_exprs_with_label_", drug, ".tsv"))
    
    if (!file.exists(input_file)) {
      cat("⚠ 文件不存在，跳过：", input_file, "\n")
      next
    }
    
    df <- read.table(input_file, header = TRUE, sep = "\t", check.names = FALSE)
    
    # 删除多余 X 列（若与 CellID 相同）
    if ("X" %in% colnames(df) && "CellID" %in% colnames(df)) {
      if (all(df$X == df$CellID)) {
        df$X <- NULL
        cat("🧹 已删除重复列 X：", input_file, "\n")
      } else {
        cat("⚠ X 列存在但不完全等于 CellID，未删除：", input_file, "\n")
      }
    }
    
    # 保存新版本，文件名中包含 drug 名
    write.table(df, file = output_file, sep = "\t", quote = FALSE, row.names = FALSE)
    cat("✅ 新文件已保存为：", output_file, "\n\n")
  }
}



