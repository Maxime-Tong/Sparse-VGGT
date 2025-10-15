#!/bin/bash

datasets=('fr1_desk' 'fr2_xyz' 'fr3_office')
methods=('none' 'H' 'HP')
result_file="result.txt"
> "$result_file"  # 清空结果文件

# 写入表头
echo "数据集,模式,RMSE ATE (m)" | tee -a "$result_file"

for dataset in "${datasets[@]}"; do
    config="/data/xthuang/code/vggt/configs/mono/tum/${dataset}.yaml"
    output="output/tum/$(echo "$dataset" | cut -d'_' -f2)"
    
    for mode in "${methods[@]}"; do
        # 运行处理脚本
        python demo_colmap.py --config "$config" --output_dir "$output" --mode "$mode"
        # 提取ATE结果
        ate_result=$(python test_ate.py --config "$config" --output "$output" | grep "RMSE ATE" | awk '{print $3}')
        
        # 简洁输出结果（终端和文件）
        echo "${dataset},${mode},${ate_result}" | tee -a "$result_file"
    done
done

echo "所有任务完成，结果已保存到 $result_file"