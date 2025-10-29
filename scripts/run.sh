config=/data/xthuang/code/vggt/configs/mono/tum/fr1_desk.yaml
output=output/tum/desk

# python demo_colmap.py --config $config --output_dir $output --mode "MLHP"
python demo_colmap.py --config $config --output_dir $output --mode "LTP"
python test_ate.py --config $config --output $output

# watch -n 2 nvidia-smi -i 0 -q -d MEMORY