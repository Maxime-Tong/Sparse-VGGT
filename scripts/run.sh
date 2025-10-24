config=/data/xthuang/code/vggt/configs/mono/tum/fr2_xyz.yaml
output=output/tum/xyz

# python demo_colmap.py --config $config --output_dir $output --mode "MLHP"
python demo_colmap.py --config $config --output_dir $output --mode "LHP"
python test_ate.py --config $config --output $output