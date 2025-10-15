config=/data/xthuang/code/vggt/configs/mono/tum/fr2_xyz.yaml
output=output/tum/xyz

# python demo_colmap.py --config /data/xthuang/code/vggt/configs/mono/tum/fr2_xyz.yaml --output_dir output/tum/xyz
python demo_colmap.py --config $config --output_dir $output
python test_ate.py --config $config --output $output
# python scripts/visualize_attn.py