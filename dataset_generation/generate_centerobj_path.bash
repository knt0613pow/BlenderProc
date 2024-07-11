root="/dataset/Objaverse-LGM/hf-objaverse-v1/glbs"
output="centerobj_path.txt"

# Find all .glb files under the $root directory and save them to $output
find "$root" -type f -name "*.glb" > "$output"