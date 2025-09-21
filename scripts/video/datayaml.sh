echo "datasets:" > datasets.yaml
for file in /your_dataset_dir/llava_ov/*.json; do
  if [[ "$file" == *"special"* ]]; then
    sampling="first:10%"
  else
    sampling="all"
  fi
  echo "  - json_path: $(realpath "$file")" >> datasets.yaml
  echo "    sampling_strategy: \"$sampling\"" >> datasets.yaml
done

for file in /your_dataset_dir/LLaVA-Video-178K/json/*.json; do
  if [[ "$file" == *"special"* ]]; then
    sampling="first:10%"
  else
    sampling="all"
  fi
  echo "  - json_path: $(realpath "$file")" >> datasets.yaml
  echo "    sampling_strategy: \"$sampling\"" >> datasets.yaml
done

