param(
  [string]$OutputFolder = ".\ver6_6_final"
)
python .\p10_feed_composition_robustness.py --input-folder ".\data\input_folder" --output-folder $OutputFolder --save-mode rich --ram-mode light --level comprehensive --n-jobs 1 --uncertainty-draws 2000 --fresh-run
