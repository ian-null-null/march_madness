# 1. run this: jupyter nbconvert --to markdown march_madness_showcase.ipynb

# 2: delete the first line of the markdown file and replace with the following:
# ---
# title: March Madness (Basketball) Prediction with Machine Learning
# author: Ian Graham
# date: March 27, 2026
# execute: 
#   enabled: False
# jupyter: python3
# page-layout: full
# toc: true
# number-sections: true
# ---
# 3. find all instances of [png] and replace with []

# 4. run this: quarto render march_madness_showcase.md
!/bin/bash

jupyter nbconvert --to markdown march_madness_showcase.ipynb

sed -i '1d' march_madness_showcase.md
sed -i '1s/^/---\ntitle: March Madness (Basketball) Prediction with Machine Learning\nauthor: Ian Graham\ndate: March 27, 2026\nexecute:\n  enabled: False\njupyter: python3\npage-layout: full\ntoc: true\nnumber-sections: true\n---\n/' march_madness_showcase.md

sed -i 's/\[png\]/[]/g' march_madness_showcase.md

quarto render march_madness_showcase.md