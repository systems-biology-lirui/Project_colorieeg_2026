import sys
import os

filepath = 'testcode/task1_paired_decoding/plot_condition_cosine_similarity.py'
with open(filepath, 'r') as f:
    content = f.read()

# Replace the build_parser function to handle the conflict
new_build_parser = '''
def build_parser():
    parser = build_common_parser('Condition similarity over time between color and gray means, using cosine similarity or Mahalanobis distance.')
    # Remove existing --metric if it exists
    for action in parser._actions:
        if "--metric" in action.option_strings:
            parser._handle_conflict_resolve(None, [("--metric", action)])
    
    parser.add_argument("--selected-categories", default="all", help="Comma-separated category names or indices (1-based). Use all for every category.")
    parser.add_argument("--smooth-win", type=int, default=DEFAULT_SMOOTH_WIN, help="Moving-average window on the time axis before similarity or distance computation.")
    parser.add_argument("--metric", choices=("cosine", "mahalanobis"), default=DEFAULT_METRIC, help="Similarity metric to compute over time.")
    return parser
'''

import re
pattern = r'def build_parser\(\):.*?return parser'
# The dotall flag is needed to match across multiple lines
fixed_content = re.sub(pattern, new_build_parser, content, flags=re.DOTALL)

with open(filepath, 'w') as f:
    f.write(fixed_content)
