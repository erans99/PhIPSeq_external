import os

if os.path.exists(__file__.replace('.py', '_local.py')):
    # noinspection PyUnresolvedReferences
    from PhIPSeq_external.Analysis_allergens_and_IEDB.config_local import *

base_path = globals().get('base_path',
                          "XXXX")  # directory with exported files: library_contents.csv, cohort.csv & fold_data.csv
out_path = globals().get('out_path', os.path.join(base_path, "analysis_res"))

MIN_OLIS = 200
