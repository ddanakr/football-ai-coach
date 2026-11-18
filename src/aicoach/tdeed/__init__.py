"""The `tdeed` stage: runs the T-DEED model on a match video (D-080).

The T-DEED code (GPL-3.0) is never imported or copied here: it is fetched into a cache outside the
repository and its unmodified `inference.py` runs as a separate process (D-046). Only the JSON file
it writes is read, by `aicoach.spotting`.
"""
