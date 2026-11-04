"""Run the labeling pipeline. See `python -m taxon.labeling --help`."""

import sys

from taxon.labeling.cli import main

sys.exit(main())
