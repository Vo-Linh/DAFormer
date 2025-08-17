# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.models.uda.dacs import DACS
from mmseg.models.uda.smdacs import SMDACS
from mmseg.models.uda.trust_smdacs import TrustAwareSMDACS
from mmseg.models.uda.trust_dev import TrustAwareDev
from mmseg.models.uda.assda import AssDA
from mmseg.models.uda.ass_uda import ASSUDA
__all__ = ['DACS', 'SMDACS', 'TrustAwareSMDACS', 'TrustAwareDev', 'ASSUDA']
