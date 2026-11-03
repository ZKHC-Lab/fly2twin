# SPDX-License-Identifier: Apache-2.0
import numpy as np


def aggregate(rows, subsets):
    by_id = {r['frame']:r for r in rows}
    expected = set().union(*(set(v) for v in subsets.values()))
    if len(by_id) != len(rows) or set(by_id) != expected:
        raise ValueError('Evaluation must cover each test ID exactly once')
    result = {}
    for name, ids in subsets.items():
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate test IDs')
        result[name] = {'frames':len(ids)}
        for mode,key in [('fixed','before'),('aligned','after')]:
            result[name][mode] = {metric:float(np.mean([by_id[i][key][metric] for i in ids])) if ids else None
                                  for metric in ('psnr','ssim','lpips')}
    return result
