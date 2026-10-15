#!/usr/bin/env python3
"""Render the actual GEOS-FP/GEOS-replay/v4.1 workflow; no HRRR substitution."""
import argparse
from merraflow.workflow_maps_v4_1 import from_evaluation,load_bundle,render
from merraflow.workflow_archive_v4_1 import make
from merraflow.evaluate_all_v4_1 import read_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--evaluation',help='Saved numerical v4.1 evaluation')
    group.add_argument('--bundle',help='Saved GEOS workflow_data.npz')
    p.add_argument('--presentation',default='present_last_v4_1',help='Default: original GEOS/v4.1 map PDFs')
    p.add_argument('--case-id',default='20260307_1930')
    p.add_argument('--output',default='output/figures/workflow_geos_v4_1')
    p.add_argument('--dpi',type=int,default=300)
    p.add_argument('--zoom',default='-110,-102,35,41.5')
    p.add_argument('--no-features',action='store_true')
    a=p.parse_args()
    if a.evaluation or a.bundle:
        case=from_evaluation(a.evaluation,read_manifest(a.evaluation)) if a.evaluation else load_bundle(a.bundle)
        zoom=tuple(float(x) for x in a.zoom.split(','))
        if len(zoom)!=4 or zoom[0]>=zoom[1] or zoom[2]>=zoom[3]:raise ValueError('Invalid zoom')
        print(render(case,a.output,a.dpi,zoom,features=not a.no_features))
    else:print(make(a.presentation,a.case_id,a.output,a.dpi))


if __name__=='__main__':main()
