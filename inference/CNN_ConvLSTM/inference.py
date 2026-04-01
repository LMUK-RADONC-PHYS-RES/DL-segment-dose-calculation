"""CNN-ConvLSTM segment-dose inference entry point."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "model", "CNN_ConvLSTM"))

from model import CNN_ConvLSTM
from pipeline import build_argparser, run

if __name__ == "__main__":
    args    = build_argparser("CNN-ConvLSTM").parse_args()
    out_dir = args.out_dir or os.path.join("output", args.patient)
    run(
        model_cls     = CNN_ConvLSTM,
        out_filename  = "prediction_ConvLSTM.mha",
        patient       = args.patient,
        sim_root      = args.sim_root,
        seg_dir       = args.seg_dir,
        ct_root       = args.ct_root,
        out_dir       = out_dir,
        model_weights = args.model_weights,
        device        = args.device,
        batch_size    = args.batch_size,
        mode          = args.mode,
        allow_tf32    = not args.no_tf32,
        use_fp16      = not args.fp32,
        print_details = args.print_details,
    )
