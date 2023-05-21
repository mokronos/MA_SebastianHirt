#!/bin/bash
echo > HEVC_SCC_CTC_constIBC_log.txt
# Encoding formatRGB.cfg + classSCC.cfg
x=37
# for x in {22..37}; do
    # SCIK-TEST
    for filename in ./cfgs_seq/*.cfg; do
        /home/och/ReadersAndCodecs/HM/bin/TAppEncoderStatic -c HEVC_SCC_CTC_cfgs/encoder_intra_main_scc_constIBC.cfg -c "$filename" -f 1 -q $x --BitstreamFile=$(basename "$filename" .cfg)"_CTC_"$x.bin --ReconFile=$(basename "$filename" .cfg)"_CTC_dec_"$x.raw >> HEVC_SCC_CTC_constIBC_log.txt
    done
# done
