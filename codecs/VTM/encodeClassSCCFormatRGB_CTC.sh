#!/bin/bash
echo > CTC_log.txt
# Encoding formatRGB.cfg + classSCC.cfg

x=37
#for x in {22..37}; do
for filename in ./cfgs_seq/*.cfg; do
    ~/ReadersAndCodecs/VVCSoftware_VTM/bin/EncoderAppStatic -c CTC_cfgs/encoder_intra_vtm.cfg -c "$filename" -c CTC_cfgs/classSCC.cfg -c CTC_cfgs/formatRGB.cfg -f 1 -q $x --BitstreamFile=$(basename "$filename" .cfg)"_CTC_"$x.bin --ReconFile=$(basename "$filename" .cfg)"_CTC_dec_"$x.raw >> CTC_log.txt
done
#done
