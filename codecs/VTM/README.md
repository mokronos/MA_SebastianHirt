# Steps for VTM encoding

1. Download and compile VTM from https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM/-/tree/VTM-17.2?ref_type=tags
2. create_raw_rgbp.sh: convert input images to RAW RGB format (other formats would also be possible, e.g. YUV444 or YUV420, but different coding configs would be needed for those)
3. create_raw_sequence_cfgs.sh: create configuration files for each input.
4. encodeCLassSCCFormatRGB_CTC.sh: Encode image
5. convert_raw2png.sh: Decode and convert back


Current configurations are applied according to https://jvet-experts.org/doc_end_user/current_document.php?id=10546
