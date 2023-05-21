# Steps for HM encoding

1. Download and compile HM from https://vcgit.hhi.fraunhofer.de/jvet/HM/-/tree/HM-16.21+SCM-8.8
2. create_raw_rgbp.sh --> see VTM: convert input images to RAW RGB format (other formats would also be possible, e.g. YUV444 or YUV420, but different coding configs would be needed for those)
3. create_raw_sequence_cfgs_HEVC.sh: create configuration files for each input.
4. encodeHEVC_SCM_CTC_constIBC.sh: Encode image
5. convert_raw2png.sh: Decode and convert back


Current configurations are applied according to http://phenix.it-sudparis.eu/jct/doc_end_user/current_document.php?id=10221 (both const IBC and complete image as IBC reference area seem to be ok according to Common Test Conditions!)
