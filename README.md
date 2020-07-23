# CHM_segmentation
Estimate segments boundaries of a song. Does not estimate labels.
Based on :
    *C. Gaudefroy H. Papadopoulos and M. Kowalski, “A Multi-Dimensional Meter-Adaptive Method For Automatic Segmentation Of Music”, in CBMI 2015.*
Default configuration should reproduce the method presented in this paper.
datasets_segmentation_loops.py contains ready-to-use functions to evaluate the segmentation on one of the following datasets:
- Beatles
- DTL1000
- Harmonix
The audio file should be in a dataset organized as follows (consistent with [MSAF](https://msaf.readthedocs.io/en/latest/datasets.html)):

     ./dataset_folder
	     /audio
	     /estimations (empty initially)
	     /features (empty initially)
	     /references

To analyse a single file outside of a dataset, segment_full_analysis.py shoud be used instead.
