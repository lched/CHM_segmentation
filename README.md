# CHM_segmentation
Estimate segments boundaries of a song. Does not estimate labels.\n
Based on :\n
    C. Gaudefroy H. Papadopoulos and M. Kowalski, “A Multi-Dimensional Meter-Adaptive Method For Automatic Segmentation Of Music”, in CBMI 2015.
Default configuration should reproduce the method presented in this paper.
The audio file should be in a dataset organized as follows (consistent with the Music Structure Analysis Framework):\n
  ./dataset_folder
      /audio
      /estimations (empty initially)
      /features (empty initially)
      /references
To analyse a single file outside of a dataset, segment_full_analysis.py (coming soon) shoud be used instaed.
