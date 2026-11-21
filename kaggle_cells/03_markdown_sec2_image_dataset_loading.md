## 2. Image Dataset Loading & Preprocessing

Class-folder names containing `fresh` map to label 1; `stale/rotten/spoiled/discard`
names map to label 0. Images are resized to 224x224, rescaled to [0,1], and split
into training and validation partitions; per-dataset counts are reported. The
discovery helpers walk every attached input root and merge all folders into one
unified image dataset.
