# Submission Format

You will need to prepare a single `ablation.zip` or `test.zip` file depending on the subset of GRIT you want to evaluate on. The following is the expected directory structure inside the `ablation.zip` file (replace `ablation` by `test` everywhere for `test.zip`) where each task json file contains predictions for the respective task and `params.json` contains parameter count in millions. The `normals/` directory contains each normal prediction saved as an RGB image. 
```
ablation/
|--params.json
|--categorization.json
|--localization.json
|--refexp.json
|--vqa.json
|--segmentation.json
|--keypoint.json
|--normal.json
|--normals/
```

If your model does not make predictions for a particular task, simply omit the correponding json file from the directory. The format for each of the json files are described below.

## params.json
This contains a dictionary with a single key-value pair
```
{
    "params_in_millions": int
}
```

## task.json
Each task json contains a list of dicts, a dict per sample. Each dict contains the following key-value pairs.
```
[
    {
        "example_id" : str,
        "confidence" : float in [0,1],
        "words"      : str,
        "bboxes"     : list of integers [x1,y1,x2,y2],        # box coordinates per instance
        "masks"      : list of masks,                         # rle encoded binary masks per instance
        "points"     : list of integers [x1,y1,...,x17,y17],  # the 17 keypoints per instance)
        "normal"     : str,                                   # path to normal image relative to `normals/`
    }
]
```

Not all keys are required to be present for each task. Here are the tasks that require specific keys to be present:
* `example_id`: all tasks
* `confidence`: all tasks
* `words`: categorization, vqa
* `bboxes`: localization, refexp
* `masks`: segmentation
* `points`: keypoint
* `normal`: normal
