# RSFomer:  Time Series Transformer for Robust Sports Action Recognition
---

Source code for RSFormer

---

##  Requirements
- python 3.7
- pytorch == 1.13.1
- numpy == 1.21.6
- pykalman == 0.9.7
- scikit-learn == 1.0.2
- tqdm == 4.64.1
---

##  Run the Codes
If training on a custom sports dataset, place your data in the `data` folder (e.g., for a boxing scenario, put the custom boxing action dataset in `data/boxing`).

```bash
python main.py --data_folder boxing --data_path ./data/boxing --layer 3 --slice_per_layer 8 2 2
```

`data_folder`: The name of the folder containing your custom dataset.

`data_path`: The path where your dataset is located. It should contain the following files:

- `X_train.npy`: Training data
- `y_train.npy`: Training data labels
- `X_test.npy`: Testing data
- `y_test.npy`: Testing data labels

`layer`: The number of CNN-TEN pairs employed in the multi-scale feature extraction mechanism of RSFormer.

`slice_per_layer`: This parameter defines the slice window sizes set for each CNN-TEN pair.

Finally, the model parameter file will be stored in subfolders of `test/args.json` path, and the best model will be placed in `test/model.pkl`.

---
##  Open Boxing Action Test Dataset
The boxing action test dataset is located at `data/boxing`, and the boxing action label information can be found in `data/boxing/boxing_label_info.txt`. The test interface is available. You can execute the test by running the following command:
```bash
python test.py --json_path ./test/args.json --model_path ./test/model.pkl
```
