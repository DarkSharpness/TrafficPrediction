# How to use our model

All the shell codes used are place in the root directory of this repository.

## Data Installation

The fastest way to get all the data required without compiling any C++ code is to run the following script:

```shell
chmod +x simple.sh
./simple.sh
```

To learn more about the installation details, you may read [this](install.md).

## Model Training

Open the file `Run.py` in the root directory, set the name, save path and other necessary configs.

Then you can run the `Run.py` to train the model:

```shell
python Run.py
```

If it's too time consuming to train the model, you can download the pre-trained model in the release of this repository,
and put it in the `save` directory.

Then you should modify the path in the `Run.py` to load the pre-trained model:

Still, you need some extra fine-tuning to make the model work with your data. Usually, you can get an not-so-bad result with 10-20 epochs of fine-tuning.

## Model Evaluation

To evaluate the model, you should modify the `isTrain` and `isPredict` variable in `Run.py`,
then run the `Run.py`. Don't forget to check or modify the save path of the prediction results.

If you want to use finetune, you can uncomment the line `do_predict_with_finetune` in `Run.py`, and modify the arguments to set epochs in finetuning before running.

Then you could evaluate the model by submitting the results to the competition website.

Note that you may not get an expected high score as we have in the competition, since we have used some extra data and techniques based on heuristics to improve the score, which are not presented in this repository because they are not the core part of the model, just some tricks used in practice.
