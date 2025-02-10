1. Train a gaussian in one of the scene from 7 scenes dataset

```
bash zenith_scripts/train_7scenes.sh
```

2. Render a gaussian in one of the scene from 7 scenes dataset

```
bash zenith_scripts/render_7scenes.sh
```

3. Pipeline of localization in (7 scenes/Cambridge landmark)

    1. export MLP dataset

    ```
    python -m mlp.export --method sp
    ```

    2. train MLP

    ```
    python -u -m mlp.train --dim 16 > log.txt 2>&1 &
    ```

    3. use MLP build gaussian feature set(tarin+test)

    ```
    bash zenith_scripts/dataset_build.sh
    ```

    4. train gaussian

    ```
    bash zenith_scripts/train_7scenes.sh
    ```

    5. localization inference(need to change to depth median)

    ```
    bash zenith_scripts/loc_inference.sh
    ```


Pipeline
1. export MLP dataset ->                                  (bash zenith_scripts/export.sh)
2. train MLP(use b5,b6) ->                                (bash zenith_scripts/mlp_train.sh)
3. use MLP build gaussian feature set(tarin+test) ->      (bash zenith_scripts/dataset_build.sh)
4. train gaussian ->                                      (bash zenith_scripts/train.sh)
4. render gaussian ->                                     (bash zenith_scripts/render.sh)
5. localization inference(need to change to depth median) (bash zenith_scripts/loc_inference.sh)
