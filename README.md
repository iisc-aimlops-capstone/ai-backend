# AIMLOPS_Capstone

* To test the script, move to `src/plant_disease_detection/` and run `python infer.py`
* Models are stored in `artifacts/model/`
* The prediction images to be stored in `pred/`
* Add Data preprocessing and validation steps in `src/plant_disease_detection/data_preprocessing.py` and `src/plant_disease_detection/data_validation.py`]
* To add the preprocessing and validation, go to `src/plant_disease_detection/infer.py` and add import the modules and add the function calls inside the **try** block.
```
    images = get_images_from_path()

    # Initialize model
    try:
        classifier = PlantDiseaseClassifier(os.path.join(root, configs['MODEL_CKPT']))
        logger.info(f"Loaded model: {os.path.join(root, configs['MODEL_CKPT'])}")
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        return
```
