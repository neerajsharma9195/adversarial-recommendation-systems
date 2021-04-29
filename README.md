# adversarial-recommendation-systems

To run the CF step:

vanilla: `python setup.py build_ext --inplace; python -m src.models.surprise_cf`
with augmentation: `python setup.py build_ext --inplace; python -m src.models.surprise_cf --use_augmentation yes --augmented_users_file_path generated_users_path --augmented_items_file_path generated_items_path`
