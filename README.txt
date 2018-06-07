Files must be run in a particular order to get this to work.

1. Run scrape_htmls.py to get all the urls to download images from.
2. Run scrape_jpgs_from_htmls.py to get all the actual images.
3. Run make_dataset_info.py which converts all images to .npy files, resizes them, and generates metadata about them.
4. Run create_train_test_val.py which splits the dataset into those three categories.
5. Run create_tfrecord_files.py to create the actual .tfrecords files for training.
6. Run create_tfrecord_metadata.py to get the metadata for the .tfrecords files.
7. Run random_grid_hyperparam_search.py to test a bunch of hyperparams for the model.
8. Run train_model.py to actually train the model on the best hyperparams.
9. Run test_model.py to test the model on the reserved test set.
10. Run baseline.py to compare the model to a naive baseline.
