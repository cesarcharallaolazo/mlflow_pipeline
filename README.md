# genre_classification

Steps MLops

1. downloading

    mlflow run . -P file_url="https://github.com/cesarcharallaolazo/mlflow_pipeline/blob/master/_data/genres_mod.parquet?raw=true" -P artifact_name=raw_data.parquet -P artifact_description="This is a test data"

2. preprocessing

    mlflow run . -P input_artifact=raw_data.parquet:latest -P artifact_name=preprocessed_data.csv -P artifact_description="This is a test data"
    
3. check/tests

    mlflow run . -P reference_artifact=preprocessed_data.csv:latest -P sample_artifact=preprocessed_data.csv:latest -P ks_alpha=0.05
    
4. segregation

    mlflow run . -P input_artifact=preprocessed_data.csv:latest -P artifact_root=data -P test_size=0.3 -P stratify=genre
    
5. modeling

    mlflow run . -P train_data=data_train.csv:latest -P model_config=rf_config.yaml -P export_artifact=model_export -P random_seed=42 -P val_size=0.3 -P stratify=genre
    
6. evaluate

     mlflow run . -P model_export=model_export:latest -P test_data=data_test.csv:latest
     
7. run hyperparameter tunning

    mlflow run . -P hydra_options="-m random_forest_pipeline.random_forest.n_estimators=10,50,80"
    
8. run hyperparameter tunning - multitask

    mlflow run . -P hydra_options="hydra/launcher=joblib random_forest_pipeline.random_forest.n_estimators=range(20,40,10) random_forest_pipeline.random_forest.max_depth=range(7,17,5) --m"
    
9. run mlflow pipeline with gihub

    mlflow run https://github.com/cesarcharallaolazo/mlflow_pipeline.git -P hydra_options="main.project_name=remote_execution"