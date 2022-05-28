# genre_classification

Step MLOps - MLFlow

1. downloading

    mlflow run ./download -P step=download_data -P file_url="https://github.com/cesarcharallaolazo/mlflow_pipeline/blob/master/_data/genres_mod.parquet?raw=true" -P artifact_name=raw_data.parquet -P artifact_description="Pipeline for data downloading" --experiment-name genre_classification --run-name download_data
    
2. preprocessing

    mlflow run ./preprocess -P step=preprocess -P input_step=download_data -P input_artifact=raw_data.parquet -P artifact_name=preprocessed_data.csv -P artifact_description="Pipeline for data preprocessing" --experiment-name genre_classification --run-name preprocess
 
3. check/tests

    mlflow run ./check_data -P step=check_data -P input_step=preprocess -P reference_artifact=preprocessed_data.csv -P sample_artifact=preprocessed_data.csv -P ks_alpha=0.05 --experiment-name genre_classification --run-name check_data
    
4. segregation

    mlflow run ./segregate -P step=segregate -P input_step=preprocess -P input_artifact=preprocessed_data.csv -P artifact_root=data -P test_size=0.3 -P stratify=genre --experiment-name genre_classification --run-name segregate
    
5. modeling

    mlflow run ./random_forest -P step=random_forest -P input_step=segregate -P train_data=data/data_train.csv -P model_config=rf_config.yaml -P export_artifact=model_export -P random_seed=42 -P val_size=0.3 -P stratify=genre --experiment-name genre_classification --run-name random_forest
    
6. evaluate

     mlflow run ./evaluate -P step=evaluate -P input_model_step=random_forest -P model_export=model_export -P input_data_step=segregate -P test_data=data/data_test.csv --experiment-name genre_classification --run-name evaluate
     
7. run hyperparameter tunning

    mlflow run . -P hydra_options="-m random_forest_pipeline.random_forest.n_estimators=10,50,80"
    mlflow run . -P hydra_options="-m random_forest_pipeline.random_forest.n_estimators=15,55,85 random_forest_pipeline.random_forest.max_depth=range(7,17,5)"
    
9. run mlflow pipeline with gihub

    mlflow run https://github.com/cesarcharallaolazo/mlflow_pipeline.git -P hydra_options="main.project_name=remote_execution"