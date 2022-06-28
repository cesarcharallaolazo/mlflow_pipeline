[model_serving]:_md_img/serve.png

# genre_classification

## Step MLOps - MLFlow

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

7. scheduled predictions

        mlflow run ./prediction
        
8. scheduled eventual retraining

        mlflow run ./retraining

9. run all ML Pipeline (workflow)     

        mlflow run .
        mlflow run . -P hydra_options="main.experiment_name=prod_all_genre_classification"

10. run hyperparameter tunning

        mlflow run . -P hydra_options="-m random_forest_pipeline.random_forest.n_estimators=10,50,80"
        mlflow run . -P hydra_options="-m main.experiment_name=prod_all_genre_classification random_forest_pipeline.random_forest.n_estimators=12,52,82"
        mlflow run . -P hydra_options="-m main.experiment_name=airflow_prod_all_genre_classification random_forest_pipeline.random_forest.n_estimators=120"
        mlflow run . -P hydra_options="-m random_forest_pipeline.random_forest.n_estimators=15,55,85 random_forest_pipeline.random_forest.max_depth=range(7,17,5)"
    
11. run mlflow pipeline from github

        mlflow run https://github.com/cesarcharallaolazo/mlflow_pipeline.git -v 4f979eea1c60ffabff0bbad2a077f6d114684a99 -P hydra_options="main.experiment_name=remote_all_genre_classification"
        mlflow run https://github.com/cesarcharallaolazo/mlflow_pipeline.git -v 4f979eea1c60ffabff0bbad2a077f6d114684a99 -P hydra_options="-m main.mlflow_tracking_url=http://localhost:7755/ main.experiment_name=remote_all_genre_classification random_forest_pipeline.random_forest.n_estimators=14,54"

## Mlflow Deployment

### Batch

a. Download the mlflow model (./serve path)

        mlflow artifacts download -u {artifact_uri(../artifacts/model_export)} -d {destination_path(.)}
        mlflow artifacts download -u {artifact_uri(../artifacts/data/data_test.csv)} -d {destination_path(.)}
        
b. Test the mlflow model

        mlflow models predict -t json -i model_export/input_example.json -m model_export
        mlflow models predict -t csv -i data_test.csv -m model_export
        
### Online

        mlflow models serve -m model_export
        mlflow models serve -m model_export & (background)
        
![alt][model_serving]

#### Extra Notes to spin-up a mlflow docker container
- create docker network: docker network create cesar_net
- run a postgress container: docker run --network cesar_net --expose=5432 -p 5432:5432 -d -v $PWD/pg_data_1/:/var/lib/postgresql/data/ --name pg_mlflow -e POSTGRES_USER='user_pg' -e POSTGRES_PASSWORD='pass_pg' postgres
- build Dockerfile: docker build -t mlflow_cesar .
- modify env var default_artifact_root in local.env with your current directory (also in docker-compose on Dag-batch-drift)
- run mlflow server container: docker run -d -p 7755:5000 -v $PWD/container_artifacts:$PWD/container_artifacts --env-file local.env --network cesar_net --name test mlflow_cesar

