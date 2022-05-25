import os
import argparse
import logging
import requests
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# os.environ["AWS_ACCESS_KEY_ID"] = "ASIAVBCDA6PS7MQ5XEAZ"
# os.environ["AWS_SECRET_ACCESS_KEY"] = "V/CL2zm83pPhhNzeQhcArc8DquadU7OzkgyXRhkm"
# os.environ["AWS_SESSION_TOKEN"] = \
#     "IQoJb3JpZ2luX2VjEDcaCXVzLXdlc3QtMiJHMEUCIQCobylIfgO+AelEWm67y1Jv2yMzlMpG+I18TtB5/AFauAIgN2wUkwNdxvT4PpRY976p6iT/0e1EF3/y027xBjdlpjMqmgMIIBADGgwzNDU4ODU0Mzg5NDkiDG+3GLed9B/q/Pxthir3AnlcMH4Wy7i+aFEmsDw5EQceTYNt3Kw222vDVw7Lj+tEutj1Dx1F2693TygisF9u+KXr5xLZNyzHL1zXfGrrhNOCnS1xXiCxlhBz1ORlgs6mhhfKLbK3YQMtWjRVDYkchTps/dXgDtx50Fec1HxyOaP9CAFjysVoGNsm4msxiAibe0kKaqVajJn4agbKU0hercKXXmCA/6NDix8LrDhUy4kB61++1auUd9ipI0Fwe1+K7vuqZVqleQzt7yiiT/CeCNUq/cdAAiEeBN6svkTwoDvoLMl+Xya/tRDQYzjAgJ1iPHXQ0cn3HkyQpjd7o9BjzbMrWYFBXACB0W+Fmj3zgtgsiUNP3R9DWEXMClE9ishJy1zNm4+l1QsqGvkb5/AUxXxF2GJGykmLSyhGt7LyvA44LZ7O3UAzIxpOQp8RMwieXim1KsvC5OG/nT5VKpH50Oym/Qp0WOpscxwYLqLZCuMT+ojd9nk4tt3GA53UPAy+f+Zi1W6isTCPu6CUBjqmAQEqrzmAmzOrfOPzktWoUZmwjBZqAyLIpd7qf0VzvpgdH3yhyau4w7o4yROLYZ13GD6I4saIRqGMw+2FoQsnBdOKkyapHKiiEl7DUAU00xIjdacxVamwIXT0MFsgaw8Cqvve/fV8gnEVjBEMj/7WocosLaK1nsRJxGlu1XapU3VXVWPlCeGuZsP80aVE+tv0BgtWi+zn4MvVMO0R473sEEqqDxBlSuk="
#
# MLFLOW_TRACKING_URL = "https://rpp-model-registry.internal-dev-k8s.rappipay.com/rappipay-ai/rpp-model-registry/"
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
# mlflow.set_experiment("demo_classification-2")


def go(args):
    # Download file
    logger.info(f"Downloading {args.file_url} ...")
    # with open(basename, 'wb+') as fp:
    with open(args.artifact_name, 'wb+') as fp:
        try:
            # Download the file streaming and write to open temp file
            with requests.get(args.file_url, stream=True) as r:
                for chunk in r.iter_content(chunk_size=8192):
                    fp.write(chunk)

            fp.flush()

            logger.info("Logging artifact")
            mlflow.log_artifact(args.artifact_name)
            os.remove(args.artifact_name)
        except Exception as e:
            logger.error("-- Failed --")
            os.remove(args.artifact_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a file and upload it as an artifact to MLFlow"
    )

    parser.add_argument(
        "--file_url", type=str, help="URL to the input file", required=True
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    with mlflow.start_run() as run:
        go(args)
        mlflow.set_tag("artifact_type", "download")
        mlflow.set_tag("current", "1")
