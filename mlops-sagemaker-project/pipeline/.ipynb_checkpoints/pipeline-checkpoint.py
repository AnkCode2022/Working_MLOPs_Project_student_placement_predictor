import boto3
import sagemaker

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.step_collections import RegisterModel

from sagemaker.processing import ScriptProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.estimator import XGBoost


# AWS setup
region = boto3.Session().region_name
session = sagemaker.session.Session()
role = sagemaker.get_execution_role()

bucket = "placement-project-bkt"

input_data = "s3://placement-project-bkt/rawdata/students_placement.csv"


# -------------------------
# Processing
# -------------------------

processor = ScriptProcessor(

    image_uri=sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1"
    ),

    command=["python3"],
    instance_type="ml.m5.large",
    instance_count=1,
    role=role
)


processing_step = ProcessingStep(

    name="PreprocessData",

    processor=processor,

    inputs=[
        ProcessingInput(
            source=input_data,
            destination="/opt/ml/processing/input"
        )
    ],

    outputs=[

        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/train",
            destination=f"s3://{bucket}/train"
        ),

        ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/test",
            destination=f"s3://{bucket}/test"
        )
    ],

    code="src/preprocess.py",

    job_arguments=[
        "--input-data", "/opt/ml/processing/input/students_placement.csv",
        "--train-output", "/opt/ml/processing/train/train.csv",
        "--test-output", "/opt/ml/processing/test/test.csv"
    ]
)


# -------------------------
# Training
# -------------------------

estimator = XGBoost(

    entry_point="train.py",
    source_dir="src",
    framework_version="1.5-1",

    instance_type="ml.m5.large",
    instance_count=1,

    role=role,
    output_path=f"s3://{bucket}/model"
)


training_step = TrainingStep(

    name="TrainModel",

    estimator=estimator,

    inputs={
        "train": TrainingInput(
            s3_data=f"s3://{bucket}/train"
        )
    }
)

training_step.add_depends_on([processing_step])


# -------------------------
# Evaluation
# -------------------------

evaluation_processor = ScriptProcessor(

    image_uri=sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.5-1"
    ),

    command=["python3"],
    instance_type="ml.m5.large",
    instance_count=1,
    role=role
)


evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)


evaluation_step = ProcessingStep(

    name="EvaluateModel",

    processor=evaluation_processor,

    inputs=[

        ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),

        ProcessingInput(
            source=processing_step.properties.ProcessingOutputConfig
            .Outputs["test"]
            .S3Output
            .S3Uri,
            destination="/opt/ml/processing/test"
        )
    ],

    outputs=[

        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
            destination=f"s3://{bucket}/evaluation"
        )
    ],

    code="src/evaluate.py",

    property_files=[evaluation_report]
)

evaluation_step.add_depends_on([training_step])


# -------------------------
# Model Registry
# -------------------------

register_model_step = RegisterModel(

    name="RegisterPlacementModel",

    estimator=estimator,

    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,

    content_types=["text/csv"],
    response_types=["text/csv"],

    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],

    model_package_group_name="placement-model",

    approval_status="PendingManualApproval"
)


# -------------------------
# Condition Step
# -------------------------

condition_step = ConditionStep(

    name="CheckAccuracy",

    conditions=[

        ConditionGreaterThanOrEqualTo(

            left=JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_report,
                json_path="classification_metrics.accuracy.value"
            ),

            right=0.80
        )
    ],

    if_steps=[register_model_step],

    else_steps=[]
)


# -------------------------
# Pipeline
# -------------------------

pipeline = Pipeline(

    name="PlacementPipeline",

    steps=[

        processing_step,
        training_step,
        evaluation_step,
        condition_step
    ]
)


pipeline.upsert(role_arn=role)

pipeline.start()

print("Pipeline started")