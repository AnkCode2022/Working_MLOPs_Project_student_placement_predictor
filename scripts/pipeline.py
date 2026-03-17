import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor, ProcessingStep
from sagemaker.workflow.pipeline import Pipeline

def get_sagemaker_pipeline(role, pipeline_name="MyModularPipeline"):
    """
    Constructs the 5-step SageMaker Pipeline.
    """
    pipeline_session = PipelineSession()
    region = pipeline_session.boto_region_name

    # Image for the container
    image_uri = sagemaker.image_uris.retrieve(framework="sklearn", region=region, version="1.2-1")

    # The shared processor definition
    processor = ScriptProcessor(
        image_uri=image_uri,
        command=['python3'],
        instance_type='ml.m5.large',
        instance_count=1,
        base_job_name="mlops-step",
        sagemaker_session=pipeline_session,
        role=role
    )

    # Define steps
    step_1 = ProcessingStep(name="Ingestion", processor=processor, code="scripts/dataingestion.py")
    step_2 = ProcessingStep(name="Preprocessing", processor=processor, code="scripts/datapreprocessing.py", depends_on=[step_1])
    step_3 = ProcessingStep(name="Training", processor=processor, code="scripts/training.py", depends_on=[step_2])
    step_4 = ProcessingStep(name="Evaluation", processor=processor, code="scripts/evaluation.py", depends_on=[step_3])
    step_5 = ProcessingStep(name="Testing", processor=processor, code="scripts/testing.py", depends_on=[step_4])

    # Create the pipeline object
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_1, step_2, step_3, step_4, step_5],
        sagemaker_session=pipeline_session
    )
    
    return pipeline