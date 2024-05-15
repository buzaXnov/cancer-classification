from cnnClassifier import logger
from cnnClassifier.components.evaluation import Evaluation
from cnnClassifier.config.configuration import ConfigurationManager

STAGE_NAME = "Model Evaluation Stage"


class EvaluationPipeline:
    def __init__(self) -> None:
        pass

    def main(self) -> None:
        config = ConfigurationManager()
        eval_config = config.get_eval_config()
        evaluation = Evaluation(eval_config)
        evaluation.get_trained_model()
        evaluation.get_dataloader()
        evaluation.test()
        evaluation.log_into_mlflow()
        evaluation.save_score()        



if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
