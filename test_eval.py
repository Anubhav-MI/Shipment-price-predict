from shipment.components.model_evaluation import ModelEvaluation

evaluator = ModelEvaluation(
    model_path="shipment/artifacts/model.pkl",   # path to your model
    test_data_path="test_data.csv"               # your test dataset
)

evaluator.evaluate()
