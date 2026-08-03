from pydantic import BaseModel, Field
from pathlib import Path


class ModelParams(BaseModel):
    n_estimators: int = Field(gt=0, description="Number of trees in the forest")
    max_depth: int = Field(gt=0, description="Maximum depth of the tree")
    random_state: int = Field(description="Random seed for reproducibility")


class Config(BaseModel):
    data_path: Path
    target_column: str
    features: list[str]
    model_params: ModelParams
    output_path: Path
