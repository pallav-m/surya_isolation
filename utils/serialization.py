from typing import Any, List, Dict
import json
from pydantic import BaseModel

def pydantic_to_dict(obj: Any) -> Any:
    """
    Recursively convert Pydantic models and nested structures to dictionaries.
    
    Args:
        obj: Any object (Pydantic model, list, dict, or primitive)
        
    Returns:
        Dictionary representation or original value
    """
    if isinstance(obj, BaseModel):
        # Pydantic model - use model_dump() (v2) or dict() (v1)
        try:
            return obj.model_dump()  # Pydantic v2
        except AttributeError:
            return obj.dict()  # Pydantic v1
    
    elif isinstance(obj, list):
        return [pydantic_to_dict(item) for item in obj]
    
    elif isinstance(obj, dict):
        return {key: pydantic_to_dict(value) for key, value in obj.items()}
    
    else:
        return obj


def serialize_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize all results to JSON-compatible format.
    
    Args:
        results: Dictionary containing Pydantic models
        
    Returns:
        JSON-serializable dictionary
    """
    return pydantic_to_dict(results)


def save_results_json(results: Dict[str, Any], output_path: str):
    """
    Save results to JSON file with proper serialization.
    
    Args:
        results: Dictionary containing Pydantic models
        output_path: Path to save JSON file
    """
    serialized = serialize_results(results)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)
