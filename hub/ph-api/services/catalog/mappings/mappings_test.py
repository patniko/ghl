import json
import pytest
from unittest.mock import patch, AsyncMock

from models import Check
from services.catalog.mappings.llm_mapping import (
    analyze_dataset_columns_with_llm,
    analyze_uploaded_dataset,
)


@pytest.mark.asyncio
@patch("services.catalog.mappings.llm_mapping.get_claude_client")
async def test_analyze_dataset_columns_with_llm(mock_get_claude_client):
    """Test analyzing dataset columns with LLM"""
    # Mock the Claude client
    mock_client = AsyncMock()
    mock_client.send_message = AsyncMock(
        return_value={
            "content": json.dumps(
                {
                    "column_mappings": {
                        "age": {
                            "inferred_data_type": "numeric",
                            "recommended_checks": [1, 2],
                            "reasoning": "Age is a numeric value",
                        },
                        "sex": {
                            "inferred_data_type": "categorical",
                            "recommended_checks": [3],
                            "reasoning": "Sex is a categorical value",
                        },
                    }
                }
            )
        }
    )
    mock_get_claude_client.return_value = mock_client

    # Test data
    headers = ["age", "sex"]
    sample_rows = [
        {"age": 25, "sex": "M"},
        {"age": 30, "sex": "F"},
    ]
    available_checks = [
        Check(
            id=1,
            name="Range Check",
            data_type="numeric",
            implementation="check_range_compliance",
        ),
        Check(
            id=2,
            name="Numeric Stats",
            data_type="numeric",
            implementation="check_numeric_statistics",
        ),
        Check(
            id=3,
            name="Categorical Dist",
            data_type="categorical",
            implementation="check_categorical_distribution",
        ),
    ]

    # Call the function
    result = await analyze_dataset_columns_with_llm(
        headers, sample_rows, available_checks
    )

    # Verify the result
    assert "column_mappings" in result
    assert "check_mappings" in result
    assert "age" in result["column_mappings"]
    assert "sex" in result["column_mappings"]
    assert result["check_mappings"]["age"] == [1, 2]
    assert result["check_mappings"]["sex"] == [3]

    # Verify the LLM was called with the correct prompt
    mock_client.send_message.assert_called_once()
    call_args = mock_client.send_message.call_args[1]
    assert "prompt" in call_args
    assert "system" in call_args
    assert "temperature" in call_args
    assert "age" in call_args["prompt"]
    assert "sex" in call_args["prompt"]


@pytest.mark.asyncio
@patch("services.catalog.mappings.llm_mapping.analyze_dataset_columns_with_llm")
async def test_analyze_uploaded_dataset(mock_analyze_dataset_columns_with_llm):
    """Test analyzing an uploaded dataset"""
    # Mock the analyze_dataset_columns_with_llm function
    mock_analyze_dataset_columns_with_llm.return_value = {
        "column_mappings": {
            "age": {
                "inferred_data_type": "numeric",
                "recommended_checks": [1, 2],
                "reasoning": "Age is a numeric value",
            },
            "sex": {
                "inferred_data_type": "categorical",
                "recommended_checks": [3],
                "reasoning": "Sex is a categorical value",
            },
        },
        "check_mappings": {
            "age": [1, 2],
            "sex": [3],
        },
    }

    # Test data
    data = [
        {"age": 25, "sex": "M"},
        {"age": 30, "sex": "F"},
        {"age": 35, "sex": "M"},
    ]
    available_checks = [
        Check(
            id=1,
            name="Range Check",
            data_type="numeric",
            implementation="check_range_compliance",
        ),
        Check(
            id=2,
            name="Numeric Stats",
            data_type="numeric",
            implementation="check_numeric_statistics",
        ),
        Check(
            id=3,
            name="Categorical Dist",
            data_type="categorical",
            implementation="check_categorical_distribution",
        ),
    ]

    # Call the function
    result = await analyze_uploaded_dataset(data, available_checks)

    # Verify the result
    assert "total_records" in result
    assert result["total_records"] == 3
    assert "column_mappings" in result
    assert "check_mappings" in result
    assert "age" in result["column_mappings"]
    assert "sex" in result["column_mappings"]
    assert result["check_mappings"]["age"] == [1, 2]
    assert result["check_mappings"]["sex"] == [3]

    # Verify analyze_dataset_columns_with_llm was called with the correct arguments
    mock_analyze_dataset_columns_with_llm.assert_called_once()
    call_args = mock_analyze_dataset_columns_with_llm.call_args[1]
    assert "headers" in call_args
    assert "sample_rows" in call_args
    assert "available_checks" in call_args
    assert call_args["headers"] == ["age", "sex"]
    assert (
        len(call_args["sample_rows"]) == 3
    )  # Should use all rows since we have fewer than 5
    assert call_args["available_checks"] == available_checks


@pytest.mark.asyncio
async def test_analyze_uploaded_dataset_empty_data():
    """Test analyzing an uploaded dataset with empty data"""
    # Test with empty data
    result = await analyze_uploaded_dataset([], [])

    # Verify the result
    assert "message" in result
    assert result["message"] == "No data available for analysis"
    assert "column_mappings" in result
    assert "check_mappings" in result
    assert result["column_mappings"] == {}
    assert result["check_mappings"] == {}
