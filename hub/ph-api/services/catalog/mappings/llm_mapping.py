import json
from typing import List, Dict, Any
from fastapi import HTTPException
from loguru import logger

from llm import get_claude_client
from models import Check


async def analyze_dataset_columns_with_llm(
    headers: List[str], sample_rows: List[Dict[str, Any]], available_checks: List[Check]
) -> Dict[str, List[int]]:
    """
    Use LLM to analyze dataset columns and suggest appropriate checks based on column headers and sample data.

    Args:
        headers: List of column headers
        sample_rows: List of dictionaries containing the first 5 rows of data
        available_checks: List of available checks in the system

    Returns:
        Dictionary mapping column names to lists of check IDs
    """
    # Prepare the prompt for the LLM
    checks_info = []
    for check in available_checks:
        check_info = {
            "id": check.id,
            "name": check.name,
            "description": check.description,
            "data_type": check.data_type,
            "implementation": check.implementation,
        }
        checks_info.append(check_info)

    # Format sample data for better readability
    formatted_samples = []
    for row in sample_rows:
        formatted_row = {}
        for header in headers:
            formatted_row[header] = row.get(header, "")
        formatted_samples.append(formatted_row)

    # Create the prompt
    prompt = f"""
You are a data quality expert. I need your help mapping columns in a dataset to appropriate data quality checks.

Here are the available data quality checks:
{json.dumps(checks_info, indent=2)}

Here are the column headers from the dataset:
{json.dumps(headers, indent=2)}

Here are the first {len(sample_rows)} rows of data:
{json.dumps(formatted_samples, indent=2)}

For each column in the dataset, please:
1. Analyze the column name and sample data
2. Determine the most appropriate data type (numeric, categorical, date, text, or boolean)
3. Recommend which data quality checks would be most appropriate for this column

IMPORTANT: Your response MUST be valid JSON. Do not include any explanatory text outside the JSON structure.
Ensure all property names are in double quotes, all strings are in double quotes, and there are no trailing commas.

Please provide your response in the following JSON format:
{{
  "column_mappings": {{
    "column_name1": {{
      "inferred_data_type": "data_type",
      "recommended_checks": [check_id1, check_id2, ...],
      "reasoning": "Brief explanation of why these checks are appropriate"
    }},
    "column_name2": {{
      "inferred_data_type": "data_type",
      "recommended_checks": [check_id1, check_id2, ...],
      "reasoning": "Brief explanation of why these checks are appropriate"
    }},
    ...
  }}
}}

Only include checks that are appropriate for each column based on its data type and content.
Double-check your JSON for syntax errors before submitting your response.
"""

    logger.info(f"Sending prompt to LLM: {prompt}")

    # Get the LLM client
    claude_client = get_claude_client()

    try:
        # Send the prompt to the LLM
        response = await claude_client.send_message(
            prompt=prompt,
            system="You are a helpful data quality assistant that analyzes datasets and recommends appropriate data quality checks. ALWAYS respond with valid JSON only. Do not include any explanatory text outside the JSON structure. Ensure all property names and strings use double quotes, arrays are properly formatted with square brackets, and there are no trailing commas. Your entire response must be parseable by Python's json.loads() function.",
            temperature=0.1,  # Lower temperature for more deterministic responses
        )

        # Extract the JSON response
        content = response.get("content", "")

        # Find JSON in the response (in case the LLM adds explanatory text)
        import re

        json_match = re.search(r"({[\s\S]*})", content)
        if json_match:
            json_str = json_match.group(1)
            try:
                # First attempt: Try to parse the JSON as-is
                result = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parsing failed: {str(e)}")

                # Second attempt: Try to fix common JSON syntax errors
                try:
                    # Fix missing commas between objects in arrays
                    fixed_json = re.sub(r"}\s*{", "},{", json_str)

                    # Fix trailing commas in arrays and objects
                    fixed_json = re.sub(r",\s*}", "}", fixed_json)
                    fixed_json = re.sub(r",\s*]", "]", fixed_json)

                    # Fix missing quotes around property names
                    fixed_json = re.sub(r"([{,])\s*(\w+):", r'\1"\2":', fixed_json)

                    # Try parsing the fixed JSON
                    result = json.loads(fixed_json)
                    logger.info("Successfully fixed and parsed JSON")
                except json.JSONDecodeError as repair_error:
                    logger.error(f"Failed to repair JSON: {str(repair_error)}")

                    # Third attempt: Fall back to a more aggressive approach - extract valid JSON objects
                    try:
                        # Create a minimal valid response
                        logger.warning("Falling back to minimal valid response")
                        result = {"column_mappings": {}}

                        # Try to extract individual column mappings using regex
                        column_pattern = r'"(\w+)"\s*:\s*{([^{}]|{[^{}]*})*}'
                        column_matches = re.finditer(column_pattern, json_str)

                        for match in column_matches:
                            try:
                                column_json = "{" + match.group(0) + "}"
                                column_data = json.loads(column_json)
                                if column_data:
                                    # Add any valid column data to our result
                                    for key, value in column_data.items():
                                        result["column_mappings"][key] = value
                            except Exception:
                                # Skip any columns that can't be parsed
                                continue
                    except Exception as fallback_error:
                        logger.error(f"Fallback parsing failed: {str(fallback_error)}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to parse LLM response as JSON: {str(e)}. Repair and fallback also failed.",
                        )

            # Extract column mappings
            column_mappings = result.get("column_mappings", {})

            # Convert to the format expected by the apply_checks endpoint
            check_mappings = {}
            for column_name, mapping in column_mappings.items():
                recommended_checks = mapping.get("recommended_checks", [])
                if recommended_checks:
                    check_mappings[column_name] = recommended_checks

            return {
                "column_mappings": column_mappings,
                "check_mappings": check_mappings,
            }
        else:
            raise HTTPException(
                status_code=500, detail="LLM response did not contain valid JSON"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling LLM: {str(e)}")


async def analyze_uploaded_dataset(
    data: List[Dict[str, Any]], available_checks: List[Check]
) -> Dict[str, Any]:
    """
    Analyze an uploaded dataset using LLM to suggest column-to-check mappings.

    Args:
        data: The dataset as a list of dictionaries
        available_checks: List of available checks in the system

    Returns:
        Dictionary with analysis results
    """
    if not data:
        return {
            "message": "No data available for analysis",
            "column_mappings": {},
            "check_mappings": {},
        }

    # Get column headers
    headers = list(data[0].keys())

    # Get sample rows (up to 5)
    sample_rows = data[: min(5, len(data))]

    # Use LLM to analyze columns and suggest checks
    analysis_result = await analyze_dataset_columns_with_llm(
        headers=headers, sample_rows=sample_rows, available_checks=available_checks
    )

    return {
        "total_records": len(data),
        "column_mappings": analysis_result.get("column_mappings", {}),
        "check_mappings": analysis_result.get("check_mappings", {}),
    }
