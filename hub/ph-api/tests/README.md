# API Tests

This directory contains tests for the API endpoints and services.

## Structure

- `__init__.py`: Makes the tests directory a Python package
- `conftest.py`: Contains pytest fixtures used across test files
- `test_factory.py`: Provides utility functions for creating test clients
- `test_users.py`: Tests for the users service endpoints

## Running Tests

To run all tests:

```bash
cd api
pytest
```

To run a specific test file:

```bash
cd api
pytest tests/test_users.py
```

To run a specific test function:

```bash
cd api
pytest tests/test_users.py::test_function_name
```

## Test Coverage

To run tests with coverage:

```bash
cd api
pytest --cov=. tests/
```

## Adding New Tests

When adding new tests:

1. Create a new test file named `test_<module_name>.py`
2. Import necessary modules and fixtures
3. Use the `create_test_client()` function from `test_factory.py` to create a test client
4. Write test functions that start with `test_`
5. Use appropriate assertions to verify expected behavior

## Mocking External Services

For tests that require external services (e.g., Twilio, Cloudflare), use the test phone numbers that bypass actual API calls:

- `+11234567890`
- `+15551234567`
- `+15625555555`

These numbers are configured in the `ignore_validation` function in the users service.
