#!/bin/bash

# Run the migration script to create the sample_datasets table
echo "Running migration to create sample_datasets table..."
python migrations/create_sample_datasets_table.py

echo "Migration completed successfully!"
