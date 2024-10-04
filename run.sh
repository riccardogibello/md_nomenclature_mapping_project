#!/bin/bash

# Check if the required arguments are provided, which consist
# of the PostgreSQL username and password
if [ -z "$1" ]; then
    echo "Error: PGUSER is not provided."
    exit 1
fi
if [ -z "$2" ]; then
    echo "Error: PGPASSWORD is not provided."
    exit 1
fi

# Set the PostgreSQL environment variables from the arguments
export PGUSER=$1
export PGPASSWORD=$2
export PGHOST=localhost
export PGPORT=5432

# Activate the virtual environment
source .venv/bin/activate

# Execute the SQL commands to create a test user for the code execution
# and to create a database for the project
psql -c "CREATE USER test WITH PASSWORD 'test';"
psql -c "CREATE DATABASE nomenclature_mapping_project;"
psql -c "GRANT ALL PRIVILEGES ON DATABASE nomenclature_mapping_project TO test;"

# Run the main.py file
python main.py