@echo off

REM Check if the required arguments are provided, which consist
REM of the PostgreSQL username and password
if "%1"=="" (
    echo Error: PGUSER is not provided.
    exit /b 1
)
if "%2"=="" (
    echo Error: PGPASSWORD is not provided.
    exit /b 1
)

REM Set the PostgreSQL environment variables from the arguments
set PGUSER=%1
set PGPASSWORD=%2
set PGHOST=localhost
set PGPORT=5432

REM Activate the virtual environment
call .venv\Scripts\activate

REM Execute the SQL commands to create a test user for the code execution
REM and to create a database for the project
psql -c "CREATE USER test WITH PASSWORD 'test';"
psql -c "CREATE DATABASE nomenclature_mapping_project;"
psql -c "GRANT ALL PRIVILEGES ON DATABASE nomenclature_mapping_project TO test;"

REM Run the main.py file
python main.py