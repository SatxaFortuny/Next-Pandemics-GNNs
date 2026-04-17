@echo off
set CONTAINER_NAME=db_preproc
set BACKUP_FILE=backup_preproc.sql

echo [+] Cleaning database...
docker exec %CONTAINER_NAME% dropdb -U postgres --if-exists db_preproc
docker exec %CONTAINER_NAME% createdb -U postgres db_preproc

echo [+] Loading backup...
type %BACKUP_FILE% | docker exec -i %CONTAINER_NAME% psql -U postgres -d db_preproc

echo [V] Backup restored.
pause