#!/bin/bash
echo "Loading chembl_36.dump..."
pg_restore --username=chembl --dbname=chembl --no-owner /docker-entrypoint-initdb.d/chembl_36.dump
