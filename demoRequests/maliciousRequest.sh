#!/bin/bash

curl "http://127.0.0.1:8000/api/data?id=SELECT%20*%20FROM%20users%20WHERE%20admin%20=%20'true'" -w "\n"
