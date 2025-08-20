#!/usr/bin/env python3
import os
import zipfile
import requests
import json
from pathlib import Path

def main():
    # Read team name from team.json
    with open('team.json', 'r') as f:
        team_data = json.load(f)
        team_name = team_data['name']
        
    zip_filename = f"{team_name}_code_submission.zip"

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            if zip_filename in files:
                files.remove(zip_filename)
            
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, '.')
                zipf.write(file_path, arcname)

    s3_url = f"https://arhackathon.s3.us-east-1.amazonaws.com/{zip_filename}"

    try:
        with open(zip_filename, 'rb') as f:
            response = requests.put(s3_url, data=f)
        
        if response.status_code == 200:
            print(f"Successfully uploaded {zip_filename}")
        else:
            print(f"Upload failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Upload failed: {e}")

    os.remove(zip_filename)

if __name__ == "__main__":
    main()
