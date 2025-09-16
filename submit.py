#!/usr/bin/env python3
import os
import requests
import json
import re

def sanitize_filename(name):
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    if sanitized and not sanitized[0].isalnum():
        sanitized = 'team_' + sanitized
    return sanitized

def main():
    with open('team.json', 'r') as f:
        team_data = json.load(f)
        team_name = sanitize_filename(team_data['name'])
    
    routing_file = "ar_hackathon/api/routing.py"
    renamed_file = f"{team_name}_routing.py"
    
    if not os.path.exists(routing_file):
        print(f"Error: {routing_file} not found")
        return

    s3_url = f"https://arhackathon.s3.us-east-1.amazonaws.com/{renamed_file}"

    try:
        with open(routing_file, 'rb') as f:
            response = requests.put(s3_url, data=f)
        
        if response.status_code == 200:
            print(f"Successfully uploaded {renamed_file}")
        else:
            print(f"Upload failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
