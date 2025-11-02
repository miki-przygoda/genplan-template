f''' - cannonical format for the layout constraints
Read backend/processed/floor_plans/floor001/metadata.json
iterate through each floor's metadata.json
for each floor generate a layout contrainst file - json file
It should contain the following:
json format: {
"schema_version": "1.0.0", - should be based off of the schema version from the metadata.json
"supporting_text": "The first bedroom is located at south west. The second bedroom is located at north east. The bathroom is located at south of the first bedroom. The living dining is located at north.",
"rooms": [
    {"name": "bedroom_1", "zone": "SW"},
    {"name": "bedroom_2", "zone": "SE"},
    {"name": "bathroom_1", "rel": "south_of", "target": "bedroom_1"},
    {"name": "living_dining", "zone": "N"}
],
    "image_size": [512, 512],
    "constraints": {
    "min_room_area_ratio": 0.01,
    "allow_overlap": false
}
}

- output file should be saved in backend/processed/layout_constraints/floor{id}.json
if the id is inside of backend/processed/no_text_ids.json, then the json file should be empty

'''

