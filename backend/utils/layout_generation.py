# TODO:
""" - cannonical format for the layout constraints
Read backend/data/processed/floor_plans/floor001/metadata.json
iterate through each floor's metadata.json
for each floor generate a layout contrainst file - json file
It should contain the following:
{
  "schema_version": 1,
  "source_id": "floor012",
  "image_size": [512, 512],
  "supporting_text": {from metadata.json if it exists: supporting_text from metadata.json if it exists},
  "rooms": [ {from text parser (text_to_support_text.py) if supporting_text exists, else empty list}]: [
   e.g. {
      "name": "bedroom_1",
      "zone": "SW",
      "relations": []
    },
    {
      "name": "bathroom_1",
      "zone": "SE",
      "relations": [
        {"type": "south_of", "target": "bedroom_1"}
      ]
    }
  ],
  "constraints": {
    "allow_overlap": false,
    "min_room_area_ratio": 0.01,
    "grid": [4, 4]
  }
}


- output file should be saved in backend/data/processed/layout_constraints/floor{id}.json
if the id is inside of backend/data/processed/no_text_ids.json, then the json file should be empty

- extracts what it can (image size, room count, supporting_text) from the metadata.json
if there is supporting text, calls the text parser to extract the rooms and their relationships
"""

