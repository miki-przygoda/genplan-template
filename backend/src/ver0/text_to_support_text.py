"""
Script used to convert supporting text from metadata.json to a list of rooms and their relationships

Input: supporting text from metadata.json
Output: list of rooms and their relationships

Example:
Input: "The first bedroom is located at south west. The second bedroom is located at north east. The bathroom is located at south of the first bedroom. The living dining is located at north."
Output: [
    {"name": "bedroom_1", "zone": "SW"},
    {"name": "bedroom_2", "zone": "NE"},
    {"name": "bathroom_1", "zone": "SE"},
    {"name": "living_dining", "zone": "N"}
]

Could use:
Spacy, NLTK, or other NLP libraries to parse the text and extract the rooms and their relationships

Could do:
use a rule /regex parser to get the rooms and their relationships
could have each room as an object with id, name, section and relationships
that way you can have the same type of room but with different ids and sections
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal



Section = Literal["NE", "NW", "SE", "SW"]
RoomToRoomRelationship = Literal["north_of", "south_of", "east_of", "west_of"]
RoomType = Literal["bedroom", "bathroom", "living", "dining", "storeroom", "kitchen", "office", "garage", "other"]

@dataclass
class Relationship:
    target: int
    relationship_type: RoomToRoomRelationship    

@dataclass
class Room:
    id: int
    room_type: RoomType
    section: Section
    relationships: list[Relationship] = field(default_factory=list)

    def add_relationship(self, relationship: Relationship):
        self.relationships.append(relationship)

def link_rooms(Room1: Room, Room2: Room):
    Room1.add_relationship(Relationship(target=Room2.id, relationship_type="south_of"))
    Room2.add_relationship(Relationship(target=Room1.id, relationship_type="north_of"))