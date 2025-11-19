"""
Version 0 natural-language → structured support text converter.

The module offers a light-weight, dependency friendly pipeline that turns the
messy supporting text found in ``metadata.json`` files into a graph of ``Room``
objects with bi-directional relationships. The implementation favours simple
rule based parsing (regular expressions, string aliases) so that we can run in
constrained environments without heavyweight NLP dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable, Literal, NamedTuple, Tuple, Dict, List

from .vars import DEFAULT_GRID_SIZE
from .grid_encoder import cell_to_nested


Section = Literal["N", "S", "E", "W", "NE", "NW", "SE", "SW", "C"]
RoomToRoomRelationship = Literal["north_of", "south_of", "east_of", "west_of"]
RoomType = Literal[
    "bedroom",
    "bathroom",
    "living",
    "dining",
    "storeroom",
    "kitchen",
    "office",
    "garage",
    "other",
]


class RoomDescriptor(NamedTuple):
    room_type: RoomType
    ordinal: int | None


@dataclass
class Relationship:
    target: int
    relationship_type: RoomToRoomRelationship


@dataclass
class Room:
    id: int
    room_type: RoomType
    section: Section = "C"
    ordinal: int | None = None
    relationships: list[Relationship] = field(default_factory=list)
    section_conflicts: list[str] = field(default_factory=list)

    @property
    def canonical_name(self) -> str:
        if self.ordinal is not None:
            return f"{self.room_type}_{self.ordinal}"
        return f"{self.room_type}_{self.id}"

    def add_relationship(self, relationship: Relationship) -> None:
        if relationship not in self.relationships:
            self.relationships.append(relationship)


RELATIONSHIP_INVERSES: dict[RoomToRoomRelationship, RoomToRoomRelationship] = {
    "north_of": "south_of",
    "south_of": "north_of",
    "east_of": "west_of",
    "west_of": "east_of",
}


SECTION_ALIASES: dict[str, Section] = {
    "north": "N",
    "n": "N",
    "south": "S",
    "s": "S",
    "east": "E",
    "e": "E",
    "west": "W",
    "w": "W",
    "north east": "NE",
    "north-east": "NE",
    "northeast": "NE",
    "ne": "NE",
    "north west": "NW",
    "north-west": "NW",
    "northwest": "NW",
    "nw": "NW",
    "south east": "SE",
    "south-east": "SE",
    "southeast": "SE",
    "se": "SE",
    "south west": "SW",
    "south-west": "SW",
    "southwest": "SW",
    "sw": "SW",
    "centre": "C",
    "center": "C",
    "central": "C",
    "middle": "C",
}

ROOM_TYPE_ALIASES: dict[str, RoomType] = {
    "bed": "bedroom",
    "bedroom": "bedroom",
    "primary bedroom": "bedroom",
    "master bedroom": "bedroom",
    "guest bedroom": "bedroom",
    "bath": "bathroom",
    "bathroom": "bathroom",
    "ensuite": "bathroom",
    "en-suite": "bathroom",
    "vc": "bathroom",
    "washroom": "bathroom",
    "toilet": "bathroom",
    "living room": "living",
    "living": "living",
    "lounge": "living",
    "family room": "living",
    "dining room": "dining",
    "dining": "dining",
    "living dining": "living",
    "dining and lounge": "living",
    "dining & lounge": "living",
    "kitchen": "kitchen",
    "pantry": "kitchen",
    "store": "storeroom",
    "storeroom": "storeroom",
    "storage": "storeroom",
    "office": "office",
    "study": "office",
    "garage": "garage",
}

RELATIONSHIP_ALIASES: dict[str, RoomToRoomRelationship] = {
    "north of": "north_of",
    "to the north of": "north_of",
    "just north of": "north_of",
    "immediately north of": "north_of",
    "south of": "south_of",
    "to the south of": "south_of",
    "just south of": "south_of",
    "immediately south of": "south_of",
    "east of": "east_of",
    "to the east of": "east_of",
    "just east of": "east_of",
    "immediately east of": "east_of",
    "west of": "west_of",
    "to the west of": "west_of",
    "just west of": "west_of",
    "immediately west of": "west_of",
}

RELATIONSHIP_DIRECTIONS: dict[str, RoomToRoomRelationship] = {
    "north": "north_of",
    "south": "south_of",
    "east": "east_of",
    "west": "west_of",
}

ORDINAL_WORDS: dict[str, int] = {
    "one": 1,
    "first": 1,
    "two": 2,
    "second": 2,
    "three": 3,
    "third": 3,
    "four": 4,
    "fourth": 4,
    "five": 5,
    "fifth": 5,
    "six": 6,
    "sixth": 6,
    "seven": 7,
    "seventh": 7,
    "eight": 8,
    "eighth": 8,
    "nine": 9,
    "ninth": 9,
    "ten": 10,
    "tenth": 10,
}

ROOM_ALIAS_PATTERN = re.compile(
    "|".join(sorted((re.escape(alias) for alias in ROOM_TYPE_ALIASES), key=len, reverse=True)),
    flags=re.IGNORECASE,
)

ORDINAL_PATTERN = re.compile(
    rf"(?P<number>\d+)|(?P<word>{'|'.join(ORDINAL_WORDS)})",
    flags=re.IGNORECASE,
)

COMPACT_ORDINAL_PATTERN = re.compile(
    r"(?:room|bedroom|bathroom)?\s*(\d+)\b",
    flags=re.IGNORECASE,
)


def link_rooms(room_a: Room, relationship_type: RoomToRoomRelationship, room_b: Room) -> None:
    """Link two rooms with bi-directional edges."""
    room_a.add_relationship(Relationship(target=room_b.id, relationship_type=relationship_type))
    inverse = RELATIONSHIP_INVERSES[relationship_type]
    room_b.add_relationship(Relationship(target=room_a.id, relationship_type=inverse))


# ---- grid helpers (for 16x16 nested layout) ----
def _levels_for_grid(grid_size: int, base: int = 4) -> int:
    levels = 1
    size = base
    while size < grid_size and levels < 5:
        levels += 1
        size *= base
    return levels

def section_to_cell(section: Section, grid_size: int = DEFAULT_GRID_SIZE) -> Tuple[int, int]:
    """Map cardinal/ordinal section to a representative cell in a grid_size x grid_size grid."""
    quarter = grid_size // 4
    # centers of quadrants
    centers = {
        "NW": (quarter, quarter),
        "NE": (quarter, grid_size - 1 - quarter),
        "SW": (grid_size - 1 - quarter, quarter),
        "SE": (grid_size - 1 - quarter, grid_size - 1 - quarter),
        "N": (quarter, grid_size // 2),
        "S": (grid_size - 1 - quarter, grid_size // 2),
        "E": (grid_size // 2, grid_size - 1 - quarter),
        "W": (grid_size // 2, quarter),
        "C": (grid_size // 2, grid_size // 2),
    }
    return centers.get(section, centers["C"])

def placement_hint_from_sections(
    rooms: Iterable[Room],
    grid_size: int = DEFAULT_GRID_SIZE,
    block_size: int | None = None,
    base: int = 4,
) -> Tuple[Dict[str, List[Tuple[int, int]]], Dict[str, List[List[Tuple[int, int]]]]]:
    """
    Given rooms with sections (from support text), build a coarse placement hint on a 16x16 grid
    plus a nested reference path per cell (coarse->fine across 4x4 tiles).
    Returns: (placement_cells, nested_paths)
    """
    placement: Dict[str, List[Tuple[int, int]]] = {}
    nested: Dict[str, List[List[Tuple[int, int]]]] = {}
    levels = _levels_for_grid(grid_size, base=base)
    # If not provided, choose a coarse block ~1/4 of the grid dimension to give rooms some area.
    block_size = block_size or max(2, grid_size // 4)

    for room in rooms:
        center_r, center_c = section_to_cell(room.section, grid_size=grid_size)
        cells: List[Tuple[int, int]] = []
        # build a small block around the center to give rooms some area; clamp to grid bounds
        for dr in range(-(block_size // 2), block_size // 2 + block_size % 2):
            for dc in range(-(block_size // 2), block_size // 2 + block_size % 2):
                r = min(grid_size - 1, max(0, center_r + dr))
                c = min(grid_size - 1, max(0, center_c + dc))
                cells.append((r, c))
        name = room.canonical_name
        placement[name] = cells
        nested[name] = [cell_to_nested(rc, base=base, levels=levels) for rc in cells]

    return placement, nested


class RoomGraphBuilder:
    """Incrementally build a graph of rooms from natural language sentences."""

    def __init__(self) -> None:
        self._rooms: list[Room] = []
        self._rooms_by_descriptor: dict[tuple[RoomType, int], Room] = {}
        self._next_id = 0
        self._next_ordinal: dict[RoomType, int] = {}

    @property
    def rooms(self) -> list[Room]:
        return self._rooms

    def get_or_create_room(self, descriptor: RoomDescriptor) -> Room:
        room_type, ordinal = descriptor
        if ordinal is None:
            existing_keys = [
                key for key in self._rooms_by_descriptor if key[0] == room_type
            ]
            if len(existing_keys) == 1:
                ordinal = existing_keys[0][1]
            else:
                ordinal = self._next_ordinal.get(room_type, 0) + 1
        key = (room_type, ordinal)
        self._next_ordinal[room_type] = max(self._next_ordinal.get(room_type, 0), ordinal)
        if key not in self._rooms_by_descriptor:
            room = Room(id=self._next_id, room_type=room_type, ordinal=ordinal)
            self._next_id += 1
            self._rooms.append(room)
            self._rooms_by_descriptor[key] = room
        return self._rooms_by_descriptor[key]

    def set_section(self, descriptor: RoomDescriptor, section: Section) -> None:
        room = self.get_or_create_room(descriptor)
        # Preserve first zone, record conflicts (don't overwrite)
        if room.section != "C" and room.section != section:
            conflict_msg = f"Section conflict: attempted to set '{section}' but room already has '{room.section}'"
            room.section_conflicts.append(conflict_msg)
            return  # Keep the old section
        if room.section == "C":
            room.section = section

    def add_relationship(
        self,
        src_descriptor: RoomDescriptor,
        relationship: RoomToRoomRelationship,
        dst_descriptor: RoomDescriptor,
    ) -> None:
        src_room = self.get_or_create_room(src_descriptor)
        dst_room = self.get_or_create_room(dst_descriptor)
        link_rooms(src_room, relationship, dst_room)


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalise_section(raw_section: str) -> Section | None:
    key = raw_section.lower().strip()
    # Strip common prefixes like "the ", "a ", "an "
    key = re.sub(r"^(the|a|an)\s+", "", key).strip()
    if key in SECTION_ALIASES:
        return SECTION_ALIASES[key]
    # Handle composed directions like "south east corner".
    for alias, section in sorted(
        SECTION_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if key.startswith(alias):
            return section
    return None


def _normalise_relationship(raw: str) -> RoomToRoomRelationship | None:
    candidate = raw.lower().strip()
    if candidate in RELATIONSHIP_DIRECTIONS:
        return RELATIONSHIP_DIRECTIONS[candidate]
    if candidate in RELATIONSHIP_ALIASES:
        return RELATIONSHIP_ALIASES[candidate]
    for alias, relationship in RELATIONSHIP_ALIASES.items():
        if alias in candidate:
            return relationship
    if candidate in RELATIONSHIP_INVERSES:
        return candidate  # Already canonical.
    return None


def _ordinal_from_match(match: re.Match[str]) -> int:
    number = match.group("number")
    if number is not None:
        return int(number)
    word = match.group("word")
    if word is not None:
        return ORDINAL_WORDS[word.lower()]
    raise ValueError("Unexpected ordinal match")


def _extract_room_descriptors(text: str) -> list[RoomDescriptor]:
    """Extract room descriptors using alias span scanning with finditer().
    
    This supports patterns like "bedroom 1 and bedroom 2" and "dining and lounge".
    Also handles compact forms like "bedroom2".
    Unknown room names are mapped to "other" type.
    """
    text = _normalise_whitespace(text.lower())
    descriptors: list[RoomDescriptor] = []
    matched_ranges: list[tuple[int, int]] = []
    
    # Find all room alias matches using finditer
    for match in ROOM_ALIAS_PATTERN.finditer(text):
        room_alias = match.group(0).lower()
        room_type = ROOM_TYPE_ALIASES.get(room_alias, "other")
        
        # Track matched ranges to avoid double-extraction
        matched_ranges.append((match.start(), match.end()))
        
        # Enhanced ordinal extraction: try (1) right side, (2) left side, (3) compact suffix
        ordinal: int | None = None
        
        # Try right side first (after the alias)
        right_context = text[match.end():]
        ordinal_match = ORDINAL_PATTERN.search(right_context)
        
        # Try left side if right side didn't match
        if ordinal_match is None:
            left_context = text[:match.start()]
            ordinal_match = ORDINAL_PATTERN.search(left_context)
        
        # Try compact suffix pattern (e.g., "bedroom2", "room1") - check text right after match
        if ordinal_match is None:
            # Check for compact form directly after the alias (no space, like "bedroom2")
            # Look at a small window after the match end
            compact_context = text[match.end():match.end()+5]  # Check up to 5 chars after
            compact_match = re.search(r'^(\d+)', compact_context)
            if compact_match:
                ordinal = int(compact_match.group(1))
            else:
                # Also check if the original text has a compact form around this position
                context_around = text[max(0, match.start()-5):match.end()+5]
                compact_match = COMPACT_ORDINAL_PATTERN.search(context_around)
                if compact_match:
                    ordinal = int(compact_match.group(1))
        
        if ordinal_match:
            ordinal = _ordinal_from_match(ordinal_match)
        
        descriptors.append(RoomDescriptor(room_type, ordinal))
    
    # Fallback: extract unknown room-like words that weren't matched by known aliases
    # Common words to skip
    skip_words = {
        "the", "a", "an", "this", "that", "these", "those", "is", "are", "was", "were",
        "at", "in", "on", "to", "of", "and", "or", "but", "with", "from", "for",
        "located", "combined", "towards", "north", "south", "east", "west", "corner",
        "center", "centre", "central", "middle", "just", "immediately"
    }
    
    # Find words that look like room names but weren't matched
    for match in re.finditer(r'\b([a-z]{3,})\b', text):
        word = match.group(1).lower()
        
        # Skip if already matched by known alias or is a common word
        if word in skip_words or word in ROOM_TYPE_ALIASES:
            continue
        
        # Check if this word overlaps with any matched range
        word_start, word_end = match.start(), match.end()
        is_overlapped = any(
            word_start < end and word_end > start
            for start, end in matched_ranges
        )
        
        if is_overlapped:
            continue
        
        # Extract ordinal if present near this word
        ordinal: int | None = None
        
        # Check right side for ordinal
        right_context = text[word_end:]
        ordinal_match = ORDINAL_PATTERN.search(right_context)
        
        # Check left side for ordinal
        if ordinal_match is None:
            left_context = text[:word_start]
            ordinal_match = ORDINAL_PATTERN.search(left_context)
        
        # Check compact form (e.g., "atrium2")
        if ordinal_match is None:
            compact_context = text[word_end:word_end+5]
            compact_match = re.search(r'^(\d+)', compact_context)
            if compact_match:
                ordinal = int(compact_match.group(1))
        
        if ordinal_match:
            ordinal = _ordinal_from_match(ordinal_match)
        
        # Map unknown word to "other" type
        descriptors.append(RoomDescriptor("other", ordinal))
        matched_ranges.append((word_start, word_end))  # Track to avoid duplicates
    
    return descriptors


def _split_sentences(text: str) -> Iterable[str]:
    for raw_sentence in re.split(r"[.;]\s*", text):
        sentence = raw_sentence.strip()
        if sentence:
            yield sentence


class TinyRelationshipClassifier:
    """Very small helper that emulates a classifier over directional phrases."""

    RELATIONSHIP_REGEX = re.compile(
        r"(?P<src>.+?)\s+(?:is|are)\s+(?:located\s+)?(?:to\s+the\s+)?(?:just\s+)?(?:immediately\s+)?"
        r"(?P<relation>north|south|east|west)\s+of\s+(?P<dst>.+)",
        flags=re.IGNORECASE,
    )

    LOCATION_REGEX = re.compile(
        r"(?P<src>.+?)\s+(?:is|are)\s+(?:combined\s+and\s+)?(?:located\s+)?"
        r"(?:at|in|on|towards)\s+(?P<section>[a-z\s-]+)",
        flags=re.IGNORECASE,
    )

    def classify(self, sentence: str) -> tuple[str, re.Match[str] | None]:
        # Use search() instead of match() to catch clauses not at sentence start
        if match := self.RELATIONSHIP_REGEX.search(sentence):
            return ("relationship", match)
        if match := self.LOCATION_REGEX.search(sentence):
            return ("location", match)
        return ("unknown", None)


def parse_supporting_text(text: str) -> list[Room]:
    """Parse natural language into a list of rooms with relationships."""
    builder = RoomGraphBuilder()
    classifier = TinyRelationshipClassifier()
    for sentence in _split_sentences(text):
        kind, match = classifier.classify(sentence)
        if match is None:
            continue
        if kind == "location":
            section = _normalise_section(match.group("section"))
            if section is None:
                continue
            for descriptor in _extract_room_descriptors(match.group("src")):
                builder.set_section(descriptor, section)
            continue
        if kind == "relationship":
            relationship = _normalise_relationship(match.group("relation"))
            if relationship is None:
                continue
            src_descriptors = _extract_room_descriptors(match.group("src"))
            dst_descriptors = _extract_room_descriptors(match.group("dst"))
            for src in src_descriptors:
                for dst in dst_descriptors:
                    builder.add_relationship(src, relationship, dst)
            continue
    return builder.rooms


def rooms_to_support_text_payload(rooms: list[Room]) -> list[dict[str, object]]:
    """Helper to export rooms into a serialisable structure."""
    payload: list[dict[str, object]] = []
    for room in rooms:
        payload.append(
            {
                "name": room.canonical_name,
                "section": room.section,
                "relationships": [
                    {"target": rel.target, "relationship_type": rel.relationship_type}
                    for rel in room.relationships
                ],
            }
        )
    return payload
