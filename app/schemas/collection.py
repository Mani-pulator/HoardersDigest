from pydantic import BaseModel


class FacebookCollection(BaseModel):
    name: str
    saves: list[str]


def parse_fb_collections(data: list[dict]) -> list[FacebookCollection]:
    """Parse hoarder export JSON into a list of Collection objects."""
    collections = []
    for item in data:
        name = ""
        saves = []

        for lv in item.get("label_values", []):
            if lv.get("label") == "Title":
                name = lv.get("value", "")
            if lv.get("title") == "Saves":
                for s in lv.get("dict", []):
                    url = next(
                        (x["value"] for x in s.get("dict", []) if x.get("label") == "URL"),
                        None,
                    )
                    if url:
                        saves.append(url)

        saves = list(dict.fromkeys(saves))
        collections.append(FacebookCollection(name=name, saves=saves))

    return collections
