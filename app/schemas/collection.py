from pydantic import BaseModel


def url_to_category(url: str) -> str:
    url_lower = url.lower()
    if "/reel/" in url_lower:
        return "reels"
    if "/groups/" in url_lower:
        return "groups"
    if "/videos/" in url_lower or "/watch/" in url_lower:
        return "videos"
    if "/posts/" in url_lower:
        return "posts"
    if "/marketplace/" in url_lower or "/products/" in url_lower:
        return "products"
    if "/events/" in url_lower:
        return "events"
    return "post"


class FacebookSave(BaseModel):
    url: str
    category: str


class FacebookCollection(BaseModel):
    name: str
    saves: list[FacebookSave]


def parse_fb_collections(data: list[dict]) -> list[FacebookCollection]:
    """Parse hoarder export JSON into a list of Collection objects with URL-based categories."""
    collections = []
    for item in data:
        name = ""
        saves: list[FacebookSave] = []
        seen_urls: set[str] = set()

        for lv in item.get("label_values", []):
            if lv.get("label") == "Title":
                name = lv.get("value", "")
            if lv.get("title") == "Saves":
                for s in lv.get("dict", []):
                    url = next(
                        (x["value"] for x in s.get("dict", []) if x.get("label") == "URL"),
                        None,
                    )
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        saves.append(
                            FacebookSave(url=url, category=url_to_category(url))
                        )

        collections.append(FacebookCollection(name=name, saves=saves))

    return collections
