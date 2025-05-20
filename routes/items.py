# 📁 llm-fastapi-demo/routes/items.py
from typing import List

from fastapi import APIRouter, HTTPException

from models.schema import Item

router = APIRouter()
items_db: List[Item] = []


@router.get("/items")
def get_items():
    return items_db


@router.get("/items/{item_id}")
def get_item(item_id: int):
    for item in items_db:
        if item.id == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")


@router.post("/items")
def create_item(item: Item):
    items_db.append(item)
    return item


@router.put("/items/{item_id}")
def update_item(item_id: int, updated_item: Item):
    for idx, item in enumerate(items_db):
        if item.id == item_id:
            items_db[idx] = updated_item
            return updated_item
    raise HTTPException(status_code=404, detail="Item not found")


@router.delete("/items/{item_id}")
def delete_item(item_id: int):
    for item in items_db:
        if item.id == item_id:
            items_db.remove(item)
            return {"message": "Item deleted"}
    raise HTTPException(status_code=404, detail="Item not found")
