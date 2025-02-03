from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Load(BaseModel):
    load_id: int
    origin: str
    destination: str
    driver_id: int
    priority: int

class Driver(BaseModel):
    driver_id: int
    name: str
    location: str
    status: str

@app.get("/loads/", response_model=List[Load])
def get_loads():
    return [{"load_id": 1, "origin": "Chicago", "destination": "NYC", "driver_id": 1, "priority": 1}]

@app.post("/assign_load/")
def assign_load(load_id: int, driver_id: int):
    return {"message": f"Load {load_id} assigned to driver {driver_id}"}
