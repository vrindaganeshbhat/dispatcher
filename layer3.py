import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from typing import List, Optional
from enum import Enum
import logging
import asyncio
import math
from dataclasses import dataclass
import json

# System Configuration
SYSTEM_TIME = datetime.strptime("2025-02-07 22:25:32", "%Y-%m-%d %H:%M:%S")
SYSTEM_USER = "vrindaganeshbhat"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core Data Models
@dataclass
class Location:
    lat: float
    lng: float

    def distance_to(self, other: 'Location') -> float:
        R = 6371
        lat1, lon1 = math.radians(self.lat), math.radians(self.lng)
        lat2, lon2 = math.radians(other.lat), math.radians(other.lng)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return R * 2 * math.asin(math.sqrt(a))

@dataclass
class Load:
    load_id: str
    pickup: Location
    delivery: Location
    pickup_time: datetime
    delivery_deadline: datetime
    weight: float
    distance: float
    rate: float

    def get_urgency_score(self) -> float:
        hours_remaining = (self.delivery_deadline - SYSTEM_TIME).total_seconds() / 3600
        return max(0.2, 1 - (hours_remaining - 48) / 72) if hours_remaining > 0 else 1.0

@dataclass
class Driver:
    driver_id: str
    location: Location
    hours_of_service: float
    performance_score: float
    status: str
    last_break: datetime

    def can_accept_load(self, load: Load) -> bool:
        return self.status == "AVAILABLE" and self.hours_of_service < 11 and (SYSTEM_TIME - self.last_break).total_seconds() / 3600 < 8

# Enums
class LoadStatus(Enum):
    NEW = "NEW"
    ASSIGNED = "ASSIGNED"
    IN_TRANSIT = "IN_TRANSIT"
    COMPLETED = "COMPLETED"

class DriverStatus(Enum):
    AVAILABLE = "AVAILABLE"
    ASSIGNED = "ASSIGNED"

class MarketCondition(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

# Main Decision Making Class
class DecisionMakingSystem:
    def __init__(self):
        self.scaler = MinMaxScaler()
        logger.info(f"Decision Making System initialized at {SYSTEM_TIME}")

    async def process_dispatch_request(self, loads: List[Load], drivers: List[Driver], market_data: dict) -> dict:
        try:
            bookings = await self.process_loads(loads, drivers, market_data)
            assignments = await self.assign_drivers(loads, drivers)
            rates = await self.negotiate_rates(loads, market_data)
            issues = await self.analyze_decisions(bookings, assignments, rates)
            return {"success": True, "data": {"bookings": bookings, "assignments": assignments, "rates": rates, "issues": issues}}
        except Exception as e:
            logger.error(f"Error in dispatch processing: {str(e)}")
            return {"success": False, "error": str(e)}

    async def process_loads(self, loads: List[Load], drivers: List[Driver], market_data: dict) -> dict:
        processed_loads = []
        for load in loads:
            scores = await asyncio.gather(
                self._calculate_time_score(load),
                self._calculate_driver_availability_score(load, drivers),
                self._calculate_market_score(load, market_data),
                self._calculate_distance_score(load)
            )
            priority_score = sum(scores) / len(scores)
            processed_loads.append({"load_id": load.load_id, "priority_score": priority_score})
        return {"loads": processed_loads}

    async def assign_drivers(self, loads: List[Load], drivers: List[Driver]) -> dict:
        assignments = []
        available_drivers = {d.driver_id for d in drivers if d.can_accept_load(loads[0])}
        for load in loads:
            best_assignment = await self._find_best_driver(load, drivers, available_drivers)
            if best_assignment:
                assignments.append(best_assignment)
                available_drivers.remove(best_assignment['driver_id'])
        return {"assignments": assignments}

    async def negotiate_rates(self, loads: List[Load], market_data: dict) -> dict:
        negotiations = []
        for load in loads:
            result = await self._negotiate_load_rate(load, market_data)
            if result:
                negotiations.append(result)
        return {"negotiations": negotiations}

    async def analyze_decisions(self, bookings: dict, assignments: dict, rates: dict) -> dict:
        issues = []
        # Analyze bookings, assignments, and rates for issues (simplified for brevity)
        return {"issues": issues}

    async def _calculate_time_score(self, load: Load) -> float:
        return load.get_urgency_score()

    async def _calculate_driver_availability_score(self, load: Load, drivers: List[Driver]) -> float:
        available_drivers = [d for d in drivers if d.can_accept_load(load)]
        return min(1.0, len(available_drivers) / max(1, len(drivers) * 0.2))

    async def _calculate_market_score(self, load: Load, market_data: dict) -> float:
        market_rate = market_data.get('average_rate', 3.0)
        load_rate = load.rate / load.distance
        rate_ratio = load_rate / market_rate
        return max(0.3, rate_ratio) if rate_ratio >= 0.8 else rate_ratio

    async def _calculate_distance_score(self, load: Load) -> float:
        optimal_distance = 500
        distance_ratio = load.distance / optimal_distance
        return max(0.3, 1 - (distance_ratio - 2.0) / 2.0) if distance_ratio > 2.0 else 1.0

    async def _find_best_driver(self, load: Load, drivers: List[Driver], available_drivers: set) -> Optional[dict]:
        best_score = -1
        best_assignment = None
        for driver in drivers:
            if driver.driver_id not in available_drivers:
                continue
            fitness_score = await self._calculate_fitness_score(load, driver)
            if fitness_score > best_score:
                best_score = fitness_score
                best_assignment = {'load_id': load.load_id, 'driver_id': driver.driver_id, 'fitness_score': fitness_score}
        return best_assignment

    async def _calculate_fitness_score(self, load: Load, driver: Driver) -> float:
        location_score = 1 - (load.pickup.distance_to(driver.location) / 100)
        hours_score = 1 - (driver.hours_of_service / 11)
        performance_score = driver.performance_score / 5.0
        return (location_score * 0.3) + (hours_score * 0.25) + (performance_score * 0.25)

    async def _negotiate_load_rate(self, load: Load, market_data: dict) -> Optional[dict]:
        market_rate = market_data.get('average_rate', 3.0)
        current_rate = load.rate / load.distance
        rate_ratio = current_rate / market_rate
        if rate_ratio >= 0.8:
            return {'load_id': load.load_id, 'final_rate': current_rate, 'status': 'ACCEPTED'}
        return {'load_id': load.load_id, 'final_rate': market_rate, 'status': 'NEGOTIATED'}

# Example usage
async def main():
    system = DecisionMakingSystem()
    loads = [
        Load(
            load_id="L1",
            pickup=Location(lat=40.7128, lng=-74.0060),
            delivery=Location(lat=34.0522, lng=-118.2437),
            pickup_time=SYSTEM_TIME,
            delivery_deadline=SYSTEM_TIME + timedelta(days=2),
            weight=15000,
            distance=2789.0,
            rate=5000.0
        )
    ]
    drivers = [
        Driver(
            driver_id="D1",
            location=Location(lat=40.7128, lng=-74.0060),
            hours_of_service=8,
            performance_score=4.5,
            status="AVAILABLE",
            last_break=SYSTEM_TIME - timedelta(hours=4)
        )
    ]
    market_data = {'condition': "MEDIUM", 'average_rate': 3.0}
    result = await system.process_dispatch_request(loads, drivers, market_data)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())