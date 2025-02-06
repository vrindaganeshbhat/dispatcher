import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Load:
    load_id: str
    pickup_location: Dict[str, float]
    delivery_location: Dict[str, float]
    pickup_time: datetime
    delivery_deadline: datetime
    weight: float
    rate: float
    status: str
    priority: int
    broker_id: str
    special_requirements: List[str] = None

@dataclass
class Driver:
    driver_id: str
    current_location: Dict[str, float]
    hours_of_service: float
    preferred_zones: List[str]
    performance_score: float
    current_status: str
    last_break_time: datetime
    specializations: List[str] = None

class AnalyticsEngine:
    def __init__(self):
        self.current_time = datetime.now()
        self.scaler = StandardScaler()
        self.label_encoders = defaultdict(LabelEncoder)
        self.models = self._initialize_models()
        self._train_models()

    def _initialize_models(self):
        return {
            'demand_prediction': GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            ),
            'delivery_time': GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            )
        }

    def _train_models(self):
        training_data = self._generate_training_data()
        for name, model in self.models.items():
            X, y = self._prepare_training_data(training_data, target=name)
            model.fit(X, y)

    def _generate_training_data(self):
        # Generate synthetic training data
        dates = pd.date_range(end=self.current_time, periods=720, freq='H')
        zones = ['Zone_A', 'Zone_B', 'Zone_C']
        data = []
        
        for date in dates:
            for zone in zones:
                demand = 5 + 3 * np.sin(date.hour/24 * 2 * np.pi)
                demand *= 1 + 0.3 * np.sin(date.weekday()/7 * 2 * np.pi)
                demand = max(0, int(demand + np.random.normal(0, 2)))
                
                data.append({
                    'timestamp': date,
                    'zone': zone,
                    'hour': date.hour,
                    'day_of_week': date.weekday(),
                    'demand': demand,
                    'delivery_time': 30 + np.random.normal(0, 5)
                })
        
        return pd.DataFrame(data)

    def _prepare_training_data(self, df, target):
        X = df[['hour', 'day_of_week']].copy()
        X['zone'] = self.label_encoders['zone'].fit_transform(df['zone'])
        y = df[target]
        return X, y

class DecisionMakingLayer:
    def __init__(self):
        self.analytics = AnalyticsEngine()
        self.load_booking = LoadBookingModule()
        self.driver_assignment = DriverAssignmentModule()
        self.rate_negotiation = RateNegotiationModule()
        self.issue_resolution = IssueResolutionModule()

    def process_decision(self, loads: List[Load], drivers: List[Driver], market_conditions: dict) -> dict:
        try:
            # Step 1: Load Booking Decisions
            filtered_loads = self.load_booking.filter_and_prioritize(loads, market_conditions)
            
            # Step 2: Driver Assignment
            assignments = self.driver_assignment.assign_drivers(filtered_loads, drivers)
            
            # Step 3: Rate Negotiations
            rates = self.rate_negotiation.negotiate_rates(filtered_loads, market_conditions)
            
            # Step 4: Issue Resolution
            issues = self.issue_resolution.check_and_resolve(assignments, rates)
            
            return {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'load_decisions': filtered_loads,
                'assignments': assignments,
                'rates': rates,
                'issues': issues,
                'recommendations': self._generate_recommendations(filtered_loads, assignments, issues)
            }
        except Exception as e:
            print(f"Error in decision making process: {e}")
            return None

    def _generate_recommendations(self, loads, assignments, issues):
        recommendations = []
        
        # Generate load-based recommendations
        for load in loads:
            if load['priority_score'] > 0.8:
                recommendations.append({
                    'type': 'HIGH_PRIORITY_LOAD',
                    'action': 'Expedite processing',
                    'load_id': load['load_id']
                })
        
        # Generate assignment-based recommendations
        for assignment in assignments:
            if assignment.get('fitness_score', 0) < 0.7:
                recommendations.append({
                    'type': 'SUBOPTIMAL_ASSIGNMENT',
                    'action': 'Review assignment',
                    'assignment_id': assignment['load_id']
                })
        
        return recommendations

class LoadBookingModule:
    def filter_and_prioritize(self, loads: List[Load], market_conditions: dict) -> List[dict]:
        prioritized_loads = []
        
        for load in loads:
            score = self._calculate_priority_score(load, market_conditions)
            if score > 0.5:  # Basic filtering threshold
                prioritized_loads.append({
                    'load_id': load.load_id,
                    'priority_score': score,
                    'recommended_action': 'BOOK' if score > 0.8 else 'REVIEW'
                })
        
        return sorted(prioritized_loads, key=lambda x: x['priority_score'], reverse=True)

    def _calculate_priority_score(self, load: Load, market_conditions: dict) -> float:
        time_score = self._calculate_time_urgency(load)
        rate_score = self._calculate_rate_attractiveness(load, market_conditions)
        return 0.6 * time_score + 0.4 * rate_score

    def _calculate_time_urgency(self, load: Load) -> float:
        hours_until_deadline = (load.delivery_deadline - datetime.now()).total_seconds() / 3600
        if hours_until_deadline <= 24:
            return 1.0
        elif hours_until_deadline <= 48:
            return 0.8
        return 0.5

    def _calculate_rate_attractiveness(self, load: Load, market_conditions: dict) -> float:
        market_rate = market_conditions.get('average_rate', 3.0)
        return min(1.0, load.rate / (market_rate * 100))

class DriverAssignmentModule:
    def assign_drivers(self, loads: List[dict], drivers: List[Driver]) -> List[dict]:
        assignments = []
        available_drivers = [d for d in drivers if self._is_driver_available(d)]
        
        for load in loads:
            best_driver = self._find_best_driver(load, available_drivers)
            if best_driver:
                assignments.append({
                    'load_id': load['load_id'],
                    'driver_id': best_driver.driver_id,
                    'fitness_score': self._calculate_fitness_score(load, best_driver)
                })
        
        return assignments

    def _is_driver_available(self, driver: Driver) -> bool:
        return (driver.current_status == 'AVAILABLE' and 
                driver.hours_of_service < 11 and 
                (datetime.now() - driver.last_break_time).total_seconds() / 3600 < 8)

    def _find_best_driver(self, load: dict, drivers: List[Driver]) -> Optional[Driver]:
        best_score = -1
        best_driver = None
        
        for driver in drivers:
            score = self._calculate_fitness_score(load, driver)
            if score > best_score:
                best_score = score
                best_driver = driver
        
        return best_driver

    def _calculate_fitness_score(self, load: dict, driver: Driver) -> float:
        # Simplified scoring - in production would include more factors
        performance_weight = 0.6
        hours_weight = 0.4
        
        performance_score = driver.performance_score / 5.0
        hours_score = 1 - (driver.hours_of_service / 11.0)
        
        return performance_score * performance_weight + hours_score * hours_weight

class RateNegotiationModule:
    def negotiate_rates(self, loads: List[dict], market_conditions: dict) -> List[dict]:
        negotiations = []
        
        for load in loads:
            base_rate = self._calculate_base_rate(load)
            adjusted_rate = self._adjust_for_market(base_rate, market_conditions)
            
            negotiations.append({
                'load_id': load['load_id'],
                'original_rate': base_rate,
                'proposed_rate': adjusted_rate,
                'negotiation_status': 'PROPOSED'
            })
        
        return negotiations

    def _calculate_base_rate(self, load: dict) -> float:
        # Simplified rate calculation
        return 300.0  # Base rate per load

    def _adjust_for_market(self, base_rate: float, market_conditions: dict) -> float:
        multiplier = 1.0
        if market_conditions['demand_level'] == 'HIGH':
            multiplier = 1.2
        elif market_conditions['demand_level'] == 'LOW':
            multiplier = 0.9
        return base_rate * multiplier

class IssueResolutionModule:
    def check_and_resolve(self, assignments: List[dict], rates: List[dict]) -> List[dict]:
        issues = []
        
        # Check assignment issues
        for assignment in assignments:
            if assignment['fitness_score'] < 0.6:
                issues.append({
                    'type': 'LOW_FITNESS_SCORE',
                    'severity': 'HIGH',
                    'load_id': assignment['load_id'],
                    'recommendation': 'Review driver assignment'
                })
        
        # Check rate issues
        for rate in rates:
            if rate['proposed_rate'] < rate['original_rate'] * 0.8:
                issues.append({
                    'type': 'LOW_RATE',
                    'severity': 'MEDIUM',
                    'load_id': rate['load_id'],
                    'recommendation': 'Review rate calculation'
                })
        
        return issues

def main():
    # Initialize decision making layer
    decision_maker = DecisionMakingLayer()
    
    # Sample test data
    test_loads = [
        Load(
            load_id="L001",
            pickup_location={"lat": 40.7128, "lng": -74.0060},
            delivery_location={"lat": 34.0522, "lng": -118.2437},
            pickup_time=datetime.now(),
            delivery_deadline=datetime.now() + timedelta(days=2),
            weight=15000.0,
            rate=3500.0,
            status="NEW",
            priority=4,
            broker_id="B001"
        )
    ]
    
    test_drivers = [
        Driver(
            driver_id="D001",
            current_location={"lat": 40.7128, "lng": -74.0060},
            hours_of_service=6.0,
            preferred_zones=["Northeast"],
            performance_score=4.8,
            current_status="AVAILABLE",
            last_break_time=datetime.now() - timedelta(hours=2)
        )
    ]
    
    test_market_conditions = {
        "demand_level": "HIGH",
        "average_rate": 3.2,
        "competition_level": "MEDIUM"
    }
    
    # Process decisions
    results = decision_maker.process_decision(
        test_loads,
        test_drivers,
        test_market_conditions
    )
    
    # Display results
    if results:
        print("\nDecision Making Results:")
        print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    main()