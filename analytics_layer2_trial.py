import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class SmartDispatchAnalytics:
    def __init__(self):
        self.current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.current_user = "vrindaganeshbhat"
        print(f"Smart Dispatch Analytics Engine Initializing...")
        print(f"Current Time (UTC): {self.current_time}")
        print(f"Dispatcher: {self.current_user}\n")

        # Initialize components
        self._init_models()
        self._init_metrics()
        self._init_rules()
        
        # Initialize zone status tracking
        self.zone_status = defaultdict(lambda: {
            'active_deliveries': 0,
            'active_drivers': set(),
            'demand_level': 'NORMAL',
            'last_updated': None
        })

    def _init_models(self):
        """Initialize prediction models"""
        try:
            self.models = {
                'demand_prediction': GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                ),
                'time_prediction': GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                ),
                'driver_allocation': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
            }
            print("Models initialized successfully")
        except Exception as e:
            print(f"Error initializing models: {e}")

    def _init_metrics(self):
        """Initialize tracking metrics"""
        self.metrics = {
            'real_time': {
                'zone_metrics': defaultdict(lambda: {
                    'active_deliveries': 0,
                    'active_drivers': set(),
                    'average_delivery_time': [],
                    'success_rate': []
                }),
                'driver_metrics': defaultdict(lambda: {
                    'deliveries_today': 0,
                    'active_hours': 0,
                    'success_rate': [],
                    'current_zone': None
                }),
                'system_metrics': {
                    'total_active_deliveries': 0,
                    'total_active_drivers': 0,
                    'system_load': 0
                }
            },
            'historical': defaultdict(list),
            'predictions': defaultdict(dict)
        }

    def _init_rules(self):
        """Initialize dispatch rules"""
        self.rules = {
            'driver_assignment': {
                'max_hours': 8,
                'max_deliveries': 20,
                'min_rest_time': 30
            },
            'zone_management': {
                'high_demand_threshold': 15,
                'low_demand_threshold': 3,
                'rebalance_threshold': 0.7
            },
            'priority_handling': {
                'urgent_response_time': 15,
                'high_priority_time': 30,
                'normal_response_time': 60
            }
        }

    def _analyze_zone_status(self, zone):
        """Analyze current zone status"""
        try:
            zone_metrics = self.metrics['real_time']['zone_metrics'][zone]
            active_deliveries = zone_metrics['active_deliveries']
            active_drivers = len(zone_metrics['active_drivers'])
            
            # Determine demand level
            if active_deliveries >= self.rules['zone_management']['high_demand_threshold']:
                demand_level = 'HIGH'
            elif active_deliveries <= self.rules['zone_management']['low_demand_threshold']:
                demand_level = 'LOW'
            else:
                demand_level = 'NORMAL'
            
            return {
                'zone': zone,
                'active_deliveries': active_deliveries,
                'active_drivers': active_drivers,
                'demand_level': demand_level,
                'driver_utilization': active_deliveries / max(active_drivers, 1),
                'needs_rebalancing': active_deliveries / max(active_drivers, 1) > self.rules['zone_management']['rebalance_threshold']
            }
        except Exception as e:
            print(f"Error analyzing zone status: {e}")
            return {
                'zone': zone,
                'demand_level': 'NORMAL',
                'needs_rebalancing': False
            }

    def _update_metrics(self, request_data):
        """Update system metrics with new request data"""
        try:
            zone = request_data['zone']
            timestamp = datetime.now()
            
            # Update zone metrics
            self.metrics['real_time']['zone_metrics'][zone]['active_deliveries'] += 1
            self.metrics['real_time']['zone_metrics'][zone]['last_updated'] = timestamp
            
            # Update system metrics
            self.metrics['real_time']['system_metrics']['total_active_deliveries'] += 1
            
            # Store historical data
            self.metrics['historical'][zone].append({
                'timestamp': timestamp,
                'request_data': request_data
            })
            
        except Exception as e:
            print(f"Error updating metrics: {e}")

    def _calculate_resource_needs(self, request_data):
        """Calculate required resources for request"""
        try:
            zone = request_data['zone']
            current_status = self._analyze_zone_status(zone)
            
            # Calculate needed drivers
            needed_drivers = max(1, current_status['active_deliveries'] // 5)
            
            return {
                'drivers_needed': needed_drivers,
                'current_availability': current_status['active_drivers'],
                'resource_gap': needed_drivers - current_status['active_drivers']
            }
        except Exception as e:
            print(f"Error calculating resource needs: {e}")
            return {'drivers_needed': 1, 'resource_gap': 0}

    def _analyze_timing(self, request_data):
        """Analyze timing requirements"""
        try:
            current_hour = datetime.now().hour
            
            # Check time window if provided
            if 'time_window' in request_data:
                time_window = request_data['time_window']
                end_hour = int(time_window['end'].split(':')[0])
                
                is_urgent = current_hour + 1 >= end_hour
                time_remaining = end_hour - current_hour
                
                return {
                    'is_urgent': is_urgent,
                    'time_remaining': time_remaining,
                    'recommended_priority': 'HIGH' if is_urgent else 'NORMAL'
                }
            
            return {
                'is_urgent': False,
                'recommended_priority': 'NORMAL'
            }
            
        except Exception as e:
            print(f"Error analyzing timing: {e}")
            return {'is_urgent': False, 'recommended_priority': 'NORMAL'}

    def process_dispatch_request(self, request_data):
        """Process a new dispatch request"""
        try:
            print(f"\nProcessing Dispatch Request: {request_data.get('request_id', 'N/A')}")
            
            # Analyze request
            analysis = {
                'zone_status': self._analyze_zone_status(request_data['zone']),
                'resource_needs': self._calculate_resource_needs(request_data),
                'timing_analysis': self._analyze_timing(request_data)
            }
            
            # Generate recommendations
            recommendations = {
                'priority': analysis['timing_analysis']['recommended_priority'],
                'resource_allocation': self._recommend_resource_allocation(analysis),
                'dispatch_timing': self._recommend_dispatch_timing(analysis)
            }
            
            # Update metrics
            self._update_metrics(request_data)
            
            # Generate alerts
            alerts = self._generate_alerts(request_data, analysis)
            
            return {
                'timestamp': self.current_time,
                'request_id': request_data.get('request_id'),
                'analysis': analysis,
                'recommendations': recommendations,
                'alerts': alerts
            }
            
        except Exception as e:
            print(f"Error processing dispatch request: {e}")
            return None

    def _recommend_resource_allocation(self, analysis):
        """Recommend resource allocation"""
        try:
            resource_needs = analysis['resource_needs']
            zone_status = analysis['zone_status']
            
            return {
                'recommended_drivers': resource_needs['drivers_needed'],
                'rebalancing_needed': zone_status['needs_rebalancing'],
                'priority_level': 'HIGH' if resource_needs['resource_gap'] > 0 else 'NORMAL'
            }
        except Exception as e:
            print(f"Error recommending resource allocation: {e}")
            return {}

    def _recommend_dispatch_timing(self, analysis):
        """Recommend dispatch timing"""
        try:
            timing = analysis['timing_analysis']
            zone_status = analysis['zone_status']
            
            if timing['is_urgent']:
                return {
                    'recommended_time': 'IMMEDIATE',
                    'reason': 'Time-critical delivery'
                }
            elif zone_status['demand_level'] == 'HIGH':
                return {
                    'recommended_time': 'PRIORITY',
                    'reason': 'High zone demand'
                }
            else:
                return {
                    'recommended_time': 'NORMAL',
                    'reason': 'Standard processing'
                }
        except Exception as e:
            print(f"Error recommending dispatch timing: {e}")
            return {}

    def _generate_alerts(self, request_data, analysis):
        """Generate alerts based on analysis"""
        try:
            alerts = []
            
            # Check zone status
            if analysis['zone_status']['demand_level'] == 'HIGH':
                alerts.append({
                    'type': 'HIGH_DEMAND',
                    'severity': 'HIGH',
                    'message': f"High demand in {request_data['zone']}",
                    'recommended_action': "Consider reassigning drivers"
                })
            
            # Check resource gap
            if analysis['resource_needs']['resource_gap'] > 0:
                alerts.append({
                    'type': 'RESOURCE_SHORTAGE',
                    'severity': 'MEDIUM',
                    'message': f"Need {analysis['resource_needs']['resource_gap']} more drivers",
                    'recommended_action': "Allocate additional drivers"
                })
            
            # Check timing
            if analysis['timing_analysis']['is_urgent']:
                alerts.append({
                    'type': 'URGENT_TIMING',
                    'severity': 'HIGH',
                    'message': "Time-critical delivery",
                    'recommended_action': "Prioritize processing"
                })
            
            return alerts
            
        except Exception as e:
            print(f"Error generating alerts: {e}")
            return []

def main():
    # Initialize analytics engine
    dispatcher = SmartDispatchAnalytics()
    
    # Test dispatch request
    test_request = {
        'request_id': 'REQ001',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'zone': 'Zone A',
        'priority': 'HIGH',
        'time_window': {
            'start': '14:00',
            'end': '16:00'
        }
    }
    
    # Process request
    result = dispatcher.process_dispatch_request(test_request)
    
    # Display results
    if result:
        print("\nDispatch Analysis Results:")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()