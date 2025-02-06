import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class EnhancedAnalyticsEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = defaultdict(LabelEncoder)
        
        # Initialize models
        self.models = {
            'zone_demand': GradientBoostingRegressor(n_estimators=100),
            'driver_readiness': RandomForestClassifier(n_estimators=100),
            'delivery_time': GradientBoostingRegressor(n_estimators=100)
        }
        
        # Initialize storage
        self.real_time_metrics = {
            'active_zones': defaultdict(int),
            'active_drivers': defaultdict(dict),
            'current_demand': defaultdict(int),
            'performance_metrics': defaultdict(float)
        }
        
        self.historical_data = []

    def process_real_time_data(self, new_data):
        """Process real-time incoming data"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Update metrics
            self._update_real_time_metrics(new_data)
            
            # Store historical data
            self.historical_data.append(new_data)
            
            # Calculate current metrics
            current_metrics = self._calculate_current_metrics(new_data)
            
            # Generate predictions
            predictions = self._generate_real_time_predictions(new_data)
            
            # Generate alerts
            alerts = self._generate_alerts(new_data)
            
            return {
                'timestamp': timestamp,
                'current_metrics': current_metrics,
                'predictions': predictions,
                'alerts': alerts
            }
            
        except Exception as e:
            print(f"Error processing real-time data: {e}")
            return None

    def _calculate_current_metrics(self, data):
        """Calculate current performance metrics"""
        try:
            zone = data['zone']
            return {
                'zone_activity': {
                    'zone': zone,
                    'current_deliveries': self.real_time_metrics['active_zones'][zone],
                    'active_drivers': len([d for d in self.real_time_metrics['active_drivers'].values() 
                                         if d.get('current_zone') == zone])
                },
                'performance': {
                    'success_rate': data.get('success_rate', 0),
                    'average_time': data.get('average_time', 0)
                }
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {}

    def _update_real_time_metrics(self, data):
        """Update real-time monitoring metrics"""
        try:
            zone = data['zone']
            driver_id = data['driver_id']
            
            # Update zone activity
            self.real_time_metrics['active_zones'][zone] += 1
            
            # Update driver metrics
            self.real_time_metrics['active_drivers'][driver_id] = {
                'last_active': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current_zone': zone,
                'deliveries_today': self.real_time_metrics['active_drivers'].get(driver_id, {}).get('deliveries_today', 0) + 1
            }
            
            # Update demand metrics
            current_hour = datetime.now().strftime('%Y-%m-%d %H')
            self.real_time_metrics['current_demand'][current_hour] += 1
            
        except Exception as e:
            print(f"Error updating metrics: {e}")

    def _generate_real_time_predictions(self, data):
        """Generate real-time predictions"""
        try:
            zone = data['zone']
            current_hour = datetime.now().hour
            
            # Simple demand prediction based on historical data
            zone_history = [d for d in self.historical_data if d['zone'] == zone]
            avg_demand = len(zone_history) / max(1, len(set([d['timestamp'][:13] for d in zone_history])))
            
            predictions = {
                'expected_demand': {
                    'zone': zone,
                    'hour': current_hour,
                    'predicted_demand': round(avg_demand, 2)
                },
                'zone_status': self._predict_zone_status(data)
            }
            
            return predictions
            
        except Exception as e:
            print(f"Error generating predictions: {e}")
            return {}

    def _predict_zone_status(self, data):
        """Predict zone status"""
        try:
            zone = data['zone']
            current_activity = self.real_time_metrics['active_zones'][zone]
            
            if current_activity > 10:
                status = 'HIGH_DEMAND'
            elif current_activity > 5:
                status = 'MODERATE_DEMAND'
            else:
                status = 'LOW_DEMAND'
                
            return {
                'zone': zone,
                'status': status,
                'activity_level': current_activity
            }
            
        except Exception as e:
            print(f"Error predicting zone status: {e}")
            return {}

    def _generate_alerts(self, data):
        """Generate real-time alerts"""
        alerts = []
        try:
            zone = data['zone']
            driver_id = data['driver_id']
            
            # Check zone demand
            if self.real_time_metrics['active_zones'][zone] > 10:
                alerts.append({
                    'type': 'HIGH_DEMAND',
                    'zone': zone,
                    'severity': 'HIGH',
                    'message': f"High demand detected in {zone}"
                })
            
            # Check driver workload
            driver_deliveries = self.real_time_metrics['active_drivers'][driver_id]['deliveries_today']
            if driver_deliveries > 15:
                alerts.append({
                    'type': 'DRIVER_WORKLOAD',
                    'driver_id': driver_id,
                    'severity': 'MEDIUM',
                    'message': f"Driver {driver_id} has completed {driver_deliveries} deliveries today"
                })
                
            return alerts
            
        except Exception as e:
            print(f"Error generating alerts: {e}")
            return []

    def visualize_patterns(self, data):
        """Generate visualizations"""
        try:
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Zone Activity', 'Driver Performance', 
                              'Hourly Demand', 'Delivery Times')
            )
            
            # Add zone activity trace
            zone_data = pd.DataFrame(self.real_time_metrics['active_zones'].items(), 
                                   columns=['zone', 'activity'])
            fig.add_trace(
                go.Bar(x=zone_data['zone'], y=zone_data['activity'], name='Zone Activity'),
                row=1, col=1
            )
            
            # Update layout
            fig.update_layout(height=800, width=1200, title_text="Dispatch Analytics Dashboard")
            
            # Save to HTML file
            fig.write_html("dispatch_analytics.html")
            print("Visualization saved as 'dispatch_analytics.html'")
            
        except Exception as e:
            print(f"Error creating visualization: {e}")

def main():
    # Initialize analytics engine
    engine = EnhancedAnalyticsEngine()
    
    # Sample real-time data
    test_data = {
        'delivery_id': 'DEL123',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'zone': 'Zone A',
        'driver_id': 'DRV001',
        'delivery_count': 5,
        'success_rate': 0.95,
        'average_time': 25
    }
    
    # Process multiple data points
    for i in range(3):  # Simulate multiple deliveries
        test_data['delivery_id'] = f'DEL12{i}'
        results = engine.process_real_time_data(test_data)
        
        if results:
            print(f"\nProcessing Delivery {i+1}:")
            print(json.dumps(results, indent=2))
    
    # Generate visualization
    engine.visualize_patterns(test_data)

if __name__ == "__main__":
    main()