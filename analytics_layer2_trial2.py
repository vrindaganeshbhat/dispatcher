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
import warnings
warnings.filterwarnings('ignore')

class SmartDispatchAnalytics:
    def __init__(self):
        self.current_time = datetime.strptime("2025-02-06 14:10:30", "%Y-%m-%d %H:%M:%S")
        self.current_user = "vrindaganeshbhat"
        print(f"Smart Dispatch Analytics Engine Initializing...")
        print(f"Current Time (UTC): {self.current_time}")
        print(f"Dispatcher: {self.current_user}\n")

        # Initialize preprocessing
        self.scaler = StandardScaler()
        self.label_encoders = defaultdict(LabelEncoder)
        
        # Initialize and train models
        self._initialize_and_train_models()

    def _initialize_and_train_models(self):
        """Initialize and train all models"""
        try:
            print("Initializing and training models...")
            
            # Initialize models
            self.models = {
                'demand_prediction': GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                ),
                'delivery_time': GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                ),
                'driver_assignment': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
            }
            
            # Generate and process training data
            training_data = self._generate_training_data()
            
            # Train models
            self._train_models(training_data)
            
            print("Models initialized and trained successfully!")
            
        except Exception as e:
            print(f"Error initializing models: {e}")

    def _generate_training_data(self):
        """Generate sample training data"""
        try:
            # Generate timestamps for the past 30 days
            dates = pd.date_range(end=self.current_time, periods=30*24, freq='H')
            
            # Generate sample data
            data = []
            zones = ['Zone A', 'Zone B', 'Zone C', 'Zone D']
            drivers = [f'Driver_{i}' for i in range(1, 11)]
            
            for date in dates:
                hour = date.hour
                day_of_week = date.weekday()
                
                for zone in zones:
                    # Base demand varies by hour and day
                    base_demand = 5 + 3 * np.sin(hour/24 * 2 * np.pi)
                    base_demand *= 1 + 0.3 * np.sin(day_of_week/7 * 2 * np.pi)
                    
                    # Add random noise
                    demand = max(0, int(base_demand + np.random.normal(0, 2)))
                    
                    # Generate delivery records
                    for _ in range(demand):
                        delivery_time = max(15, int(30 + np.random.normal(0, 5)))
                        success = np.random.random() > 0.1
                        
                        data.append({
                            'timestamp': date,
                            'zone': zone,
                            'hour': hour,
                            'day_of_week': day_of_week,
                            'driver_id': np.random.choice(drivers),
                            'delivery_time': delivery_time,
                            'success': success,
                            'demand': demand
                        })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Error generating training data: {e}")
            return pd.DataFrame()

    def _train_models(self, df):
        """Train all models with generated data"""
        try:
            # Train demand prediction model
            X_demand = df[['hour', 'day_of_week']].copy()
            X_demand['zone'] = self.label_encoders['zone'].fit_transform(df['zone'])
            y_demand = df['demand']
            
            X_train_demand, X_test_demand, y_train_demand, y_test_demand = train_test_split(
                X_demand, y_demand, test_size=0.2, random_state=42
            )
            
            self.models['demand_prediction'].fit(X_train_demand, y_train_demand)
            
            # Train delivery time model
            X_time = df[['hour', 'day_of_week', 'demand']].copy()
            X_time['zone'] = self.label_encoders['zone'].transform(df['zone'])
            y_time = df['delivery_time']
            
            X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(
                X_time, y_time, test_size=0.2, random_state=42
            )
            
            self.models['delivery_time'].fit(X_train_time, y_train_time)
            
            # Calculate and store model metrics
            self.model_metrics = {
                'demand_prediction': {
                    'mae': mean_absolute_error(
                        y_test_demand,
                        self.models['demand_prediction'].predict(X_test_demand)
                    ),
                    'r2': r2_score(
                        y_test_demand,
                        self.models['demand_prediction'].predict(X_test_demand)
                    )
                },
                'delivery_time': {
                    'mae': mean_absolute_error(
                        y_test_time,
                        self.models['delivery_time'].predict(X_test_time)
                    ),
                    'r2': r2_score(
                        y_test_time,
                        self.models['delivery_time'].predict(X_test_time)
                    )
                }
            }
            
            print("\nModel Performance Metrics:")
            print(f"Demand Prediction - MAE: {self.model_metrics['demand_prediction']['mae']:.2f}, "
                  f"R2: {self.model_metrics['demand_prediction']['r2']:.2f}")
            print(f"Delivery Time - MAE: {self.model_metrics['delivery_time']['mae']:.2f}, "
                  f"R2: {self.model_metrics['delivery_time']['r2']:.2f}")
            
        except Exception as e:
            print(f"Error training models: {e}")

    def predict_demand(self, zone, hour, day_of_week):
        """Predict demand for a specific zone and time"""
        try:
            X = pd.DataFrame({
                'hour': [hour],
                'day_of_week': [day_of_week],
                'zone': self.label_encoders['zone'].transform([zone])
            })
            
            return max(0, int(self.models['demand_prediction'].predict(X)[0]))
            
        except Exception as e:
            print(f"Error predicting demand: {e}")
            return None

    def predict_delivery_time(self, zone, hour, day_of_week, current_demand):
        """Predict delivery time for a specific zone and conditions"""
        try:
            X = pd.DataFrame({
                'hour': [hour],
                'day_of_week': [day_of_week],
                'zone': self.label_encoders['zone'].transform([zone]),
                'demand': [current_demand]
            })
            
            return max(15, int(self.models['delivery_time'].predict(X)[0]))
            
        except Exception as e:
            print(f"Error predicting delivery time: {e}")
            return None

    def analyze_request(self, request_data):
        """Analyze a new dispatch request"""
        try:
            zone = request_data['zone']
            current_hour = self.current_time.hour
            current_day = self.current_time.weekday()
            
            # Get predictions
            predicted_demand = self.predict_demand(zone, current_hour, current_day)
            predicted_delivery_time = self.predict_delivery_time(
                zone, current_hour, current_day, predicted_demand
            )
            
            # Generate analysis
            analysis = {
                'request_id': request_data.get('request_id'),
                'timestamp': self.current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'zone': zone,
                'predictions': {
                    'expected_demand': predicted_demand,
                    'estimated_delivery_time': predicted_delivery_time,
                    'confidence_scores': {
                        'demand': self.model_metrics['demand_prediction']['r2'],
                        'delivery_time': self.model_metrics['delivery_time']['r2']
                    }
                },
                'insights': self._generate_insights(zone, predicted_demand, predicted_delivery_time)
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing request: {e}")
            return None

    def _generate_insights(self, zone, predicted_demand, predicted_delivery_time):
        """Generate insights based on predictions"""
        try:
            demand_level = 'HIGH' if predicted_demand > 15 else 'MEDIUM' if predicted_demand > 8 else 'LOW'
            
            return {
                'demand_status': {
                    'level': demand_level,
                    'predicted_demand': predicted_demand,
                    'recommended_drivers': max(1, predicted_demand // 3)
                },
                'timing': {
                    'estimated_delivery_time': predicted_delivery_time,
                    'recommended_buffer': max(10, int(predicted_delivery_time * 0.2))
                },
                'recommendations': self._generate_recommendations(demand_level, predicted_delivery_time)
            }
            
        except Exception as e:
            print(f"Error generating insights: {e}")
            return {}

    def _generate_recommendations(self, demand_level, delivery_time):
        """Generate specific recommendations"""
        recommendations = []
        
        if demand_level == 'HIGH':
            recommendations.append({
                'type': 'RESOURCE',
                'priority': 'HIGH',
                'action': 'Increase driver allocation in zone'
            })
        
        if delivery_time > 45:
            recommendations.append({
                'type': 'OPTIMIZATION',
                'priority': 'MEDIUM',
                'action': 'Consider route optimization'
            })
        
        return recommendations

def main():
    # Initialize analytics engine
    engine = SmartDispatchAnalytics()
    
    # Test request
    test_request = {
        'request_id': 'REQ001',
        'zone': 'Zone A',
        'timestamp': '2025-02-06 14:10:30',
        'priority': 'HIGH'
    }
    
    # Analyze request
    print("\nAnalyzing test request...")
    result = engine.analyze_request(test_request)
    
    # Display results
    if result:
        print("\nAnalysis Results:")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()