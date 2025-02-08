import logging
import asyncio
from typing import List, Dict
from communication_layer import DriverAppIntegration, BrokerCommunicationModule, NotificationSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoadAssignmentExecution:
    def __init__(self, driver_app_integration):
        self.driver_app_integration = driver_app_integration

    async def execute_load_assignment(self, assignments: List[Dict]):
        for assignment in assignments:
            driver_id = assignment['driver_id']
            load_info = {"load_id": assignment['load_id']}
            await self.driver_app_integration.send_load_info(driver_id, load_info)
            confirmation = await self.driver_app_integration.receive_confirmation(driver_id)
            if confirmation['status'] == 'success':
                logger.info(f"Load {assignment['load_id']} assigned to driver {driver_id}")
            else:
                logger.error(f"Failed to assign load {assignment['load_id']} to driver {driver_id}")

class RateConfirmationExecution:
    def __init__(self, broker_comm_module):
        self.broker_comm_module = broker_comm_module

    async def execute_rate_confirmation(self, rate_confirmations: List[Dict]):
        for rate_confirmation in rate_confirmations:
            broker_id = rate_confirmation['broker_id']
            rate_info = {"load_id": rate_confirmation['load_id'], "rate": rate_confirmation['final_rate']}
            response = await self.broker_comm_module.send_rate_confirmation(broker_id, rate_info)
            if response['status'] == 'success':
                logger.info(f"Rate confirmation for load {rate_confirmation['load_id']} sent to broker {broker_id}")
            else:
                logger.error(f"Failed to send rate confirmation for load {rate_confirmation['load_id']} to broker {broker_id}")

class IssueResolutionExecution:
    def __init__(self, driver_app_integration, notification_system):
        self.driver_app_integration = driver_app_integration
        self.notification_system = notification_system

    async def execute_issue_resolution(self, issues: List[Dict]):
        for issue in issues:
            if issue['type'] == 'REROUTE':
                driver_id = issue['driver_id']
                new_route = issue['new_route']
                await self.driver_app_integration.send_load_info(driver_id, new_route)
                await self.notification_system.send_notification(driver_id, f"Reroute: {new_route}")
                logger.info(f"Rerouted driver {driver_id} to new route {new_route}")
            elif issue['type'] == 'RESCHEDULE':
                driver_id = issue['driver_id']
                new_schedule = issue['new_schedule']
                await self.driver_app_integration.send_load_info(driver_id, new_schedule)
                await self.notification_system.send_notification(driver_id, f"Reschedule: {new_schedule}")
                logger.info(f"Rescheduled driver {driver_id} with new schedule {new_schedule}")

# Example usage
async def main():
    driver_app = DriverAppIntegration()
    broker_comm = BrokerCommunicationModule()
    notification_sys = NotificationSystem()

    load_assignment_exec = LoadAssignmentExecution(driver_app)
    rate_confirmation_exec = RateConfirmationExecution(broker_comm)
    issue_resolution_exec = IssueResolutionExecution(driver_app, notification_sys)

    # Sample data
    assignments = [{"load_id": "L1", "driver_id": "D1"}]
    rate_confirmations = [{"load_id": "L1", "broker_id": "B1", "final_rate": 5000}]
    issues = [{"type": "REROUTE", "driver_id": "D1", "new_route": {"load_id": "L1", "pickup": "Location A", "delivery": "Location B"}}]

    # Execute assignments
    await load_assignment_exec.execute_load_assignment(assignments)
    # Execute rate confirmations
    await rate_confirmation_exec.execute_rate_confirmation(rate_confirmations)
    # Execute issue resolutions
    await issue_resolution_exec.execute_issue_resolution(issues)

if __name__ == "__main__":
    asyncio.run(main())