from collections import defaultdict
from statistics import mean, stdev


class BehavioralBaseline:
    """
    Stores and calculates normal behavioral statistics
    for each employee.
    """

    def __init__(self):
        # {
        #   "EMP_001": {
        #       "file_access_count": [10, 12, 15, 11],
        #       "usb_connection_count": [0, 1, 0, 0],
        #   }
        # }
        self.history = defaultdict(lambda: defaultdict(list))

    def add_features(self, features: dict):
        """
        Add one feature vector to the employee's history.
        """

        user_id = features.get("user_id")

        if not user_id:
            return

        for feature, value in features.items():

            # user_id itself isn't a numerical behavioral feature
            if feature == "user_id":
                continue

            # Only store numerical values
            if isinstance(value, (int, float)):
                self.history[user_id][feature].append(value)

    def calculate_baseline(self, user_id: str) -> dict:
        """
        Calculate mean and standard deviation
        for each behavioral feature.
        """

        if user_id not in self.history:
            return {}

        baseline = {}

        for feature, values in self.history[user_id].items():

            if not values:
                continue

            avg = mean(values)

            # Standard deviation needs at least 2 values
            deviation = stdev(values) if len(values) > 1 else 0.0

            baseline[feature] = {
                "mean": avg,
                "std": deviation,
                "samples": len(values),
            }

        return baseline