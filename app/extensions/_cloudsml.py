# encoding: utf-8
"""
CloudSML API Client Adapter
-----------------
"""
import cloudsml


class CloudSML(object):  # pylint: disable=too-many-instance-attributes
    """
    Wrapper class around CloudSML API Client
    """

    def __init__(self):
        self.configuration = cloudsml.Configuration()
        self.auth_api = None
        self.data_api = None
        self.predictive_analytics_api = None
        self.scorings_api = None
        self.users_api = None
        self.workspaces_api = None
        self.models = cloudsml.models

    def init_app(self, app):
        """
        Initiates the cloudsml library with app's config
        """
        self.configuration.host = app.config.CLOUDSML_API_BASE_URL
        self.configuration.oauth2_url = app.config.CLOUDSML_API_OAUTH2_URL
        self.configuration.get_oauth2_token(**app.config.CLOUDSML_API_CREDENTIALS)

        api_client_instance = cloudsml.ApiClient(self.configuration)

        self.auth_api = cloudsml.AuthApi(api_client_instance)
        self.data_api = cloudsml.DataApi(api_client_instance)
        self.predictive_analytics_api = cloudsml.PredictiveAnalyticsApi(api_client_instance)
        self.scorings_api = cloudsml.ScoringsApi(api_client_instance)
        self.users_api = cloudsml.UsersApi(api_client_instance)
        self.workspaces_api = cloudsml.WorkspacesApi(api_client_instance)
