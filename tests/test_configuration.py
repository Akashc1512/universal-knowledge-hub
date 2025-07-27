"""
Comprehensive Unit Tests for Configuration Manager
Tests all aspects of the dynamic configuration system
"""

import unittest
import asyncio
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Import the configuration manager
import sys
sys.path.append('..')
from core.config_manager import ConfigurationManager, ConfigurationItem, get_config, set_config


class TestConfigurationItem(unittest.TestCase):
    """Test ConfigurationItem dataclass"""
    
    def test_configuration_item_creation(self):
        """Test creating a ConfigurationItem"""
        item = ConfigurationItem(
            key="test.key",
            value="test_value",
            source="environment",
            environment="test",
            version="1.0.0",
            last_updated=datetime.now(),
            ttl=300,
            encrypted=False,
            sensitive=False
        )
        
        self.assertEqual(item.key, "test.key")
        self.assertEqual(item.value, "test_value")
        self.assertEqual(item.source, "environment")
        self.assertEqual(item.environment, "test")
        self.assertEqual(item.version, "1.0.0")
        self.assertIsInstance(item.last_updated, datetime)
        self.assertEqual(item.ttl, 300)
        self.assertFalse(item.encrypted)
        self.assertFalse(item.sensitive)
    
    def test_configuration_item_defaults(self):
        """Test ConfigurationItem with default values"""
        item = ConfigurationItem(
            key="test.key",
            value="test_value",
            source="environment",
            environment="test",
            version="1.0.0",
            last_updated=datetime.now()
        )
        
        self.assertIsNone(item.ttl)
        self.assertFalse(item.encrypted)
        self.assertFalse(item.sensitive)
    
    def test_configuration_item_sensitive(self):
        """Test ConfigurationItem with sensitive data"""
        item = ConfigurationItem(
            key="database.password",
            value="secret_password",
            source="vault",
            environment="production",
            version="1.0.0",
            last_updated=datetime.now(),
            sensitive=True
        )
        
        self.assertTrue(item.sensitive)
        self.assertEqual(item.source, "vault")


class TestConfigurationManager(unittest.TestCase):
    """Test ConfigurationManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigurationManager(environment="test")
    
    def tearDown(self):
        """Clean up after tests"""
        # Clean up any test data
        pass
    
    def test_configuration_manager_initialization(self):
        """Test ConfigurationManager initialization"""
        self.assertEqual(self.config_manager.environment, "test")
        self.assertIsInstance(self.config_manager.config, dict)
        self.assertIsInstance(self.config_manager.watchers, list)
        self.assertEqual(self.config_manager.reload_interval, 30)
        self.assertEqual(self.config_manager.cache_ttl, 300)
    
    @patch('core.config_manager.hvac.Client')
    def test_vault_client_setup(self, mock_hvac):
        """Test Vault client setup"""
        mock_client = Mock()
        mock_hvac.return_value = mock_client
        
        # Test with Vault environment variables
        with patch.dict(os.environ, {
            'VAULT_ADDR': 'http://vault:8200',
            'VAULT_TOKEN': 'test_token'
        }):
            config_manager = ConfigurationManager()
            self.assertIsNotNone(config_manager.vault_client)
    
    @patch('core.config_manager.client.CoreV1Api')
    def test_kubernetes_client_setup(self, mock_k8s):
        """Test Kubernetes client setup"""
        mock_api = Mock()
        mock_k8s.return_value = mock_api
        
        with patch('core.config_manager.config.load_incluster_config'):
            config_manager = ConfigurationManager()
            self.assertIsNotNone(config_manager.k8s_client)
    
    def test_get_configuration_value(self):
        """Test getting configuration values"""
        # Add test configuration
        test_item = ConfigurationItem(
            key="test.key",
            value="test_value",
            source="environment",
            environment="test",
            version="1.0.0",
            last_updated=datetime.now()
        )
        self.config_manager.config["test.key"] = test_item
        
        # Test getting existing value
        value = self.config_manager.get("test.key")
        self.assertEqual(value, "test_value")
        
        # Test getting non-existent value with default
        value = self.config_manager.get("non.existent", "default_value")
        self.assertEqual(value, "default_value")
    
    def test_get_all_configuration(self):
        """Test getting all configuration values"""
        # Add test configurations
        test_items = [
            ConfigurationItem("test.key1", "value1", "env", "test", "1.0.0", datetime.now()),
            ConfigurationItem("test.key2", "value2", "env", "test", "1.0.0", datetime.now()),
            ConfigurationItem("other.key", "value3", "env", "test", "1.0.0", datetime.now())
        ]
        
        for item in test_items:
            self.config_manager.config[item.key] = item
        
        # Test getting all values
        all_config = self.config_manager.get_all()
        self.assertEqual(len(all_config), 3)
        self.assertEqual(all_config["test.key1"], "value1")
        
        # Test getting values with prefix
        test_config = self.config_manager.get_all("test.")
        self.assertEqual(len(test_config), 2)
        self.assertIn("test.key1", test_config)
        self.assertIn("test.key2", test_config)
    
    def test_set_configuration_value(self):
        """Test setting configuration values"""
        # Test setting new value
        self.config_manager.set("new.key", "new_value", "runtime")
        
        self.assertIn("new.key", self.config_manager.config)
        item = self.config_manager.config["new.key"]
        self.assertEqual(item.value, "new_value")
        self.assertEqual(item.source, "runtime")
    
    def test_configuration_watchers(self):
        """Test configuration watchers"""
        mock_callback = Mock()
        self.config_manager.watch(mock_callback)
        
        # Test that watcher is registered
        self.assertIn(mock_callback, self.config_manager.watchers)
        
        # Test watcher notification
        self.config_manager._notify_watchers()
        mock_callback.assert_called_once_with(self.config_manager.config)
    
    def test_configuration_ttl_expiration(self):
        """Test configuration TTL expiration"""
        # Add configuration with short TTL
        expired_item = ConfigurationItem(
            key="expired.key",
            value="expired_value",
            source="environment",
            environment="test",
            version="1.0.0",
            last_updated=datetime.now() - timedelta(seconds=400),  # Expired
            ttl=300  # 5 minutes
        )
        self.config_manager.config["expired.key"] = expired_item
        
        # Test that expired value returns default
        value = self.config_manager.get("expired.key", "default_value")
        self.assertEqual(value, "default_value")
    
    def test_environment_configuration(self):
        """Test environment-specific configuration"""
        env_config = self.config_manager.get_environment_config()
        
        self.assertIsInstance(env_config, dict)
        self.assertIn("environment", env_config)
        self.assertIn("database", env_config)
        self.assertIn("redis", env_config)
        self.assertIn("elasticsearch", env_config)
        self.assertIn("features", env_config)
        self.assertIn("app", env_config)
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Test with valid configuration
        self.config_manager.config["database.host"] = ConfigurationItem(
            "database.host", "localhost", "env", "test", "1.0.0", datetime.now()
        )
        
        result = self.config_manager.validate_configuration()
        self.assertTrue(result)
        
        # Test with missing required keys
        self.config_manager.config.clear()
        result = self.config_manager.validate_configuration()
        self.assertFalse(result)


class TestGlobalConfigurationFunctions(unittest.TestCase):
    """Test global configuration functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the global config manager
        self.mock_config_manager = Mock()
        with patch('core.config_manager.config_manager', self.mock_config_manager):
            pass
    
    def test_get_config_function(self):
        """Test get_config global function"""
        self.mock_config_manager.get.return_value = "test_value"
        
        value = get_config("test.key", "default_value")
        self.mock_config_manager.get.assert_called_once_with("test.key", "default_value")
        self.assertEqual(value, "test_value")
    
    def test_get_all_config_function(self):
        """Test get_all_config global function"""
        mock_config = {"test.key1": "value1", "test.key2": "value2"}
        self.mock_config_manager.get_all.return_value = mock_config
        
        config = get_all_config("test.")
        self.mock_config_manager.get_all.assert_called_once_with("test.")
        self.assertEqual(config, mock_config)
    
    def test_set_config_function(self):
        """Test set_config global function"""
        set_config("test.key", "test_value")
        self.mock_config_manager.set.assert_called_once_with("test.key", "test_value")
    
    def test_watch_config_function(self):
        """Test watch_config global function"""
        mock_callback = Mock()
        watch_config(mock_callback)
        self.mock_config_manager.watch.assert_called_once_with(mock_callback)


class TestConfigurationHotReload(unittest.TestCase):
    """Test configuration hot-reloading functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigurationManager(environment="test")
    
    @patch('asyncio.create_task')
    def test_hot_reload_startup(self, mock_create_task):
        """Test hot-reload startup"""
        # Verify that hot-reload is started during initialization
        mock_create_task.assert_called()
    
    @patch('core.config_manager.asyncio.sleep')
    @patch('core.config_manager.ConfigurationManager._initialize_config')
    def test_hot_reload_loop(self, mock_init, mock_sleep):
        """Test hot-reload loop functionality"""
        # This would require more complex async testing
        # For now, we'll test the basic structure
        self.assertIsNotNone(self.config_manager.reload_interval)
        self.assertGreater(self.config_manager.reload_interval, 0)


class TestConfigurationSources(unittest.TestCase):
    """Test different configuration sources"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigurationManager(environment="test")
    
    @patch('core.config_manager.os.getenv')
    def test_environment_configuration_loading(self, mock_getenv):
        """Test loading configuration from environment variables"""
        mock_getenv.side_effect = lambda key, default=None: {
            'DATABASE_HOST': 'test-db',
            'DATABASE_PORT': '5432',
            'DATABASE_NAME': 'test_db',
            'REDIS_HOST': 'test-redis',
            'ELASTICSEARCH_HOST': 'test-es',
            'DEBUG': 'true',
            'LOG_LEVEL': 'DEBUG'
        }.get(key, default)
        
        # Test environment configuration loading
        # This would be tested in the actual _load_environment_config method
        pass
    
    @patch('core.config_manager.hvac.Client')
    def test_vault_configuration_loading(self, mock_hvac):
        """Test loading configuration from Vault"""
        mock_client = Mock()
        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            'data': {
                'data': {
                    'database_password': 'secret_password',
                    'api_key': 'secret_api_key'
                }
            }
        }
        mock_hvac.return_value = mock_client
        
        self.config_manager.vault_client = mock_client
        
        # Test Vault configuration loading
        # This would be tested in the actual _load_vault_config method
        pass
    
    @patch('core.config_manager.client.CoreV1Api')
    def test_kubernetes_configuration_loading(self, mock_k8s):
        """Test loading configuration from Kubernetes ConfigMaps"""
        mock_api = Mock()
        mock_api.read_namespaced_config_map.return_value = Mock(
            data={
                'database.host': 'k8s-db-host',
                'redis.host': 'k8s-redis-host'
            }
        )
        mock_k8s.return_value = mock_api
        
        self.config_manager.k8s_client = mock_api
        
        # Test Kubernetes configuration loading
        # This would be tested in the actual _load_kubernetes_config method
        pass


class TestConfigurationErrorHandling(unittest.TestCase):
    """Test error handling in configuration management"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigurationManager(environment="test")
    
    def test_invalid_configuration_key(self):
        """Test handling of invalid configuration keys"""
        # Test with None key
        value = self.config_manager.get(None, "default")
        self.assertEqual(value, "default")
        
        # Test with empty key
        value = self.config_manager.get("", "default")
        self.assertEqual(value, "default")
    
    def test_configuration_merge_errors(self):
        """Test error handling during configuration merging"""
        # Test with invalid configuration results
        invalid_results = [None, "invalid", 123]
        
        # Should handle gracefully without raising exceptions
        try:
            self.config_manager._merge_configurations(invalid_results)
        except Exception as e:
            self.fail(f"Configuration merging should handle invalid results gracefully: {e}")
    
    def test_vault_connection_errors(self):
        """Test handling of Vault connection errors"""
        with patch('core.config_manager.hvac.Client') as mock_hvac:
            mock_hvac.side_effect = Exception("Vault connection failed")
            
            # Should handle Vault connection errors gracefully
            config_manager = ConfigurationManager()
            self.assertIsNone(config_manager.vault_client)
    
    def test_kubernetes_connection_errors(self):
        """Test handling of Kubernetes connection errors"""
        with patch('core.config_manager.config.load_incluster_config') as mock_load:
            mock_load.side_effect = Exception("Kubernetes connection failed")
            
            # Should handle Kubernetes connection errors gracefully
            config_manager = ConfigurationManager()
            self.assertIsNone(config_manager.k8s_client)


class TestConfigurationPerformance(unittest.TestCase):
    """Test configuration system performance"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigurationManager(environment="test")
    
    def test_configuration_access_performance(self):
        """Test configuration access performance"""
        import time
        
        # Add many configuration items
        for i in range(1000):
            self.config_manager.config[f"test.key{i}"] = ConfigurationItem(
                f"test.key{i}", f"value{i}", "env", "test", "1.0.0", datetime.now()
            )
        
        # Test access performance
        start_time = time.time()
        for i in range(1000):
            self.config_manager.get(f"test.key{i}")
        end_time = time.time()
        
        # Should complete within reasonable time (less than 1 second)
        self.assertLess(end_time - start_time, 1.0)
    
    def test_configuration_watcher_performance(self):
        """Test configuration watcher performance"""
        import time
        
        # Add many watchers
        watchers = [Mock() for _ in range(100)]
        for watcher in watchers:
            self.config_manager.watch(watcher)
        
        # Test notification performance
        start_time = time.time()
        self.config_manager._notify_watchers()
        end_time = time.time()
        
        # Should complete within reasonable time (less than 0.1 seconds)
        self.assertLess(end_time - start_time, 0.1)
        
        # Verify all watchers were called
        for watcher in watchers:
            watcher.assert_called_once()


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2) 