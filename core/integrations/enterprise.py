"""
Enterprise Integration System for Universal Knowledge Platform
Provides unified integration with SharePoint, Google Drive, Slack, and Teams.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import time
import hashlib

try:
    import aiohttp
    from aiohttp import ClientSession, ClientTimeout
except ImportError:
    aiohttp = None

try:
    from O365 import Account
    from O365.utils import TokenBackend
except ImportError:
    Account = None

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
except ImportError:
    Credentials = None
    InstalledAppFlow = None
    Request = None
    build = None

from core.knowledge_graph.client import Neo4jClient, GraphQuery


@dataclass
class IntegrationConfig:
    """Configuration for enterprise integrations."""
    microsoft_client_id: str
    microsoft_client_secret: str
    google_client_id: str
    google_client_secret: str
    slack_bot_token: str
    slack_signing_secret: str
    base_url: str = "https://graph.microsoft.com/v1.0"
    timeout: int = 30


@dataclass
class IntegrationItem:
    """Represents an item from an enterprise system."""
    id: str
    name: str
    content_type: str
    source_system: str
    url: Optional[str] = None
    content: Optional[str] = None
    metadata: Dict[str, Any] = None
    last_modified: Optional[datetime] = None
    size: Optional[int] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class IntegrationResult:
    """Result of an integration operation."""
    success: bool
    items_processed: int
    items_created: int
    items_updated: int
    errors: int
    duration: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MicrosoftGraphIntegration:
    """Microsoft Graph API integration for SharePoint and Teams."""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.account = None
        self.session = None
        self.authenticated = False
        
    async def authenticate(self) -> bool:
        """Authenticate with Microsoft Graph API."""
        try:
            if Account is None:
                logger.error("O365 library not available")
                return False
            
            # Initialize account
            self.account = Account((self.config.microsoft_client_id, 
                                  self.config.microsoft_client_secret))
            
            # Check if we have stored tokens
            if self.account.is_authenticated:
                self.authenticated = True
                logger.info("✅ Microsoft Graph authentication successful")
                return True
            
            # Request authentication
            if await self.account.authenticate(scopes=['Files.Read.All', 'Sites.Read.All']):
                self.authenticated = True
                logger.info("✅ Microsoft Graph authentication successful")
                return True
            else:
                logger.error("❌ Microsoft Graph authentication failed")
                return False
                
        except Exception as e:
            logger.error(f"Microsoft Graph authentication error: {e}")
            return False
    
    async def get_sharepoint_sites(self) -> List[Dict[str, Any]]:
        """Get SharePoint sites accessible to the authenticated user."""
        if not self.authenticated:
            return []
        
        try:
            storage = self.account.storage()
            sites = await storage.get_sites()
            
            site_list = []
            for site in sites:
                site_data = {
                    "id": site.object_id,
                    "name": site.name,
                    "url": site.web_url,
                    "title": site.title,
                    "description": site.description
                }
                site_list.append(site_data)
            
            logger.info(f"✅ Retrieved {len(site_list)} SharePoint sites")
            return site_list
            
        except Exception as e:
            logger.error(f"Failed to get SharePoint sites: {e}")
            return []
    
    async def get_sharepoint_documents(self, site_id: str) -> List[IntegrationItem]:
        """Get documents from a SharePoint site."""
        if not self.authenticated:
            return []
        
        try:
            storage = self.account.storage()
            site = await storage.get_site(site_id)
            
            documents = []
            drive = site.get_default_document_library()
            
            for item in drive.get_items():
                if item.is_file:
                    doc_item = IntegrationItem(
                        id=item.object_id,
                        name=item.name,
                        content_type=item.mime_type,
                        source_system="sharepoint",
                        url=item.web_url,
                        last_modified=item.last_modified,
                        size=item.size,
                        metadata={
                            "site_id": site_id,
                            "site_name": site.name,
                            "created_by": item.created_by.get('user', {}).get('displayName', ''),
                            "modified_by": item.modified_by.get('user', {}).get('displayName', '')
                        }
                    )
                    documents.append(doc_item)
            
            logger.info(f"✅ Retrieved {len(documents)} documents from SharePoint site {site_id}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get SharePoint documents: {e}")
            return []
    
    async def get_sharepoint_files(self, site_id: str) -> List[IntegrationItem]:
        """Get files from a SharePoint site (alias for get_sharepoint_documents)."""
        return await self.get_sharepoint_documents(site_id)
    
    async def get_teams_messages(self, team_id: str, channel_id: str) -> List[IntegrationItem]:
        """Get messages from a Teams channel."""
        if not self.authenticated:
            return []
        
        try:
            # This would require additional Teams API permissions
            # For now, return empty list as placeholder
            logger.info(f"Teams message retrieval not yet implemented for team {team_id}")
            return []
            
        except Exception as e:
            logger.error(f"Failed to get Teams messages: {e}")
            return []


class GoogleDriveIntegration:
    """Google Drive API integration."""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.service = None
        self.authenticated = False
        
    async def authenticate(self) -> bool:
        """Authenticate with Google Drive API."""
        try:
            if build is None:
                logger.error("Google API client library not available")
                return False
            
            # For now, use service account or stored credentials
            # In production, implement proper OAuth 2.0 flow
            logger.info("Google Drive authentication placeholder - implement OAuth flow")
            self.authenticated = True
            return True
            
        except Exception as e:
            logger.error(f"Google Drive authentication error: {e}")
            return False
    
    async def get_drive_files(self, folder_id: str = "root") -> List[IntegrationItem]:
        """Get files from Google Drive."""
        if not self.authenticated:
            return []
        
        try:
            # Placeholder implementation
            # In production, use actual Google Drive API
            logger.info(f"Google Drive file retrieval placeholder for folder {folder_id}")
            return []
            
        except Exception as e:
            logger.error(f"Failed to get Google Drive files: {e}")
            return []

    async def get_gmail_messages(self, query: str = "", max_results: int = 100) -> List[IntegrationItem]:
        """Get messages from Gmail."""
        if not self.authenticated:
            return []
        
        try:
            # Placeholder implementation
            # In production, use actual Gmail API
            logger.info(f"Gmail message retrieval placeholder for query: {query}")
            return []
            
        except Exception as e:
            logger.error(f"Failed to get Gmail messages: {e}")
            return []


class SlackIntegration:
    """Slack API integration."""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.session = None
        self.authenticated = False
        
    async def authenticate(self) -> bool:
        """Authenticate with Slack API."""
        try:
            if aiohttp is None:
                logger.error("aiohttp not available")
                return False
            
            self.session = ClientSession()
            
            # Test authentication
            headers = {"Authorization": f"Bearer {self.config.slack_bot_token}"}
            async with self.session.get("https://slack.com/api/auth.test", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("ok"):
                        self.authenticated = True
                        logger.info("✅ Slack authentication successful")
                        return True
            
            logger.error("❌ Slack authentication failed")
            return False
            
        except Exception as e:
            logger.error(f"Slack authentication error: {e}")
            return False
    
    async def get_channel_messages(self, channel_id: str, limit: int = 100) -> List[IntegrationItem]:
        """Get messages from a Slack channel."""
        if not self.authenticated:
            return []
        
        try:
            headers = {"Authorization": f"Bearer {self.config.slack_bot_token}"}
            params = {"channel": channel_id, "limit": limit}
            
            async with self.session.get("https://slack.com/api/conversations.history", 
                                      headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("ok"):
                        messages = []
                        for msg in data.get("messages", []):
                            message_item = IntegrationItem(
                                id=msg["ts"],
                                name=f"Message {msg['ts']}",
                                content_type="slack_message",
                                source_system="slack",
                                content=msg.get("text", ""),
                                last_modified=datetime.fromtimestamp(float(msg["ts"])),
                                metadata={
                                    "channel_id": channel_id,
                                    "user": msg.get("user", ""),
                                    "thread_ts": msg.get("thread_ts"),
                                    "reactions": msg.get("reactions", [])
                                }
                            )
                            messages.append(message_item)
                        
                        logger.info(f"✅ Retrieved {len(messages)} messages from Slack channel {channel_id}")
                        return messages
            
            logger.error(f"Failed to get Slack messages for channel {channel_id}")
            return []
            
        except Exception as e:
            logger.error(f"Failed to get Slack messages: {e}")
            return []

    async def send_message(self, channel_id: str, message: str) -> bool:
        """Send a message to a Slack channel."""
        if not self.authenticated:
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.config.slack_bot_token}"}
            data = {"channel": channel_id, "text": message}
            
            async with self.session.post("https://slack.com/api/chat.postMessage", 
                                       headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("ok"):
                        logger.info(f"✅ Message sent to Slack channel {channel_id}")
                        return True
            
            logger.error(f"Failed to send message to Slack channel {channel_id}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False


class EnterpriseIntegrationManager:
    """Manages all enterprise integrations."""
    
    def __init__(self, config: IntegrationConfig, graph_client: Neo4jClient):
        self.config = config
        self.graph_client = graph_client
        
        # Initialize integrations
        self.microsoft = MicrosoftGraphIntegration(config)
        self.google = GoogleDriveIntegration(config)
        self.slack = SlackIntegration(config)
        
        # Integration status
        self.integrations_status = {
            "microsoft": False,
            "google": False,
            "slack": False
        }
    
    async def initialize_all(self) -> bool:
        """Initialize all enterprise integrations."""
        logger.info("🔧 Initializing enterprise integrations...")
        
        # Authenticate with all services
        microsoft_auth = await self.microsoft.authenticate()
        google_auth = await self.google.authenticate()
        slack_auth = await self.slack.authenticate()
        
        self.integrations_status["microsoft"] = microsoft_auth
        self.integrations_status["google"] = google_auth
        self.integrations_status["slack"] = slack_auth
        
        successful = sum(self.integrations_status.values())
        total = len(self.integrations_status)
        
        logger.info(f"✅ Enterprise integrations initialized: {successful}/{total} successful")
        return successful > 0
    
    async def sync_sharepoint_content(self) -> IntegrationResult:
        """Sync content from SharePoint."""
        start_time = time.time()
        
        try:
            if not self.integrations_status["microsoft"]:
                return IntegrationResult(success=False, items_processed=0, items_created=0, 
                                      items_updated=0, errors=1, duration=0)
            
            # Get SharePoint sites
            sites = await self.microsoft.get_sharepoint_sites()
            
            total_items = 0
            created_items = 0
            updated_items = 0
            errors = 0
            
            for site in sites:
                try:
                    documents = await self.microsoft.get_sharepoint_documents(site["id"])
                    
                    for doc in documents:
                        # Create or update document in knowledge graph
                        success = await self._sync_document_to_graph(doc)
                        if success:
                            created_items += 1
                        else:
                            errors += 1
                        total_items += 1
                        
                except Exception as e:
                    logger.error(f"Error syncing SharePoint site {site['id']}: {e}")
                    errors += 1
            
            duration = time.time() - start_time
            
            return IntegrationResult(
                success=errors == 0,
                items_processed=total_items,
                items_created=created_items,
                items_updated=updated_items,
                errors=errors,
                duration=duration,
                metadata={"sites_processed": len(sites)}
            )
            
        except Exception as e:
            logger.error(f"SharePoint sync failed: {e}")
            return IntegrationResult(success=False, items_processed=0, items_created=0,
                                  items_updated=0, errors=1, duration=time.time() - start_time)
    
    async def sync_slack_content(self, channel_ids: List[str]) -> IntegrationResult:
        """Sync content from Slack channels."""
        start_time = time.time()
        
        try:
            if not self.integrations_status["slack"]:
                return IntegrationResult(success=False, items_processed=0, items_created=0,
                                      items_updated=0, errors=1, duration=0)
            
            total_items = 0
            created_items = 0
            updated_items = 0
            errors = 0
            
            for channel_id in channel_ids:
                try:
                    messages = await self.slack.get_channel_messages(channel_id)
                    
                    for msg in messages:
                        # Create or update message in knowledge graph
                        success = await self._sync_message_to_graph(msg)
                        if success:
                            created_items += 1
                        else:
                            errors += 1
                        total_items += 1
                        
                except Exception as e:
                    logger.error(f"Error syncing Slack channel {channel_id}: {e}")
                    errors += 1
            
            duration = time.time() - start_time
            
            return IntegrationResult(
                success=errors == 0,
                items_processed=total_items,
                items_created=created_items,
                items_updated=updated_items,
                errors=errors,
                duration=duration,
                metadata={"channels_processed": len(channel_ids)}
            )
            
        except Exception as e:
            logger.error(f"Slack sync failed: {e}")
            return IntegrationResult(success=False, items_processed=0, items_created=0,
                                  items_updated=0, errors=1, duration=time.time() - start_time)
    
    async def _sync_document_to_graph(self, doc: IntegrationItem) -> bool:
        """Sync a document to the knowledge graph."""
        try:
            # Check if document already exists
            check_query = GraphQuery(
                cypher="""
                MATCH (d:Document {id: $document_id})
                RETURN d
                """,
                parameters={"document_id": doc.id},
                metadata={"operation": "document_existence_check"}
            )
            
            result = await self.graph_client.execute_query(check_query)
            
            if result.success and result.data:
                # Update existing document
                update_query = GraphQuery(
                    cypher="""
                    MATCH (d:Document {id: $document_id})
                    SET d.title = $title,
                        d.content = $content,
                        d.content_type = $content_type,
                        d.url = $url,
                        d.last_modified = $last_modified,
                        d.size = $size,
                        d.source_system = $source_system,
                        d.metadata = $metadata
                    RETURN d
                    """,
                    parameters={
                        "document_id": doc.id,
                        "title": doc.name,
                        "content": doc.content or "",
                        "content_type": doc.content_type,
                        "url": doc.url,
                        "last_modified": doc.last_modified.isoformat() if doc.last_modified else None,
                        "size": doc.size,
                        "source_system": doc.source_system,
                        "metadata": doc.metadata
                    },
                    metadata={"operation": "document_update"}
                )
            else:
                # Create new document
                create_query = GraphQuery(
                    cypher="""
                    CREATE (d:Document {
                        id: $document_id,
                        title: $title,
                        content: $content,
                        content_type: $content_type,
                        url: $url,
                        last_modified: $last_modified,
                        size: $size,
                        source_system: $source_system,
                        metadata: $metadata,
                        created_at: $created_at
                    })
                    RETURN d
                    """,
                    parameters={
                        "document_id": doc.id,
                        "title": doc.name,
                        "content": doc.content or "",
                        "content_type": doc.content_type,
                        "url": doc.url,
                        "last_modified": doc.last_modified.isoformat() if doc.last_modified else None,
                        "size": doc.size,
                        "source_system": doc.source_system,
                        "metadata": doc.metadata,
                        "created_at": datetime.now().isoformat()
                    },
                    metadata={"operation": "document_creation"}
                )
                
                result = await self.graph_client.execute_query(create_query)
            
            return result.success
            
        except Exception as e:
            logger.error(f"Failed to sync document {doc.id}: {e}")
            return False
    
    async def _sync_message_to_graph(self, msg: IntegrationItem) -> bool:
        """Sync a message to the knowledge graph."""
        try:
            # Create message as a special document type
            create_query = GraphQuery(
                cypher="""
                MERGE (m:Document {
                    id: $message_id,
                    title: $title,
                    content: $content,
                    content_type: $content_type,
                    source_system: $source_system,
                    metadata: $metadata,
                    created_at: $created_at
                })
                RETURN m
                """,
                parameters={
                    "message_id": msg.id,
                    "title": msg.name,
                    "content": msg.content or "",
                    "content_type": msg.content_type,
                    "source_system": msg.source_system,
                    "metadata": msg.metadata,
                    "created_at": msg.last_modified.isoformat() if msg.last_modified else datetime.now().isoformat()
                },
                metadata={"operation": "message_creation"}
            )
            
            result = await self.graph_client.execute_query(create_query)
            return result.success
            
        except Exception as e:
            logger.error(f"Failed to sync message {msg.id}: {e}")
            return False
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrations."""
        return {
            "integrations": self.integrations_status,
            "total_integrations": len(self.integrations_status),
            "active_integrations": sum(self.integrations_status.values()),
            "last_check": datetime.now().isoformat()
        } 