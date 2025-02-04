from typing import List, Dict, Any, Optional
import sqlite3
import json
from datetime import datetime, timedelta
import tiktoken
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class Message:
    role: str
    content: str
    timestamp: str
    metadata: Dict[str, Any]
    essential: bool = True  # Flag to mark conversation-critical messages
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class MemoryManager:
    def __init__(
        self,
        max_tokens: int = 2000,
        db_path: str = "memory.db",
        model: str = "gpt-4-turbo-preview",
        summary_interval: int = 10  # Number of messages before triggering summarization
    ):
        self.max_tokens = max_tokens
        self.db_path = db_path
        self.encoding = tiktoken.encoding_for_model(model)
        self.summary_interval = summary_interval
        self.model = model
        
        # If database exists, drop it for fresh start (REMOVE THIS IN PRODUCTION)
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database with enhanced schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Drop existing tables if they exist
            c.execute('DROP TABLE IF EXISTS conversations')
            c.execute('DROP TABLE IF EXISTS summaries')
            
            # Create conversations table with essential flag
            c.execute('''
                CREATE TABLE conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    token_count INTEGER NOT NULL,
                    tool_call_id TEXT,
                    name TEXT,
                    tool_calls TEXT,
                    essential BOOLEAN NOT NULL DEFAULT 1,
                    summarized BOOLEAN NOT NULL DEFAULT 0
                )
            ''')
            
            # Create summaries table
            c.execute('''
                CREATE TABLE summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    start_message_id INTEGER,
                    end_message_id INTEGER,
                    token_count INTEGER NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def _count_tokens(self, text: str) -> int:
        """Count tokens in a piece of text"""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            return 0

    async def add_message(self, message: Message) -> None:
        """Add a new message to storage and trigger summarization if needed"""
        try:
            if not message.essential:
                logger.debug("Skipping non-essential message")
                return
                
            token_count = self._count_tokens(message.content)
            
            # Ensure tool_calls have type field
            tool_calls = message.tool_calls
            if tool_calls:
                for tool_call in tool_calls:
                    if "function" in tool_call and "type" not in tool_call:
                        tool_call["type"] = "function"
            
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute(
                '''INSERT INTO conversations 
                   (role, content, timestamp, metadata, token_count, tool_call_id, 
                    name, tool_calls, essential) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    message.role,
                    message.content,
                    message.timestamp,
                    json.dumps(message.metadata),
                    token_count,
                    message.tool_call_id,
                    message.name,
                    json.dumps(tool_calls) if tool_calls else None,
                    message.essential
                )
            )
            
            # Check if summarization is needed
            message_count = c.execute(
                'SELECT COUNT(*) FROM conversations WHERE summarized = 0'
            ).fetchone()[0]
            
            conn.commit()
            conn.close()
            
            if message_count >= self.summary_interval:
                await self.generate_summary()
                await self.prune_old_messages()
            
            logger.debug(f"Added message: {message.role}")
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            raise

    async def generate_summary(self) -> Optional[str]:
        """Generate a summary of recent unsummarized messages"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Get unsummarized messages
            rows = c.execute('''
                SELECT id, role, content, timestamp 
                FROM conversations 
                WHERE summarized = 0 AND essential = 1
                ORDER BY id ASC
            ''').fetchall()
            
            if not rows:
                conn.close()
                return None
            
            # Build conversation for summarization
            conversation = []
            for row in rows:
                conversation.append(f"{row[1]}: {row[2]}")
            
            # Create summary prompt
            prompt = (
                "Please provide a concise summary of the key points and insights from "
                "this conversation. Focus on the main topics, decisions, and important "
                "information that would be relevant for future context:\n\n"
                + "\n".join(conversation)
            )
            
            # In a real implementation, you would call your LLM here
            # For now, we'll create a placeholder summary
            summary = f"Summary of conversation from {rows[0][3]} to {rows[-1][3]}"
            
            # Store the summary
            token_count = self._count_tokens(summary)
            c.execute('''
                INSERT INTO summaries 
                (content, timestamp, start_message_id, end_message_id, token_count)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                summary,
                datetime.now().isoformat(),
                rows[0][0],
                rows[-1][0],
                token_count
            ))
            
            # Mark messages as summarized
            c.execute('''
                UPDATE conversations 
                SET summarized = 1 
                WHERE id BETWEEN ? AND ?
            ''', (rows[0][0], rows[-1][0]))
            
            conn.commit()
            conn.close()
            
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return None

    async def prune_old_messages(self, days_to_keep: int = 7):
        """Archive or delete old messages that have been summarized"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Delete old, non-essential messages that have been summarized
            c.execute('''
                DELETE FROM conversations 
                WHERE timestamp < ? 
                AND essential = 0 
                AND summarized = 1
            ''', (cutoff_date,))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error pruning old messages: {e}")

    def get_recent_context(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent conversation context including summaries"""
        try:
            max_tokens = max_tokens or self.max_tokens
            messages = []
            total_tokens = 0
            
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # First, get recent summaries
            summaries = c.execute('''
                SELECT content, timestamp, token_count 
                FROM summaries 
                ORDER BY id DESC 
                LIMIT 2
            ''').fetchall()
            
            # Add summaries as system messages
            for summary in summaries:
                if total_tokens + summary[2] <= max_tokens:
                    messages.append({
                        "role": "system",
                        "content": f"Previous conversation summary: {summary[0]}"
                    })
                    total_tokens += summary[2]
            
            # Then get recent messages that haven't been summarized
            rows = c.execute('''
                SELECT role, content, timestamp, metadata, token_count, 
                       tool_call_id, name, tool_calls 
                FROM conversations 
                WHERE summarized = 0 AND essential = 1
                ORDER BY id ASC
            ''').fetchall()
            
            for row in rows:
                if total_tokens + row[4] > max_tokens:
                    break
                
                message = {
                    "role": row[0],
                    "content": row[1],
                }
                
                # Add tool-related fields if present
                if row[7]:  # tool_calls
                    try:
                        tool_calls = json.loads(row[7])
                        if tool_calls:
                            for tool_call in tool_calls:
                                if "function" in tool_call and "type" not in tool_call:
                                    tool_call["type"] = "function"
                            message["tool_calls"] = tool_calls
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding tool_calls JSON")
                
                if row[5]:  # tool_call_id
                    message["tool_call_id"] = row[5]
                if row[6]:  # name
                    message["name"] = row[6]
                
                messages.append(message)
                total_tokens += row[4]
            
            conn.close()
            return messages
        except Exception as e:
            logger.error(f"Error getting recent context: {e}")
            return []

    def search_memory(self, query: str, include_summaries: bool = True, limit: int = 5) -> List[Dict[str, Any]]:
        """Search through memory including both messages and summaries"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            results = []
            
            # Search conversations
            message_results = c.execute('''
                SELECT role, content, timestamp, metadata 
                FROM conversations 
                WHERE content LIKE ? AND essential = 1
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (f'%{query}%', limit)).fetchall()
            
            for row in message_results:
                results.append({
                    "type": "message",
                    "role": row[0],
                    "content": row[1],
                    "timestamp": row[2],
                    "metadata": json.loads(row[3]) if row[3] else {}
                })
            
            # Search summaries if requested
            if include_summaries:
                summary_results = c.execute('''
                    SELECT content, timestamp
                    FROM summaries 
                    WHERE content LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (f'%{query}%', limit)).fetchall()
                
                for row in summary_results:
                    results.append({
                        "type": "summary",
                        "content": row[0],
                        "timestamp": row[1]
                    })
            
            conn.close()
            return sorted(results, key=lambda x: x["timestamp"], reverse=True)[:limit]
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return []

    def clear_memory(self):
        """Clear all data from memory"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('DELETE FROM conversations')
            c.execute('DELETE FROM summaries')
            conn.commit()
            conn.close()
            logger.info("Memory cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
            raise

    def get_memory_size(self) -> int:
        """Get total token count from both conversations and summaries"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Get total from conversations
            conv_tokens = c.execute(
                'SELECT SUM(token_count) FROM conversations'
            ).fetchone()[0] or 0
            
            # Get total from summaries
            summary_tokens = c.execute(
                'SELECT SUM(token_count) FROM summaries'
            ).fetchone()[0] or 0
            
            conn.close()
            return conv_tokens + summary_tokens
        except Exception as e:
            logger.error(f"Error getting memory size: {e}")
            return 0
