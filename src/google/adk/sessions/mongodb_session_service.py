from __future__ import annotations

import copy
from datetime import datetime
from datetime import timezone
import logging
from typing import Any
from typing import Optional
import uuid

# Removed SQLAlchemy specific imports
# from google.genai import types # Assuming types.Action is still defined and Pydantic based
from pymongo import MongoClient
from pymongo.errors import (
    ConnectionFailure,
    InvalidURI,
    DuplicateKeyError,
)  # Specific MongoDB errors
from typing_extensions import override
from tzlocal import get_localzone

from . import _session_util
from ..events.event import Event
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListSessionsResponse
from .session import Session
from .state import State

logger = logging.getLogger("google_adk." + __name__)

DEFAULT_MAX_KEY_LENGTH = 128
DEFAULT_MAX_VARCHAR_LENGTH = 256  # Not directly used in PyMongo for schema definition


class MongoDBSessionService(BaseSessionService):
    """A session service that uses a MongoDB database for storage."""

    def __init__(self, db_url: str, **kwargs: Any):
        """Initializes the database session service with a database URL."""
        try:
            # PyMongo connects lazily. InvalidURI is checked immediately.
            self.client = MongoClient(db_url, **kwargs)

            # The database name is usually part of the db_url (e.g., mongodb://localhost:27017/my_database)
            # If not provided in the URL, PyMongo defaults to 'test'.
            self.db = self.client.get_default_database()

        except InvalidURI as e:
            raise ValueError(
                f"Invalid MongoDB connection URI format for '{db_url}': {e}"
            ) from e
        except ConnectionFailure as e:
            # This might catch connection issues on first operation if not already apparent from URI.
            raise ValueError(f"Failed to connect to MongoDB at '{db_url}': {e}") from e
        except Exception as e:
            # Catch any other unexpected errors during client creation
            raise ValueError(
                f"An unexpected error occurred initializing MongoDB client for '{db_url}': {e}"
            ) from e

        # Define collection objects
        self.sessions_collection = self.db["sessions"]
        self.events_collection = self.db["events"]
        self.app_states_collection = self.db["app_states"]
        self.user_states_collection = self.db["user_states"]

        # Create indexes to ensure uniqueness and optimize queries, mimicking relational behavior.
        # Sessions: Unique compound index on app_name, user_id, id (acts as primary key)
        # This is implicitly handled if `_id` is chosen as the compound key as done below.
        # If using `ObjectId` for `_id` for sessions:
        # self.sessions_collection.create_index([
        #    ("app_name", 1), ("user_id", 1), ("id", 1)
        # ], unique=True)

        # For Events:
        # Mimics foreign key for session lookups and ensures unique event ID within session.
        # Note: If Event.id is used as _id, we only need to ensure the _id field's uniqueness.
        # But since `session_id` and `timestamp` are frequently queried, add indexes on those.
        # (app_name, user_id, session_id, id) is unique because `id` is primary key for event
        # and `(app_name, user_id, session_id)` represents the session.
        self.events_collection.create_index(
            [
                ("session_id", 1),
                ("timestamp", 1),
            ]  # For sorting by timestamp in get_session
        )
        # An index on `id` (which is _id for events documents) is automatically present.

        self.local_timezone = get_localzone()
        logger.info(
            f"MongoDB session service initialized. Connected to DB: {self.db.name} at {db_url}"
        )

    @override
    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        # Generate session_id if not provided
        session_id = session_id if session_id is not None else str(uuid.uuid4())
        current_utc_time = datetime.now(timezone.utc)

        # Define the composite _id for the session document
        session_doc_id = {"app_name": app_name, "user_id": user_id, "id": session_id}

        # Fetch app and user states from storage (by their respective _id formats)
        storage_app_state_doc = self.app_states_collection.find_one({"_id": app_name})
        storage_user_state_doc = self.user_states_collection.find_one(
            {"_id": {"app_name": app_name, "user_id": user_id}}
        )

        # Initialize states from fetched documents or empty dicts
        app_state = storage_app_state_doc["state"] if storage_app_state_doc else {}
        user_state = storage_user_state_doc["state"] if storage_user_state_doc else {}

        # Extract state deltas from the incoming `state` parameter
        app_state_delta, user_state_delta, session_state = _extract_state_delta(state)

        # Apply deltas to current (fetched) states
        app_state.update(app_state_delta)
        user_state.update(user_state_delta)

        # Update/Upsert app state document
        self.app_states_collection.update_one(
            {"_id": app_name},
            {"$set": {"state": app_state, "update_time": current_utc_time}},
            upsert=True,
        )
        # Update/Upsert user state document
        self.user_states_collection.update_one(
            {"_id": {"app_name": app_name, "user_id": user_id}},
            {"$set": {"state": user_state, "update_time": current_utc_time}},
            upsert=True,
        )

        # Prepare session document
        session_document = {
            "_id": session_doc_id,  # Use composite _id
            "app_name": app_name,
            "user_id": user_id,
            "id": session_id,
            "state": session_state,  # This is the session-specific state
            "create_time": current_utc_time,
            "update_time": current_utc_time,  # Initial update time is create time
        }

        try:
            # Insert the new session document. This will raise DuplicateKeyError if _id already exists.
            self.sessions_collection.insert_one(session_document)
        except DuplicateKeyError as e:
            raise ValueError(
                f"Session with id '{session_id}' for user '{user_id}' in app '{app_name}' already exists."
            ) from e

        # Merge all state parts for the Session object returned to the user
        merged_full_state = _merge_state(app_state, user_state, session_state)

        # Convert update_time (UTC datetime) to Unix timestamp
        last_update_time_ts = (
            session_document["update_time"].replace(tzinfo=timezone.utc).timestamp()
        )

        session = Session(
            app_name=app_name,
            user_id=user_id,
            id=session_id,
            state=merged_full_state,
            last_update_time=last_update_time_ts,
        )
        return session

    @override
    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        # Define _id for session lookup
        session_doc_id = {"app_name": app_name, "user_id": user_id, "id": session_id}

        session_document = self.sessions_collection.find_one(session_doc_id)
        if session_document is None:
            return None

        # Fetch app, user, and session states from storage
        storage_app_state_doc = self.app_states_collection.find_one({"_id": app_name})
        storage_user_state_doc = self.user_states_collection.find_one(
            {"_id": {"app_name": app_name, "user_id": user_id}}
        )

        app_state = storage_app_state_doc["state"] if storage_app_state_doc else {}
        user_state = storage_user_state_doc["state"] if storage_user_state_doc else {}
        session_state = session_document.get("state", {})  # Use .get() for safety

        # Merge states for the returned Session object
        merged_full_state = _merge_state(app_state, user_state, session_state)

        # Build query for events based on config
        event_query_filter: dict[str, Any] = {
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session_id,
        }
        if config and config.after_timestamp:
            # Convert Unix timestamp to UTC datetime for comparison in MongoDB
            event_query_filter["timestamp"] = {
                "$gte": datetime.fromtimestamp(config.after_timestamp, tz=timezone.utc)
            }

        # Fetch events, sorted by timestamp (ascending for chronological order)
        event_cursor = self.events_collection.find(event_query_filter).sort(
            "timestamp", 1
        )

        if config and config.num_recent_events:
            # To match "num_recent_events" logic, we need the *last* N events.
            # So, sort descending, limit, then reverse to get oldest of N first.
            event_cursor = (
                self.events_collection.find(event_query_filter)
                .sort("timestamp", -1)
                .limit(config.num_recent_events)
            )
            storage_events_docs = list(event_cursor)
            events = [self._to_event(e) for e in reversed(storage_events_docs)]
        else:
            # If no limit, get all events in chronological order (sort ascending)
            storage_events_docs = list(event_cursor)
            events = [self._to_event(e) for e in storage_events_docs]

        # Convert session document's update_time to Unix timestamp
        last_update_time_ts = (
            session_document["update_time"].replace(tzinfo=timezone.utc).timestamp()
        )

        session = Session(
            app_name=app_name,
            user_id=user_id,
            id=session_id,
            state=merged_full_state,
            last_update_time=last_update_time_ts,
        )
        session.events = events
        return session

    @override
    async def list_sessions(
        self, *, app_name: str, user_id: str
    ) -> ListSessionsResponse:
        # Find all sessions matching app_name and user_id using the fields directly
        results_cursor = self.sessions_collection.find(
            {"app_name": app_name, "user_id": user_id}
        )
        sessions = []
        for session_doc in results_cursor:
            # Convert update_time (UTC datetime) to Unix timestamp
            last_update_time_ts = (
                session_doc["update_time"].replace(tzinfo=timezone.utc).timestamp()
            )

            session = Session(
                app_name=session_doc["app_name"],
                user_id=session_doc["user_id"],
                id=session_doc["id"],  # Access the 'id' field, not '_id'
                state={},  # As per original SQLAlchemy implementation for list_sessions
                last_update_time=last_update_time_ts,
            )
            sessions.append(session)
        return ListSessionsResponse(sessions=sessions)

    @override
    async def delete_session(
        self, app_name: str, user_id: str, session_id: str
    ) -> None:
        session_doc_id = {"app_name": app_name, "user_id": user_id, "id": session_id}

        # Delete the session document itself
        self.sessions_collection.delete_one(session_doc_id)

        # Delete associated event documents, mimicking CASCADE behavior
        self.events_collection.delete_many(
            {"app_name": app_name, "user_id": user_id, "session_id": session_id}
        )

    @override
    async def append_event(self, session: Session, event: Event) -> Event:
        logger.info(f"Append event: {event.id} to session {session.id}")

        if event.partial:
            return event

        session_doc_id = {
            "app_name": session.app_name,
            "user_id": session.user_id,
            "id": session.id,
        }

        # 1. Fetch current stored session to check for staleness and get its current state
        storage_session_doc = self.sessions_collection.find_one(session_doc_id)
        if not storage_session_doc:
            raise ValueError(f"Session '{session_doc_id}' not found.")

        storage_session_update_time_ts = (
            storage_session_doc["update_time"].replace(tzinfo=timezone.utc).timestamp()
        )

        # Check for stale session against the client's last_update_time
        if storage_session_update_time_ts > session.last_update_time:
            raise ValueError(
                "The last_update_time provided in the session object"
                f" {datetime.fromtimestamp(session.last_update_time, tz=timezone.utc):'%Y-%m-%d %H:%M:%S'} is"
                " earlier than the update_time in the storage_session"
                f" {datetime.fromtimestamp(storage_session_update_time_ts, tz=timezone.utc):'%Y-%m-%d %H:%M:%S'}."
                " Please check if it is a stale session."
            )

        # 2. Fetch current app and user states
        storage_app_state_doc = self.app_states_collection.find_one(
            {"_id": session.app_name}
        )
        storage_user_state_doc = self.user_states_collection.find_one(
            {"_id": {"app_name": session.app_name, "user_id": session.user_id}}
        )

        app_state = storage_app_state_doc["state"] if storage_app_state_doc else {}
        user_state = storage_user_state_doc["state"] if storage_user_state_doc else {}
        session_state = storage_session_doc.get("state", {})

        # 3. Extract state deltas from the event's actions
        app_state_delta = {}
        user_state_delta = {}
        session_state_delta = {}
        if event.actions and event.actions.state_delta:
            app_state_delta, user_state_delta, session_state_delta = (
                _extract_state_delta(event.actions.state_delta)
            )

        # 4. Prepare updates and update states in MongoDB
        update_time_for_db = datetime.now(timezone.utc)

        # Update app state if there are changes
        if app_state_delta:
            app_state.update(app_state_delta)
            self.app_states_collection.update_one(
                {"_id": session.app_name},
                {"$set": {"state": app_state, "update_time": update_time_for_db}},
                upsert=True,  # Upsert for safety if it somehow didn't exist unexpectedly
            )

        # Update user state if there are changes
        if user_state_delta:
            user_state.update(user_state_delta)
            self.user_states_collection.update_one(
                {"_id": {"app_name": session.app_name, "user_id": session.user_id}},
                {"$set": {"state": user_state, "update_time": update_time_for_db}},
                upsert=True,
            )

        # Update session state and its update_time
        # Even if session_state_delta is empty, we must update the session's update_time
        # to reflect the event append, aligning with `onupdate=func.now()` from SQLAlchemy.
        if session_state_delta:
            session_state.update(session_state_delta)
            self.sessions_collection.update_one(
                session_doc_id,
                {"$set": {"state": session_state, "update_time": update_time_for_db}},
            )

        # 5. Create and insert the event document
        event_document = self._from_event(session, event)
        self.events_collection.insert_one(event_document)

        # 6. Update the in-memory session object's last_update_time
        if session_state_delta:
            session.last_update_time = update_time_for_db.replace(
                tzinfo=timezone.utc
            ).timestamp()

        # Also call super method to handle in-memory event appending if any
        await super().append_event(session=session, event=event)
        return event

    def _from_event(self, session: Session, event: Event) -> dict[str, Any]:
        """Converts an Event object to a MongoDB document dictionary."""
        doc = {
            "_id": event.id,  # Using event.id as _id for event documents
            "id": event.id,
            "invocation_id": event.invocation_id,
            "author": event.author,
            "branch": event.branch,
            # Convert Pydantic actions model to a dict (JSON serializable)
            "actions": (
                event.actions.model_dump(exclude_none=True, mode="json")
                if event.actions
                else None
            ),
            "session_id": session.id,
            "app_name": session.app_name,
            "user_id": session.user_id,
            # Store Python datetime objects directly (MongoDB handles UTC BSON date)
            "timestamp": datetime.fromtimestamp(event.timestamp, tz=timezone.utc),
            # Store set[str] as a list directly in MongoDB.
            "long_running_tool_ids": (
                list(event.long_running_tool_ids) if event.long_running_tool_ids else []
            ),
            "partial": event.partial,
            "turn_complete": event.turn_complete,
            "error_code": event.error_code,
            "error_message": event.error_message,
            "interrupted": event.interrupted,
        }
        if event.content:
            doc["content"] = event.content.model_dump(exclude_none=True, mode="json")
        if event.grounding_metadata:
            doc["grounding_metadata"] = event.grounding_metadata.model_dump(
                exclude_none=True, mode="json"
            )

        return doc

    def _to_event(self, doc: dict[str, Any]) -> Event:
        """Converts a MongoDB document dictionary back to an Event object."""
        # Ensure MongoDB datetime object (UTC) is converted to Unix timestamp correctly
        timestamp_from_db = (
            doc["timestamp"].replace(tzinfo=timezone.utc).timestamp()
            if isinstance(doc["timestamp"], datetime)
            else doc["timestamp"]
        )

        return Event(
            id=doc["id"],
            invocation_id=doc["invocation_id"],
            author=doc["author"],
            branch=doc.get("branch"),  # Safely get nullable fields
            # 'actions' is stored as a dict, return as dict. Assume Event constructor handles this.
            # If types.Action model is required, it must be reconstructed via `types.Action.model_validate(doc.get("actions"))`
            actions=doc.get("actions"),
            timestamp=timestamp_from_db,
            content=_session_util.decode_content(doc.get("content")),
            # Convert list back to set for event object
            long_running_tool_ids=set(doc.get("long_running_tool_ids", [])),
            partial=doc.get("partial"),
            turn_complete=doc.get("turn_complete"),
            error_code=doc.get("error_code"),
            error_message=doc.get("error_message"),
            interrupted=doc.get("interrupted"),
            grounding_metadata=_session_util.decode_grounding_metadata(
                doc.get("grounding_metadata")
            ),
        )


# Helper functions that operate on dictionaries remain unchanged
def _extract_state_delta(
    state: Optional[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    app_state_delta = {}
    user_state_delta = {}
    session_state_delta = {}
    if state:
        for key, value in state.items():
            if key.startswith(State.APP_PREFIX):
                app_state_delta[key.removeprefix(State.APP_PREFIX)] = value
            elif key.startswith(State.USER_PREFIX):
                user_state_delta[key.removeprefix(State.USER_PREFIX)] = value
            elif not key.startswith(State.TEMP_PREFIX):
                session_state_delta[key] = value
            # end if
        # end for
    # end if
    return app_state_delta, user_state_delta, session_state_delta


def _merge_state(
    app_state: dict[str, Any], user_state: dict[str, Any], session_state: dict[str, Any]
) -> dict[str, Any]:
    # Merge states for response
    merged_state = copy.deepcopy(session_state)
    # App state keys are prefixed
    for key, value in app_state.items():
        merged_state[State.APP_PREFIX + key] = value
    # User state keys are prefixed
    for key, value in user_state.items():
        merged_state[State.USER_PREFIX + key] = value
    return merged_state
