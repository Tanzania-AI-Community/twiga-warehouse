from enum import Enum


class SubjectClassStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class ResourceType(str, Enum):
    TEXTBOOK = "textbook"
    CURRICULUM = "curriculum"
    DOCUMENT = "document"


class Role(str, Enum):
    ADMIN = "admin"
    TEACHER = "teacher"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class GradeLevel(str, Enum):
    P1 = "p1"
    P2 = "p2"
    P3 = "p3"
    P4 = "p4"
    P5 = "p5"
    P6 = "p6"
    OS1 = "os1"
    OS2 = "os2"
    OS3 = "os3"
    OS4 = "os4"
    AS1 = "as1"
    AS2 = "as2"

    @property
    def display_format(self) -> str:
        """Returns a nicely formatted string for display"""
        grade_display = {
            "p1": "Standard 1",
            "p2": "Standard 2",
            "p3": "Standard 3",
            "p4": "Standard 4",
            "p5": "Standard 5",
            "p6": "Standard 6",
            "os1": "Form 1",
            "os2": "Form 2",
            "os3": "Form 3",
            "os4": "Form 4",
            "as1": "Form 5",
            "as2": "Form 6",
        }
        return grade_display[self]


class OnboardingState(str, Enum):
    NEW = "new"
    PERSONAL_INFO_SUBMITTED = "personal_info_submitted"
    COMPLETED = "completed"


class UserState(str, Enum):
    BLOCKED = "blocked"
    RATE_LIMITED = "rate_limited"
    NEW = "new"
    ONBOARDING = "onboarding"
    ACTIVE = "active"


class SubjectName(str, Enum):
    GEOGRAPHY = "geography"

    @property
    def display_format(self) -> str:
        emoji_map = {"geography": "🌎"}
        emoji = emoji_map.get(self, "")
        return f"{self.capitalize()} {emoji}"


class ChunkType(str, Enum):
    TEXT = "text"
    EXERCISE = "exercise"
    IMAGE = "image"
    TABLE = "table"
    OTHER = "other"
