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


class ChunkType(str, Enum):
    TEXT = "text"
    EXERCISE = "exercise"
    IMAGE = "image"
    TABLE = "table"
    OTHER = "other"
