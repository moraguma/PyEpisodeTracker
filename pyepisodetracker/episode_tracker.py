import numpy as np
import cv2
from peasyprofiller.profiller import profiller as pprof


class Vector2():
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
    

    def __sub__(self, other):
        self.x -= other.x
        self.y -= other.y
    

    def __add__(self, other):
        self.x += other.x
        self.y += other.y
    

    def __mul__(self, other):
        self.x *= other
        self.y *= other
    

    def __div__(self, other):
        self.x /= other
        self.y /= other

    
    def __eq__(self, value: object) -> bool:
        return self.x == value.x and self.y == value.y


class Object():
    def __init__(self, position: Vector2, size: Vector2, color: np.array):
        self.position = position
        self.size = size
        self.color = color
        self.category = None


class ObjectCategory():
    def __init__(self, size: Vector2, color: np.array) -> None:
        self.size = size
        self.color = color
    
    
    def belongs(self, object: Object) -> bool:
        return self.size == object.size


class ObjectTransition():
    def __init__(self, object_from: Object | None, object_to: Object | None) -> None:
        self.object_from = object_from
        self.object_to = object_to


class Event():
    def __init__(self, obj_category: str, category: str, current_pos: Vector2) -> None:
        self.obj_category = obj_category
        self.category = category
        self.current_pos = current_pos


class AppearanceEvent():
    def __init__(self, obj_category: str, current_pos: Vector2) -> None:
        super().__init__(obj_category, "APPEARANCE", current_pos)


class MovementEvent():
    def __init__(self, obj_category: str, initial_pos: Vector2, initial_timestamp: int, current_pos: Vector2):
        super().__init__(obj_category, "MOVEMENT", current_pos)
        self.initial_pos = initial_pos
        self.initial_timestamp = initial_timestamp
    

    def get_state(self, current_timestep: int):
        """
        Given a timestep, returns the current position of this event and its average velocity
        """
        return self.current_pos, (self.current_pos - self.initial_pos) / float(current_timestep - self.initial_timestamp)


class EpisodeTracker():
    def __init__(self, background_colors: np.array) -> None:
        self.background_colors = background_colors

        self.object_categories: list[ObjectCategory] = []
        self.tracked_objects: list[Object] = []
        self.tracked_events: list[Event] = []
        self.timestep: int = 0


    def process_frame(self, data: np.array) -> list[Event]:
        separated_bg = self.background_separation(data)
        objs = self.object_identification(separated_bg)
        objs = self.object_categorization(objs)
        transitions = self.object_tracking(objs)
        events = self.event_tracking(transitions)

        self.timestep += 1

        return separated_bg, objs, transitions, events


    def background_separation(self, data: np.array, threshold=1) -> np.array:
        pprof.start("ET_BGSEP")

        mask = np.ones(data.shape[:2], dtype=np.uint8) * 255

        for color in self.background_colors:
            lower_bound = np.array(color) - threshold
            upper_bound = np.array(color) + threshold

            lower_bound = np.clip(lower_bound, 0, 255)
            upper_bound = np.clip(upper_bound, 0, 255)

            color_mask = cv2.inRange(data, lower_bound, upper_bound)

            mask[color_mask > 0] = 0

        pprof.stop("ET_BGSEP")

        return mask

    
    def object_identification(self, data: np.array) -> list[Object]:
        pprof.start("ET_OBJID")
        pprof.stop("ET_OBJID")

    
    def object_categorization(self, data: list[Object]) -> list[Object]:
        pprof.start("ET_OBJCAT")
        pprof.stop("ET_OBJCAT")

    
    def object_tracking(self, data: list[Object]) -> list[ObjectTransition]:
        pprof.start("ET_OBJTRK")
        pprof.stop("ET_OBJTRK")


    def event_tracking(self, data: list[ObjectTransition]) -> list[Event]:
        pprof.start("ET_EVTRK")
        pprof.stop("ET_EVTRK")

    
    def finish_episode(self) -> None:
        self.tracked_objects = []
        self.tracked_events = []
        self.timestep = 0