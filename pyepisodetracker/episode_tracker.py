import numpy as np
import cv2
from peasyprofiller.profiller import profiller as pprof


class Vector2():
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


    def manhat_length(self):
        return abs(self.x) + abs(self.x)


    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)
    

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)


    def __mul__(self, other):
        return Vector2(self.x * other, self.y * other)


    def __div__(self, other):
        return Vector2(self.x / other, self.y / other)

    
    def __eq__(self, value: object) -> bool:
        return self.x == value.x and self.y == value.y


class Object():
    def __init__(self, position: Vector2, size: Vector2, color: int):
        self.position = position
        self.size = size
        self.color = color
        self.category = None


CAT_SIZE_TOLERANCE = 2


class ObjectCategory():
    def __init__(self, size: Vector2, color: int, indicator_color: np.array) -> None:
        self.size = size
        self.color = color
        self.indicator_color = indicator_color
    
    
    def belongs(self, object: Object) -> bool:
        return (self.size - object.size).manhat_length() <= CAT_SIZE_TOLERANCE
    

    @staticmethod
    def from_obj(object: Object):
        return ObjectCategory(object.size, object.color, np.random.randint(256, size=(3)))



class ObjectTransition():
    def __init__(self, object_from: Object | None, object_to: Object | None) -> None:
        self.object_from = object_from
        self.object_to = object_to


class Event():
    def __init__(self, obj_category: str, category: str, current_pos: Vector2) -> None:
        self.obj_category = obj_category
        self.category = category
        self.current_pos = current_pos


class AppearanceEvent(Event):
    def __init__(self, obj_category: str, current_pos: Vector2) -> None:
        super().__init__(obj_category, "APPEARANCE", current_pos)


class MovementEvent(Event):
    def __init__(self, obj_category: str, initial_pos: Vector2, initial_timestamp: int, current_pos: Vector2):
        super().__init__(obj_category, "MOVEMENT", current_pos)
        self.initial_pos = initial_pos
        self.initial_timestamp = initial_timestamp


    def get_vel(self, current_timestep: int):
        return (self.current_pos - self.initial_pos) / float(current_timestep - self.initial_timestamp)


    def get_state(self, current_timestep: int):
        """
        Given a timestep, returns the current position of this event and its average velocity
        """
        return self.current_pos, self.get_vel(current_timestep)


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

        # Render mode

        for obj in objs:
            top_left = obj.position
            bottom_right = obj.position + obj.size
            cv2.rectangle(separated_bg, (top_left.x, top_left.y), (bottom_right.x, bottom_right.y), tuple(map(int, obj.category.indicator_color)), 2)

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
        
        result = cv2.bitwise_and(data, data, mask=mask)

        pprof.stop("ET_BGSEP")

        return result

    
    def object_identification(self, data: np.array) -> list[Object]:
        pprof.start("ET_OBJID")

        r, _, _ = cv2.split(data)
        peaks = np.unique(r)

        objects = []
        for i, peak in enumerate(peaks):
            if peak == 0: # We're not considering black
                continue 
            
            peak = np.array(peak)
            mask = cv2.inRange(r, peak, peak)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for j, contour in enumerate(contours):
                bbox = cv2.boundingRect(contour)
                if bbox[2] <= 1 or bbox[3] <= 1 or bbox[0] == 0 or bbox[1] == 0 or bbox[0] + bbox[2] >= r.shape[1] or bbox[1] + bbox[3] >= r.shape[0]:
                    continue

                objects.append(Object(Vector2(bbox[0], bbox[1]), Vector2(bbox[2], bbox[3]), r))

        pprof.stop("ET_OBJID")

        return objects

    
    def object_categorization(self, data: list[Object]) -> list[Object]:
        pprof.start("ET_OBJCAT")

        for obj in data:
            for cat in self.object_categories:
                if cat.belongs(obj):
                    obj.category = cat
                    break
            if obj.category is None:
                new_cat = ObjectCategory.from_obj(obj)
                self.object_categories.append(new_cat)
                obj.category = new_cat

        pprof.stop("ET_OBJCAT")

        return data


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