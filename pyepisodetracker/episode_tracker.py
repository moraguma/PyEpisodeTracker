import numpy as np
import copy
import cv2
from typing import Dict
from peasyprofiller.profiller import profiller as pprof


ORIENTATION_DIF_TOLERANCE = 0.2
AREA_DIF_TOLERANCE = 16

SPEED_DIF_TOLERANCE = 6
MAX_SPEED = 50


class Vector2():
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


    def area(self):
        return abs(self.x * self.y)


    def manhat_length(self):
        return abs(self.x) + abs(self.y)


    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)
    

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)


    def __mul__(self, other):
        return Vector2(self.x * other, self.y * other)


    def __truediv__(self, other):
        return Vector2(self.x / other, self.y / other)

    
    def __eq__(self, value: object) -> bool:
        return self.x == value.x and self.y == value.y


    def __str__(self) -> str:
        return f"({self.x},{self.y})"

    def __repr__(self) -> str:
        return f"({self.x},{self.y})"


class Object():
    def __init__(self, position: Vector2, size: Vector2, orientation: float, color: int):
        self.position = position
        self.size = size
        self.orientation = orientation
        self.color = color
        self.category = None


    def __repr__(self) -> str:
        return f"obj @ {self.position} -> [Area: {self.size.area()}, Solidity: {self.orientation}]"


class ObjectCategory():
    def __init__(self, size: Vector2, orientation: float, indicator_color: np.array) -> None:
        self.size = size
        self.orientation = orientation
        self.indicator_color = indicator_color
    
    
    def belongs(self, object: Object) -> bool:
        return abs(self.size.area() - object.size.area()) <= AREA_DIF_TOLERANCE and abs(self.orientation - object.orientation) <= ORIENTATION_DIF_TOLERANCE
    

    @staticmethod
    def from_obj(object: Object):
        return ObjectCategory(object.size, object.orientation, np.random.randint(256, size=(3)))


    def __repr__(self) -> str:
        return f"{self.size}"


class Event():
    def __init__(self, obj_category: str, category: str, current_pos: Vector2) -> None:
        self.obj_category = obj_category
        self.category = category
        self.current_pos = current_pos


class AppearanceEvent(Event):
    def __init__(self, obj_category: str, current_pos: Vector2) -> None:
        super().__init__(obj_category, "APPEARANCE", current_pos)
    

    def __repr__(self) -> str:
        return f"app @ {self.current_pos}"


class DisappearanceEvent(Event):
    def __init__(self, obj_category: str, current_pos: Vector2) -> None:
        super().__init__(obj_category, "DISAPPEARANCE", current_pos)
    

    def __repr__(self) -> str:
        return f"dis @ {self.current_pos}"


class MovementEvent(Event):
    def __init__(self, obj_category: str, initial_pos: Vector2, initial_timestep: int, current_pos: Vector2):
        super().__init__(obj_category, "MOVEMENT", current_pos)
        self.initial_pos = initial_pos
        self.initial_timestep = initial_timestep


    def get_vel(self, current_timestep: int) -> Vector2:
        return (self.current_pos - self.initial_pos) / float(current_timestep - self.initial_timestep)


    def get_state(self, current_timestep: int):
        """
        Given a timestep, returns the current position of this event and its average velocity
        """
        return self.current_pos, self.get_vel(current_timestep)


    def __repr__(self) -> str:
        return f"{self.initial_pos} -> {self.current_pos}"


class EpisodeTracker():
    def __init__(self, background_colors: np.array, relevant_cat_count: int=2, headless: bool=False) -> None:
        self.background_colors = background_colors

        self.headless = headless
        self.object_categories: list[ObjectCategory] = []
        self.tracked_objects: list[Object] = []
        self.tracked_events: list[Event] = []
        self.timestep: int = 0
        
        self.category_counts: dict[ObjectCategory, int] = {}
        self.total_category_appearances = 0
        self.relevant_cat_count = relevant_cat_count


    def process_frame(self, data: np.array) -> list[Event]:
        separated_bg = self.background_separation(data)
        objs = self.object_identification(separated_bg)
        objs = self.object_categorization(objs)
        transitions = self.object_tracking(objs)
        events = self.event_tracking(objs, transitions)
        filtered_events, valid_categories = self.filtering(events)

        self.tracked_objects = objs
        self.tracked_events = events

        self.timestep += 1

        # Render mode

        if not self.headless:
            for obj in objs:
                top_left = obj.position
                bottom_right = obj.position + obj.size
                cv2.rectangle(separated_bg, (top_left.x, top_left.y), (bottom_right.x, bottom_right.y), tuple(map(int, obj.category.indicator_color)), 2)
            for obj in transitions:
                cv2.arrowedLine(separated_bg, (obj.position.x, obj.position.y), (transitions[obj].position.x, transitions[obj].position.y), (255, 255, 255), 2)
            for event in filtered_events:
                if event.category == "APPEARANCE":
                    cv2.circle(separated_bg, (event.current_pos.x, event.current_pos.y), 4, (0, 255, 0), 2)
                elif event.category == "DISAPPEARANCE":
                    cv2.circle(separated_bg, (event.current_pos.x, event.current_pos.y), 4, (255, 0, 0), 2)
                else:
                    cv2.arrowedLine(separated_bg, (event.initial_pos.x, event.initial_pos.y), (event.current_pos.x, event.current_pos.y), (0, 0, 255), 2)
        return separated_bg, filtered_events, valid_categories


    def get_event_vel(self, event: MovementEvent) -> Vector2:
        return event.get_vel(self.timestep)


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

                orientation = cv2.contourArea(contour) / (bbox[2] * bbox[3])
                objects.append(Object(Vector2(bbox[0], bbox[1]), Vector2(bbox[2], bbox[3]), orientation, r))

        pprof.stop("ET_OBJID")

        return objects


    def update_category_importance(self, category: ObjectCategory):
        self.category_counts[category] += 1
        self.total_category_appearances += 1


    def get_category_importance(self, category: ObjectCategory) -> float:
        return self.category_counts[category] / self.total_category_appearances

    
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

                self.category_counts[new_cat] = 0
            self.update_category_importance(obj.category)

        pprof.stop("ET_OBJCAT")

        return data


    def get_closest_object(self, pos: Vector2, l: list[Object]) -> Object:
        best_distance = (l[0].position - pos).manhat_length()
        best_obj = l[0]
        
        for i in range(1, len(l)):
            new_distance = (l[i].position - pos).manhat_length()
            if new_distance < best_distance:
                best_distance = new_distance
                best_obj = l[i]
        
        return best_obj


    def put_closest_object(self, expected_positions, object_transitions_base, overfilled_objects, obj, dest_list):
        if len(dest_list) == 0:
            return

        closest_object = self.get_closest_object(expected_positions[obj], dest_list)
        if closest_object in object_transitions_base:
            if not closest_object in overfilled_objects:
                overfilled_objects.append(closest_object)

            object_transitions_base[closest_object].append(obj)
        else:
            object_transitions_base[closest_object] = [obj]


    def object_tracking(self, data: list[Object]) -> Dict[Object, Object]:
        pprof.start("ET_OBJTRK")

        expected_positions = {}
        for obj in self.tracked_objects:
            expected_positions[obj] = obj.position
            if obj in self.tracked_events and self.tracked_events[obj].category == "MOVEMENT":
                expected_positions[obj] += self.tracked_events[obj].get_vel(self.timestep)

        new_objs = copy.copy(data)
        object_transitions_base = {} # Dest obj to source objs that considered it closest obj
        overfilled_objects = []
        for obj in self.tracked_objects:
            self.put_closest_object(expected_positions, object_transitions_base, overfilled_objects, obj, new_objs)

        while len(overfilled_objects) > 0:
            obj = overfilled_objects.pop()
            closest_dist = (obj.position - object_transitions_base[obj][0].position).manhat_length()
            closest_obj = object_transitions_base[obj][0]
            for i in range(1, len(object_transitions_base[obj])):
                new_dist = (obj.position - object_transitions_base[obj][i].position).manhat_length()
                if new_dist < closest_dist:
                    closest_dist = new_dist
                    closest_obj = object_transitions_base[obj][i]
            
            new_objs.remove(obj)
            for i in range(0, len(object_transitions_base[obj])):
                if object_transitions_base[obj][i] != closest_obj:
                    self.put_closest_object(expected_positions, object_transitions_base, overfilled_objects, object_transitions_base[obj][i], new_objs)

            object_transitions_base[obj] = [closest_obj]
        
        object_transitions = {}
        for obj in object_transitions_base:
            object_transitions[object_transitions_base[obj][0]] = obj
        
        pprof.stop("ET_OBJTRK")

        return object_transitions


    def event_tracking(self, objects: list[Object], transitions: Dict[Object, Object]) -> list[Event]:
        pprof.start("ET_EVTRK")

        events = []

        dests = transitions.values()
        for obj in objects:
            if not obj in dests:
                events.append(AppearanceEvent(obj.category, obj.position))
        sources = transitions.keys()
        for obj in self.tracked_objects:
            if not obj in sources:
                events.append(DisappearanceEvent(obj.category, obj.position))
        
        events_by_pos: Dict[str, Event] = {}
        for event in self.tracked_events:
            events_by_pos[str(event.current_pos)] = event
        for obj in transitions:
            event = events_by_pos[str(obj.position)]
            if event.category == "APPEARANCE":
                events.append(MovementEvent(obj.category, event.current_pos, self.timestep - 1, transitions[obj].position))
            else:
                vel = transitions[obj].position - obj.position
                if (vel - event.get_vel(self.timestep)).manhat_length() < SPEED_DIF_TOLERANCE and vel.manhat_length() != 0:
                    event.current_pos = transitions[obj].position
                    events.append(event)
                else:
                    events.append(MovementEvent(obj.category, obj.position, self.timestep - 1, transitions[obj].position))

        pprof.stop("ET_EVTRK")

        return events


    def filtering(self, events: list[Event]) -> list[Event]:
        pprof.start("ET_FLTR")

        # Order categories
        categories = list(self.category_counts.keys())

        for i in range(1, len(categories)):
            key = categories[i]
            j = i - 1

            while j >= 0 and self.get_category_importance(key) > self.get_category_importance(categories[j]):
                categories[j + 1] = categories[j]
                j -= 1
            categories[j + 1] = key

        # Delete all non relevant categories and overly speedy movement events
        filtered_events = events.copy()
        valid_categories = categories[:self.relevant_cat_count]
        pos = 0
        while pos < len(filtered_events):
            if not filtered_events[pos].obj_category in valid_categories or (filtered_events[pos].category == "MOVEMENT" and filtered_events[pos].get_vel(self.timestep).manhat_length() > MAX_SPEED): 
                filtered_events.pop(pos)
            else:
                pos += 1

        pprof.stop("ET_FLTR")

        return filtered_events, valid_categories

    
    def finish_episode(self) -> None:
        self.tracked_objects = []
        self.tracked_events = []
        self.timestep = 0