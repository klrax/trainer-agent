"""The agent's memory: a JSON file on disk.

Why this matters for the agent: the model forgets everything between conversations.
Anything the trainer must remember (your profile, past workouts, meals) lives HERE,
and the model reads it through tools. "State lives in the store, not in the chat."

Run:  uv run pytest -k s1
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

PROFILE_FIELDS = {"name", "sex", "age", "height_cm", "weight_kg", "activity", "goal",
                  "train_days", "experience", "injuries"}


class Store:
    """Words:
    - the file holds ONE dict: {"profile": None or dict, "workouts": [...], "meals": [...]}
    - a "workout" is {"date": "YYYY-MM-DD", "exercises": [{"name": str, "sets": [{"reps": int, "weight_kg": float}]}]}
    - a "meal" is {"date", "name", "kcal", "protein_g", "carbs_g", "fat_g"}

    __init__(path):
    - if the file exists: load it with json.loads(Path(path).read_text())
    - if not: start with {"profile": None, "workouts": [], "meals": []}   (don't write the file yet)

    Saving:
    - EVERY method that changes data saves the whole dict to the file before returning
    - Store(same_path) made later sees the saved data   (the test makes a 2nd Store on the same file)
    - Hint: write to "<path>.tmp" first, then os.replace(tmp, path). If the program crashes mid-write,
      the old file is still whole (an atomic write).

    Copies:
    - methods return COPIES (copy.deepcopy), so a caller changing the result can't change the store
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.exists():
            self.store = json.loads(self.path.read_text())
        else:
            self.store = {"profile": None, "workouts": [], "meals": []}   # file is created on first save

    def _save(self) -> None:
        """Atomic write: a crash mid-write leaves the old file whole, never a half-written one."""
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.store, indent=2))
        os.replace(tmp, self.path)

    def get_profile(self) -> dict[str, Any] | None:
        """- None if no profile was saved yet
        - otherwise a copy of the profile dict
        """
        return copy.deepcopy(self.store["profile"])

    def update_profile(self, **fields: Any) -> dict[str, Any]:
        """- merges fields into the profile (starts from {} if there is none yet)
        - s.update_profile(name="Kat", age=30) -> {"name": "Kat", "age": 30}
        - then s.update_profile(age=31)        -> {"name": "Kat", "age": 31}   (name is kept)
        - a field not in PROFILE_FIELDS raises ValueError, and nothing is changed
        - saves, then returns a copy of the whole profile
        """
        unknown = set(fields) - PROFILE_FIELDS          # check ALL fields before changing anything
        if unknown:
            raise ValueError(f"unknown profile fields: {sorted(unknown)}")
        if self.store["profile"] is None:
            self.store["profile"] = {}
        self.store["profile"].update(copy.deepcopy(fields))
        self._save()
        return self.get_profile()

    def log_workout(self, date: str, exercises: list[dict[str, Any]]) -> None:
        """- appends {"date": date, "exercises": exercises} to "workouts", then saves
        - exercises == [] raises ValueError
        """
        if not exercises:
            raise ValueError("a workout needs at least one exercise")
        self.store["workouts"].append({"date": date, "exercises": copy.deepcopy(exercises)})
        self._save()

    def exercise_history(self, name: str, limit: int = 5) -> list[dict[str, Any]]:
        """One entry per workout that contains this exercise: {"date": ..., "sets": [...]}.

        - the name matches in any case: "SQUAT" finds "Squat" and "squat"
        - order: by date, OLDEST first (workouts can be logged out of order, so sort by "date")
        - only the newest `limit` entries:  limit=2 -> the last 2 of the sorted list
        - no match -> []
        - limit < 1 raises ValueError
        """
        if limit < 1:
            raise ValueError("limit must be >= 1")
        history = []
        for workout in sorted(self.store["workouts"], key=lambda w: w["date"]):
            for exercise in workout["exercises"]:
                if exercise["name"].lower() == name.lower():
                    history.append({"date": workout["date"], "sets": copy.deepcopy(exercise["sets"])})
        return history[-limit:]

    def log_meal(self, date: str, name: str, kcal: float, protein_g: float,
                 carbs_g: float, fat_g: float) -> dict[str, Any]:
        """- appends the meal dict (keys: date, name, kcal, protein_g, carbs_g, fat_g), saves, returns a copy
        - any of kcal / protein_g / carbs_g / fat_g < 0 raises ValueError
        """
        meal = {"date": date, "name": name, "kcal": kcal, "protein_g": protein_g,
                "carbs_g": carbs_g, "fat_g": fat_g}
        for key in ("kcal", "protein_g", "carbs_g", "fat_g"):
            if meal[key] < 0:
                raise ValueError(f"{key} must be >= 0, got {meal[key]}")
        self.store["meals"].append(meal)
        self._save()
        return dict(meal)

    def meals_on(self, date: str) -> list[dict[str, Any]]:
        """- copies of the meals with this date, in the order they were logged
        - none -> []
        """
        return [dict(m) for m in self.store["meals"] if m["date"] == date]
