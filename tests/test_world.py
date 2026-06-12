import unittest

from emergence.world import (
    Facility,
    FacilityType,
    World,
    build_default_world,
    chebyshev,
)


class TestWorld(unittest.TestCase):
    def test_default_layout_has_40_plus_facilities(self):
        world = build_default_world()
        self.assertGreaterEqual(len(world.facilities), 40)

    def test_default_layout_has_civic_buildings(self):
        world = build_default_world()
        for ftype in (
            FacilityType.TOWN_HALL,
            FacilityType.LIBRARY,
            FacilityType.POLICE_STATION,
            FacilityType.FARM,
            FacilityType.GRANARY,
        ):
            self.assertTrue(world.facilities_of(ftype), f"missing {ftype}")

    def test_nearest_returns_closest(self):
        world = World(10, 10)
        near = world.add_facility(Facility("Near", FacilityType.FARM, 1, 1))
        world.add_facility(Facility("Far", FacilityType.FARM, 9, 9))
        self.assertIs(world.nearest((0, 0), FacilityType.FARM), near)

    def test_step_towards_moves_one_king_step(self):
        world = World(10, 10)
        self.assertEqual(world.step_towards((0, 0), (5, 5)), (1, 1))
        self.assertEqual(world.step_towards((0, 0), (5, 0)), (1, 0))

    def test_step_towards_respects_bounds(self):
        world = World(10, 10)
        self.assertEqual(world.step_towards((0, 0), (-5, -5)), (0, 0))

    def test_chebyshev(self):
        self.assertEqual(chebyshev((0, 0), (3, 1)), 3)


if __name__ == "__main__":
    unittest.main()
